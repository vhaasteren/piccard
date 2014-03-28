from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os as os
import glob
import sys

from .piccard import *
from . import pytwalk                  # Internal module
from . import pydnest                  # Internal module
from . import rjmcmchammer as rjemcee  # Internal module
from . import PTMCMC_generic as ptmcmc

try:    # Fall back to internal emcee implementation
    import emcee as emceehammer
    emcee = emceehammer
except ImportError:
    import mcmchammer as mcmchammer
    emcee = mcmchammer

class pulsarNoiseLL(object):
    """
    This class simplifies sampling from a pulsar noise-only distribution, by
    avoiding having to go over the whole signal dictionary
    """
    residuals = None        # The residuals
    Nvec = None             # The full noise vector
    vNvec = []              # Which residuals a parameter affects (varying)
    fNvec = []              # Which residuals a parameter affects (fixed)
    vis_efac = []           # Is it an efac, or an equad?
    fis_efac = []           # Is it an efac, or an equad?
    pmin = None             # The minimum value for the parameters (varying)
    pmax = None             # The maximum value for the parameters (varying)
    pstart = None           # The start value for the parameters (varying)
    pwidth = None           # The width value for the parameters (varying)
    pindex = None           # Index of the parameter
    fval = None             # The current value for the parameters (fixed)

    def __init__(self, residuals):
        """
        @param residuals:   Initialise the residuals we'll work with
        """
        self.vNvec = []
        self.fNvec = []
        self.vis_efac = []
        self.fis_efac = []

        self.residuals = residuals
        self.Nvec = np.zeros(len(residuals))
        self.pmin = np.zeros(0)
        self.pmax = np.zeros(0)
        self.pstart = np.zeros(0)
        self.pwidth = np.zeros(0)
        self.fval = np.zeros(0)
        self.pindex = np.zeros(0, dtype=np.int)

    def addSignal(self, Nvec, is_efac, pmin, pmax, pstart, pwidth, index, fixed=False):
        """
        Add a efac/equad signal to this model
        @param Nvec:    To which residuals does this signal apply?
        @param is_efac: Is this signal an efac, or an equad?
        @param pmin:    Minimum of the prior domain
        @param pmax:    Maximum of the prior domain
        @param fixed:   Whether or not we vary this parameter
        """
        if not fixed:
            self.vNvec.append(Nvec)
            self.vis_efac.append(is_efac)
            self.pmin = np.append(self.pmin, [pmin])
            self.pmax = np.append(self.pmax, [pmax])
            self.pstart = np.append(self.pstart, [pstart])
            self.pwidth = np.append(self.pwidth, [pwidth])
            self.pindex = np.append(self.pindex, index)
        else:
            self.fNvec.append(Nvec)
            self.fis_efac.append(is_efac)
            self.fval = np.append(self.fval, pstart)

    def loglikelihood(self, parameters):
        """
        @param parameters:  the vector with model parameters
        """
        self.Nvec[:] = 0
        for ii, par in enumerate(parameters):
            if self.vis_efac[ii]:
                self.Nvec += par**2 * self.vNvec[ii]
            else:
                self.Nvec += 10**(2*par) * self.vNvec[ii]

        for ii, par in enumerate(self.fval):
            if self.fis_efac[ii]:
                self.Nvec += par**2 * self.fNvec[ii]
            else:
                self.Nvec += 10**(2*par) * self.fNvec[ii]

        return -0.5 * np.sum(self.residuals**2 / self.Nvec) - \
                0.5 * np.sum(np.log(self.Nvec))

    def logprior(self, parameters):
        bok = -np.inf
        if np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax):
            bok = 0

        return bok

    def dimensions(self):
        if len(self.vNvec) != len(self.vis_efac):
            raise ValueError("dimensions not set correctly")

        return len(self.vNvec)


class pulsarPSDLL(object):
    """
    Like pulsarNoiseLL, but for the power spectrum coefficients. Expects the
    matrix to be diagonal. Is always a function of amplitude and spectral index
    """
    a = None
    pmin = None
    pmax = None
    pstart = None
    pwidth = None
    pindex = None

    def __init__(self, a, freqs, Tmax, pmin, pmax, pstart, pwidth, index, bvary):
        """
        @param a:       The Fourier components of the signal
        @param freqs:   The frequencies of the signal
        @param pmin:    Minimum value of the two parameters
        """

        self.a = a
        self.freqs = freqs
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.pindex = index
        self.bvary = bvary
        self.Tmax = Tmax

    def loglikelihood(self, parameters):
        pars = self.pstart.copy()
        pars[self.bvary] = parameters

        freqpy = self.freqs * pic_spy
        pcdoubled = ((10**(2*pars[0])) * pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-pars[1])

        return -0.5 * np.sum(self.a**2 / pcdoubled) - 0.5*np.sum(np.log(pcdoubled))

    def logprior(self, parameters):
        bok = -np.inf
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def dimensions(self):
        return len(self.pmin)



def gibbs_sample_a(self):
    """
    Assume that all the noise parameters have been set (N, Phi, Theta). Given
    that, return a sample from the coefficient/timing model parameters

    @return: list of coefficients/timing model parameters per pulsar
    """

    a = []
    for ii, psr in enumerate(self.ptapsrs):
        zindex = np.sum(self.npz[:ii])
        nzs = self.npz[ii]
        mindex = np.sum(self.npm[:ii])
        nms = self.npm[ii]
        findex = np.sum(self.npf[:ii])
        nfs = self.npf[ii]
        fdmindex = np.sum(self.npfdm[:ii])
        nfdms = self.npfdm[ii]
        uindex = np.sum(self.npu[:ii])
        npus = self.npu[ii]

        # Make ZNZ
        ZNZ = np.dot(psr.Zmat.T, ((1.0/psr.Nvec) * psr.Zmat.T).T)

        di = np.diag_indices(ZNZ.shape[0])

        indsp = range(zindex+nms, zindex+nms+nfs)
        indst = range(zindex+nms+nfs, zindex+nms+nfs+nfdms)
        indsu = range(zindex+nms+nfs+nfdms, zindex+nms+nfs+nfdms+npus)

        Sigma = ZNZ.copy()
        Sigma[indsp, indsp] += 1.0 / self.Phivec[findex:findex+nfs]

        if nfdms > 0:
            # We have DM variations
            Sigma[indst, indst] += 1.0 / self.Thetavec[fdmindex:fdmindex+nfdms]

        if nzs == nms + nfs + nfdms + npus:
            # We have correlated equads
            Sigma[indsu, indsu] += 1.0 / psr.Jvec

        # ahat is the slice ML value for the coefficients. Need ENx
        ENx = np.dot(psr.Zmat.T, psr.detresiduals / psr.Nvec)

        try:
            #raise np.linalg.LinAlgError("")
            # Use a QR decomposition for the inversions
            Qs,Rs = sl.qr(Sigma) 

            Qsb = np.dot(Qs.T, np.eye(Sigma.shape[0])) # computing Q^T*b (project b onto the range of A)
            Sigi = sl.solve(Rs,Qsb) # solving R*x = Q^T*b
            
            # Ok, we've got the inverse... now what? Do SVD?
            U, s, Vt = sl.svd(Sigi)
            Li = U * np.sqrt(s)

            ahat = np.dot(Sigi, ENx)

        except np.linalg.LinAlgError:
            print "ERROR in QR decomp"
            #"""
            try:
                cfL = sl.cholesky(Sigma, lower=True)
                cf = (cfL, True)

                # Calculate the inverse Cholesky factor (can we do this faster?)
                cfLi = sl.cho_factor(cfL, lower=True)
                Li = sl.cho_solve(cfLi, np.eye(Sigma.shape[0]))

                ahat = sl.cho_solve(cf, ENx)
            #"""
            except np.linalg.LinAlgError:
            #if True:
                U, s, Vt = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                Sigi = np.dot(U, np.dot(np.diag(1.0/s), Vt))
                Li = U * (1.0 / np.sqrt(s))
                #Li = np.dot(U, np.diag(1.0 / np.sqrt(s)))

                ahat = np.dot(Sigi, ENx)
        except ValueError:
            print "WTF?"
            print Sigma
            np.savetxt("temp.txt", Sigma)
            raise

        # Get a sample from the coefficient distribution
        aadd = np.dot(Li, np.random.randn(Li.shape[0]))
        addcoefficients = ahat + aadd

        psr.gibbscoefficients = addcoefficients.copy()
        psr.gibbscoefficients[:psr.Mmat.shape[1]] = np.dot(psr.tmpConv, \
                addcoefficients[:psr.Mmat.shape[1]])

        # addres = np.dot(psr.Zmat, aadd)
        #print "Addres: ", addres

        # We really do not care about the tmp's at this point. Save them
        # separately
        a.append(psr.gibbscoefficients)
        #a.append(psr.gibbscoefficients[psr.Mmat.shape[1]:])
        psr.gibbsresiduals = psr.detresiduals - np.dot(psr.Zmat, addcoefficients)

    return a


def gibbs_sample_N(self, curpars):
    # Given the values for a, which have been subtracted from the residuals, we
    # now need to find the PSD and noise coefficients.
    newpars = curpars.copy()

    for ii, psr in enumerate(self.ptapsrs):
        pnl = pulsarNoiseLL(psr.gibbsresiduals)

        # Add the signals
        for ss, signal in enumerate(self.ptasignals):
            pstart = curpars[signal['parindex']]
            if signal['pulsarind'] == ii and signal['stype'] == 'efac':
                # We have a winner: add this signal
                pnl.addSignal(signal['Nvec'], True, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        signal['parindex'], fixed=(not signal['bvary'][0]))
            elif signal['pulsarind'] == ii and signal['stype'] == 'equad':
                # We have a winner: add this signal
                pnl.addSignal(signal['Nvec'], False, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        signal['parindex'], fixed=(not signal['bvary'][0]))


        temp = pnl.loglikelihood(pnl.pstart)

        #"""
        # Run a tiny MCMC of one correlation length, and return the parameters
        ndim = pnl.dimensions()
        cov = np.diag(pnl.pwidth**2)
        p0 = pnl.pstart
        sampler = ptmcmc.PTSampler(ndim, pnl.loglikelihood, pnl.logprior, cov=cov, \
                outDir='./gibbs-chains/', verbose=False, nowrite=True)

        steps = ndim*20
        sampler.sample(p0, steps, thin=1, burn=10)

        newpars[pnl.pindex] = sampler._chain[steps-1,:]
        #"""

    return newpars



def gibbs_sample_Phi(self, a, curpars):
    """
    Same as gibbs_sample_N, but for the phi frequency components

    """
    newpars = curpars.copy()

    for ii, psr in enumerate(self.ptapsrs):
        for ss, signal in enumerate(self.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'spectrum':
                nms = self.npm[ii]
                # Loop over the frequencies
                for jj in range(signal['ntotpars']):
                    # We can sample directly from this distribution.
                    # Prior domain:
                    pmin = signal['pmin'][jj]
                    pmax = signal['pmax'][jj]
                    rhomin = 10**pmin
                    rhomax = 10**pmax
                    tau = 0.5*np.sum(a[ii][nms+2*jj:nms+2*jj+2]**2)

                    # Draw samples between rhomax and rhomin, according to
                    # an exponential distribution
                    scale = 1 - np.exp(tau*(1.0/rhomax-1.0/rhomin))
                    eta = np.random.rand(1) * scale
                    rhonew = -tau / (np.log(1-eta)-tau/rhomax)

                    newpars[signal['parindex']+jj] = np.log10(rhonew)

            elif signal['pulsarind'] == ii and signal['stype'] == 'powerlaw':
                bvary = signal['bvary']
                pindex = signal['parindex']
                pstart = signal['pstart']
                pmin = signal['pmin']
                pmax = signal['pmax']
                pwidth = signal['pwidth']
                Tmax = signal['Tmax']
                nms = self.npm[ii]
                nfs = self.npf[ii]

                ndim = np.sum(bvary)
                if ndim > 0:
                    pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                    psd = pulsarPSDLL(a[ii][nms:nms+nfs], psr.Ffreqs, Tmax, pmin, \
                            pmax, pstart, pwidth, pindex, bvary)

                    cov = np.diag(pwidth[bvary]**2)
                    p0 = pstart[bvary]
                    sampler = ptmcmc.PTSampler(ndim, psd.loglikelihood, psd.logprior, cov=cov, \
                            outDir='./gibbs-chains/', verbose=False, nowrite=True)

                    steps = ndim*40
                    sampler.sample(p0, steps, thin=1, burn=10)

                    newpars[psd.pindex:psd.pindex+ndim] = \
                            sampler._chain[steps-1,:]

            elif signal['pulsarind'] == ii and signal['stype'] == 'dmspectrum':
                nms = self.npm[ii]
                nfs = self.npf[ii]
                nfdms = self.npfdm[ii]
                # Loop over the frequencies
                for jj in range(signal['ntotpars']):
                    # We can sample directly from this distribution.
                    # Prior domain:
                    pmin = signal['pmin'][jj]
                    pmax = signal['pmax'][jj]
                    rhomin = 10**pmin
                    rhomax = 10**pmax
                    tau = 0.5*np.sum(a[ii][nms+nfs+2*jj:nms+nfs+2*jj+2]**2)

                    # Draw samples between rhomax and rhomin, according to
                    # an exponential distribution
                    scale = 1 - np.exp(tau*(1.0/rhomax-1.0/rhomin))
                    eta = np.random.rand(1) * scale
                    rhonew = -tau / (np.log(1-eta)-tau/rhomax)

                    newpars[signal['parindex']+jj] = np.log10(rhonew)
            elif signal['pulsarind'] == ii and signal['stype'] == 'dmpowerlaw':
                bvary = signal['bvary']
                pindex = signal['parindex']
                pstart = signal['pstart']
                pmin = signal['pmin']
                pmax = signal['pmax']
                pwidth = signal['pwidth']
                Tmax = signal['Tmax']
                nms = self.npm[ii]
                nfs = self.npf[ii]
                nfdms = self.npfdm[ii]

                ndim = np.sum(bvary)
                if ndim > 0:
                    pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                    psd = pulsarPSDLL(a[ii][nms+nfs:nms+nfs+nfdms], psr.Ffreqs, Tmax, pmin, \
                            pmax, pstart, pwidth, pindex, bvary)

                    cov = np.diag(pwidth[bvary]**2)
                    p0 = pstart[bvary]
                    sampler = ptmcmc.PTSampler(ndim, psd.loglikelihood, psd.logprior, cov=cov, \
                            outDir='./gibbs-chains/', verbose=False, nowrite=True)

                    steps = ndim*40
                    sampler.sample(p0, steps, thin=1, burn=10)

                    newpars[psd.pindex:psd.pindex+ndim] = \
                            sampler._chain[steps-1,:]

    return newpars



def gibbs_sample_J(self, a, curpars):
    """
    Same as gibbs_sample_N, but now for the pulse jitter. We are actually
    running 1D MCMC chains here, but for now that is easier (and faster?) than
    implementing a direct 1D sampler based on interpolations
    """
    newpars = curpars.copy()

    for ii, psr in enumerate(self.ptapsrs):
        zindex = np.sum(self.npz[:ii])
        nzs = self.npz[ii]
        mindex = np.sum(self.npm[:ii])
        nms = self.npm[ii]
        findex = np.sum(self.npf[:ii])
        nfs = self.npf[ii]
        fdmindex = np.sum(self.npfdm[:ii])
        nfdms = self.npfdm[ii]
        uindex = np.sum(self.npu[:ii])
        npus = self.npu[ii]

        # Add the signals
        for ss, signal in enumerate(self.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'jitter' and \
                    signal['bvary'][0] == True:
                # We have a jitter parameter
                inds = zindex+nms+nfs+nfdms
                inde = zindex+nms+nfs+nfdms+npus

                # To which avetoas does this 'jitter' apply?
                select = np.array(signal['Jvec'], dtype=np.bool)
                selall = np.ones(np.sum(select))

                res = a[ii][inds:inde][select]
                pstart = curpars[signal['parindex']]

                # Set up the conditional log-likelihood
                pnl = pulsarNoiseLL(res)
                pnl.addSignal(selall, False, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        signal['parindex'], fixed=(not signal['bvary'][0]))

                # Prepare the sampler
                temp = pnl.loglikelihood(pnl.pstart)
                ndim = pnl.dimensions()
                cov = np.diag(pnl.pwidth**2)
                p0 = pnl.pstart
                sampler = ptmcmc.PTSampler(ndim, pnl.loglikelihood, pnl.logprior, cov=cov, \
                        outDir='./gibbs-chains/', verbose=False, nowrite=True)

                # Run a tiny MCMC of one correlation length, and return the parameters

                steps = ndim*40
                sampler.sample(p0, steps, thin=1, burn=10)

                newpars[pnl.pindex] = sampler._chain[steps-1,:]

    return newpars


def gibbsQuantities(self, parameters):
    npsrs = len(self.ptapsrs)

    # MARK A

    self.setPsrNoise(parameters)

    # MARK B

    self.constructPhiAndTheta(parameters, phimat=False)

    # MARK ??
    if self.haveDetSources:
        self.updateDetSources(parameters)



def RunGibbs(likob, steps, chainsdir):
    """
    Run a gibbs sampler on a, for now, simplified version of the likelihood.
    Only allows spectrum red noise, and should only be done on a single pulsar

    Parameters are grouped into three categories:
    1) a, the Fourier coefficients and timing model parameters
    2) N, the white noise parameters
    3) Phi, the red noise PSD coefficients
    """
    if not likob.likfunc in ['gibbs1', 'gibbs2']:
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    # Clear the file for writing
    chainfilename = chainsdir + '/chain_1.txt'
    chainfile = open(chainfilename, 'w')
    chainfile.close()

    # Dump samples to file every dumpint steps (no thinning)
    dumpint = 100

    ndim = likob.dimensions
    pars = likob.pstart.copy()
    ncoeffs = np.sum(likob.npz)

    #chain = np.zeros((steps, ndim))
    #chain2 = np.zeros((steps, ncoeffs))
    samples = np.zeros((min(dumpint, steps), ndim+ncoeffs))
    stepind = 0

    for step in range(steps):
        doneIteration = False
        iter = 0

        # Start with calculating the required likelihood quantities
        gibbsQuantities(likob, pars)

        while not doneIteration:
            try:
                # Generate new coefficients
                a = gibbs_sample_a(likob)

                samples[stepind, ndim:] = np.hstack(a)

                doneIteration = True

            except np.linalg.LinAlgError:
                # Why does SVD sometimes not converge?
                # Try different values...
                iter += 1

                if iter > 100:
                    print "WARNING: numpy.linalg problems"

            # Generate new white noise parameers
            pars = gibbs_sample_N(likob, pars)
            #print "N:", pars

            # Generate new red noise parameters
            pars = gibbs_sample_Phi(likob, a, pars)
            #print "P:", pars

            # Generate new correlated equad/jitter parameters
            pars = gibbs_sample_J(likob, a, pars)

        samples[stepind, :ndim] = pars

        stepind += 1
        # Write to file if necessary
        if (stepind % dumpint == 0 or step == steps-1):
            nwrite = dumpint

            # Check how many samples we are writing
            if step == steps-1:
                nwrite = stepind

            # Open the file in append mode
            chainfile = open(chainfilename, 'a+')
            for jj in range(nwrite):
                chainfile.write('0.0\t  0.0\t  0.0\t')
                chainfile.write('\t'.join(["%.17e"%\
                        (samples[jj,kk]) for kk in range(ndim+ncoeffs)]))
                chainfile.write('\n')
            chainfile.close()
            stepind = 0

        percent = (step * 100.0 / steps)
        sys.stdout.write("\rGibbs: %d%%" %percent)
        sys.stdout.flush()

    sys.stdout.write("\n")

    """
    #extracols = np.zeros((chain.shape[0], 3))
    #savechain = np.append(extracols, chain, axis=1)
    savechain = np.zeros((chain.shape[0], chain.shape[1]+3))
    savechain[:, 3:] = chain
    #np.savetxt(chainsdir + '/chain_1.txt.orig', chain)
    np.savetxt(chainsdir + '/chain_1.txt', savechain)

    sys.stdout.write("\n")
    np.savetxt(chainsdir+'/chain2_1.txt', chain2)
    #return chain
    """
