from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os as os
import glob
import sys

from .piccard import *
from . import PTMCMC_generic as ptmcmc


"""
This file implements a blocked Gibbs sampler. Gibbs sampling is a special case
of Metropolis-Hastings, and in Pulsar Timing data analysis it can be used to
increase the mixing rate of the chain. We still use the PAL/PAL2 version of
PTMCMC_generic to do the actual MCMC steps, but the steps are performed in
parameter blocks.
"""

class pulsarNoiseLL(object):
    """
    This class represents the likelihood function in the block with
    white-noise-only parameters.
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
    pindex = None           # Index of the parameters
    psrindex = None         # Index of the pulsar
    fval = None             # The current value for the parameters (fixed)
    sampler = None          # Copy of the MCMC sampler
    maskJvec = None         # To which avetoas does this Jvec apply? (only jitter)

    def __init__(self, residuals, psrindex, maskJvec=None):
        """
        @param residuals:   Initialise the residuals we'll work with
        @param psrindex:    Index of the pulsar this noise applies to
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

        self.psrindex = psrindex
        self.maskJvec = maskJvec


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

    def setSampler(self, sampler):
        self.sampler = sampler

    def setNewData(self, residuals):
        self.residuals = residuals

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
    bvary = None
    sampler = None
    gindices = None
    psrindex = None

    def __init__(self, a, freqs, Tmax, pmin, pmax, pstart, pwidth, index, \
            psrindex, gindices, bvary):
        """
        @param a:           The Fourier components of the signal
        @param freqs:       The frequencies of the signal
        @param pmin:        Minimum value of the parameters prior domain
        @param pmax:        Maximum value of the parameters prior domain
        @param pwidth:      Initial step-size of this parameter
        @param index:       First index in the parameter array of this signal
        @param psrindex:    Index of the pulsar this applies to
        @param gindices:    The indices of the relevant Gibbs-parameters
        @param bvary:       Which parameters are actually varying
        """

        self.a = a
        self.freqs = freqs
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.pindex = index
        self.psrindex = psrindex
        self.gindices = gindices
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

    def setNewData(self, a):
        self.a = a

    def setSampler(self, sampler):
        self.sampler = sampler

    def dimensions(self):
        return np.sum(self.bvary)



class corrPSDLL(object):
    """
    Like pulsarPSDLL, but now for correlated signals.
    """
    b = None
    pmin = None
    pmax = None
    pstart = None
    pwidth = None
    pindex = None
    bvary = None
    sampler = None
    gindices = None
    psrindex = None

    Scor_inv = None
    Scor_ldet = None

    def __init__(self, b, freqs, Tmax, pmin, pmax, pstart, pwidth, index, \
            psrindex, gindices, bvary):
        """
        @param b:           The Fourier components of the signal
                            (list for pulsars)
        @param freqs:       The frequencies of the signal
        @param pmin:        Minimum value of the parameters prior domain
        @param pmax:        Maximum value of the parameters prior domain
        @param pwidth:      Initial step-size of this parameter
        @param index:       First index in the parameter array of this signal
        @param psrindex:    Index of the pulsar this applies to
        @param gindices:    The indices of the relevant Gibbs-parameters
        @param bvary:       Which parameters are actually varying
        """

        self.b = b
        self.freqs = freqs
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.pindex = index
        self.psrindex = psrindex
        self.gindices = gindices
        self.bvary = bvary
        self.Tmax = Tmax

    def loglikelihood(self, parameters):
        pars = self.pstart.copy()
        pars[self.bvary] = parameters

        freqpy = self.freqs * pic_spy
        pcdoubled = ((10**(2*pars[0])) * pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-pars[1])

        xi2 = 0
        ldet = 0
        # For every frequency, do an H&D matrix inverse inner-product
        for ii, pc in enumerate(pcdoubled):
            msk = self.freqmask[:, ii]
            xi2 += np.dot(self.freqb[msk,ii], np.dot(\
                    self.Scor_inv[msk,:][:,msk], self.freqb[msk,ii])) / pc
            ldet += self.Scor_ldet + np.sum(msk)*np.log(pc)

        #print "shapes:", self.freqs.shape, msk.shape, self.freqmask.shape
        #print "more:", self.freqb[msk,ii].shape, self.Scor_inv[msk,:][:,msk].shape,\
        #        self.freqb[msk,ii].shape
        return -0.5 * xi2 - 0.5 * ldet


    def logprior(self, parameters):
        bok = -np.inf
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def setNewData(self, b, Scor_inv, Scor_ldet):
        """
        Set new data at the beginning of each small MCMC chain

        @param b:           New Gibbs modes
        @param Scor_inv:    New pulsar correlation inverse
        @param Scor_ldet:   New pulsar correlation log-det
        """
        self.b = b
        self.Scor_inv = Scor_inv
        self.Scor_ldet = Scor_ldet

        self.numPsrFreqs = []
        self.freqmask = np.zeros((len(self.b), len(self.freqs)), dtype=np.bool)
        self.freqb = np.zeros((len(self.b), len(self.freqs)))

        # Make masks that show which pulsar has how many frequencies
        for ii in range(len(self.b)):
            self.numPsrFreqs.append(len(self.b[ii]))
            self.freqmask[ii,:len(self.b[ii])] = True

        for ii in range(len(self.b)):
            for jj in range(len(self.freqs)):
                if self.freqmask[ii, jj]:
                    self.freqb[ii, jj] = self.b[ii][jj]


    def setSampler(self, sampler):
        self.sampler = sampler

    def dimensions(self):
        return np.sum(self.bvary)





class pulsarDetLL(object):
    """
    Like pulsarNoiseLL, but for all other deterministic sources, like GW BWM
    """
    def __init__(self, likob, allpars, mask, pmin, pmax, pstart, pwidth, bvary):
        """
        @param likob:   The likelihood object, containing all the pulsars etc.
        @param allpars: Array with _all_ parameters, not just the det-sources
        @param mask:    Boolean mask that selects the det-source parameters
        @param pmin:    Minimum bound of all det-source parameters
        @param pmax:    Maximum bound of all det-source parameters
        @param pstart:  Start-parameter of all ''
        @param pwidth:  Width-parameter of all ''
        @param bvary:   Whether or not we vary the parameters (also with mask?)
        """

        self.likob = likob
        self.allpars = allpars.copy()
        self.mask = mask
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.bvary = bvary

    def loglikelihood(self, parameters):
        """
        Calculate the conditional likelihood, as a function of the deterministic
        model parameters. This is an all-pulsar likelihood
        """
        self.allpars[self.mask] = parameters
        self.likob.updateDetSources(self.allpars)

        xi2 = 0
        ldet = 0

        for psr in self.likob.ptapsrs:
            xi2 += np.sum((psr.detresiduals-psr.gibbssubresiduals)**2/psr.Nvec)
            ldet += np.sum(np.log(psr.Nvec))

        return -0.5*xi2 - 0.5*ldet


    def logprior(self, parameters):
        """
        Only return 0 when the parameters are within the prior domain
        """
        bok = -np.inf
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def setSampler(self, sampler):
        self.sampler = sampler

    def dimensions(self):
        return np.sum(self.bvary)


def gibbs_prepare_loglik_N(likob, curpars):
    """
    Prepares the likelihood objects for white noise of all pulsars

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """

    loglik_N = []
    for ii, psr in enumerate(likob.ptapsrs):
        tempres = np.zeros(len(psr.residuals))
        loglik_N.append(pulsarNoiseLL(tempres, ii))
        pnl = loglik_N[-1]

        # Add the signals
        for ss, signal in enumerate(likob.ptasignals):
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

        pnl.setSampler(sampler)

        #steps = ndim*20
        #sampler.sample(p0, steps, thin=1, burn=10)

    return loglik_N

def gibbs_sample_loglik_N(likob, curpars, loglik_N):
    """
    @param likob:       the full likelihood object
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_N:    List of prepared likelihood/samplers for the noise
    """
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pnl in loglik_N:
        # Update the sampler with the new Gibbs residuals
        psr = likob.ptapsrs[pnl.psrindex]
        pnl.setNewData(psr.gibbsresiduals)

        ndim = pnl.dimensions()
        p0 = newpars[pnl.pindex]

        steps = ndim*20
        pnl.sampler.sample(p0, steps, thin=1, burn=10)

        newpars[pnl.pindex] = pnl.sampler._chain[steps-1,:]

    return newpars


def gibbs_prepare_loglik_J(likob, curpars):
    """
    Prepares the likelihood objects for correlated equad of all pulsars

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """
    loglik_J = []

    for ii, psr in enumerate(likob.ptapsrs):
        #zindex = np.sum(likob.npz[:ii])
        zindex = 0          # Per pulsar basis
        nzs = likob.npz[ii]
        mindex = np.sum(likob.npm[:ii])
        nms = likob.npm[ii]
        findex = np.sum(likob.npf[:ii])
        nfs = likob.npf[ii]
        fdmindex = np.sum(likob.npfdm[:ii])
        nfdms = likob.npfdm[ii]
        uindex = np.sum(likob.npu[:ii])
        npus = likob.npu[ii]

        # Figure out where to start counting
        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms
        if 'rednoise' in likob.gibbsmodel:
            ntot += nfs
        if 'dm' in likob.gibbsmodel:
            ntot += nfdms


        # Add the signals
        for ss, signal in enumerate(likob.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'jitter' and \
                    signal['bvary'][0] == True:
                # We have a jitter parameter
                inds = ntot
                inde = ntot+npus

                # To which avetoas does this 'jitter' apply?
                select = np.array(signal['Jvec'], dtype=np.bool)
                selall = np.ones(np.sum(select))

                res = np.zeros(np.sum(select))
                pstart = curpars[signal['parindex']]

                # Set up the conditional log-likelihood
                loglik_J.append(pulsarNoiseLL(res, ii, maskJvec=signal['Jvec']))
                pnl = loglik_J[-1]
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

                pnl.setSampler(sampler)

                # Run a tiny MCMC of one correlation length, and return the parameters
                #steps = ndim*40
                #sampler.sample(p0, steps, thin=1, burn=10)

                #newpars[pnl.pindex] = sampler._chain[steps-1,:]


    return loglik_J

def gibbs_sample_loglik_J(likob, a, curpars, loglik_J):
    """
    @param likob:       the full likelihood object
    @param a:       list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_J:    List of prepared likelihood/samplers for the correlated noise
    """
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pnl in loglik_J:
        # Update the sampler with the new Gibbs residuals
        ii = pnl.psrindex
        psr = likob.ptapsrs[ii]

        # Indices
        nms = likob.npm[ii]
        nfs = likob.npf[ii]
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        # Figure out where the indices start
        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms
        if 'rednoise' in likob.gibbsmodel:
            ntot += nfs
        if 'dm' in likob.gibbsmodel:
            ntot += nfdms


        # Gibbs parameter selectors
        inds = ntot
        inde = ntot+npus

        # Set the new 'residuals'
        select = np.array(pnl.maskJvec, dtype=np.bool)
        res = a[ii][inds:inde][select]
        pnl.setNewData(res)

        # Number of dimensions really is 'just' 1, but do it anyway
        ndim = pnl.dimensions()
        p0 = newpars[pnl.pindex]

        steps = ndim*40
        pnl.sampler.sample(p0, steps, thin=1, burn=10)

        newpars[pnl.pindex] = pnl.sampler._chain[steps-1,:]

    return newpars


def gibbs_prepare_loglik_Phi(likob, curpars):
    """
    Prepares the likelihood objects for the power-spectra of all pulsars

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """

    loglik_Phi = []
    for ii, psr in enumerate(likob.ptapsrs):
        for ss, signal in enumerate(likob.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'powerlaw' and \
                    signal['corr'] == 'single':
                bvary = signal['bvary']
                pindex = signal['parindex']
                pstart = signal['pstart']
                pmin = signal['pmin']
                pmax = signal['pmax']
                pwidth = signal['pwidth']
                Tmax = signal['Tmax']
                nms = likob.npm[ii]
                nfs = likob.npf[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms

                ndim = np.sum(bvary)
                if ndim > 0:
                    pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                    loglik_Phi.append(pulsarPSDLL(np.zeros(nfs), \
                            psr.Ffreqs, Tmax, pmin, pmax, pstart, pwidth, \
                            pindex, ii, np.arange(ntot, ntot+nfs), bvary))
                    psd = loglik_Phi[-1]

                    cov = np.diag(pwidth[bvary]**2)
                    p0 = pstart[bvary]
                    sampler = ptmcmc.PTSampler(ndim, psd.loglikelihood, psd.logprior, cov=cov, \
                            outDir='./gibbs-chains/', verbose=False, nowrite=True)

                    psd.setSampler(sampler)

                    #steps = ndim*40
                    #sampler.sample(p0, steps, thin=1, burn=10)

                    #newpars[psd.pindex:psd.pindex+ndim] = \
                    #        sampler._chain[steps-1,:]

            elif signal['pulsarind'] == ii and signal['stype'] == 'dmpowerlaw':
                bvary = signal['bvary']
                pindex = signal['parindex']
                pstart = signal['pstart']
                pmin = signal['pmin']
                pmax = signal['pmax']
                pwidth = signal['pwidth']
                Tmax = signal['Tmax']
                nms = likob.npm[ii]
                nfs = likob.npf[ii]
                nfdms = likob.npfdm[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms
                if 'rednoise' in likob.gibbsmodel:
                    ntot += nfs

                ndim = np.sum(bvary)
                if ndim > 0:
                    pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                    loglik_Phi.append(pulsarPSDLL(np.zeros(nfdms), \
                        psr.Fdmfreqs, Tmax, pmin, pmax, pstart, pwidth, pindex, \
                                ii, np.arange(ntot, ntot+nfdms), bvary))
                    psd = loglik_Phi(-1)

                    cov = np.diag(pwidth[bvary]**2)
                    p0 = pstart[bvary]
                    sampler = ptmcmc.PTSampler(ndim, psd.loglikelihood, psd.logprior, cov=cov, \
                            outDir='./gibbs-chains/', verbose=False, nowrite=True)

                    psd.setSampler(sampler)

                    #steps = ndim*40
                    #sampler.sample(p0, steps, thin=1, burn=10)

                    #newpars[psd.pindex:psd.pindex+ndim] = \
                    #        sampler._chain[steps-1,:]

    return loglik_Phi

def gibbs_sample_loglik_Phi_an(likob, a, curpars):
    """
    Sample the Phi-loglikelihood conditional for the analytic parameters. The
    numerical (MCMC) parameters are done elsewhere

    @param likob:   the full likelihood object
    @param a:       list of arrays with all the Gibbs-only parameters
    @param curpars: the current value of all non-Gibbs parameters
    """
    newpars = curpars.copy()

    # Sample the power spectra analytically
    for ii, psr in enumerate(likob.ptapsrs):
        for ss, signal in enumerate(likob.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'spectrum':
                nms = likob.npm[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms

                # Loop over the frequencies
                for jj in range(signal['ntotpars']):
                    # We can sample directly from this distribution.
                    # Prior domain:
                    pmin = signal['pmin'][jj]
                    pmax = signal['pmax'][jj]
                    rhomin = 10**pmin
                    rhomax = 10**pmax
                    tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)

                    # Draw samples between rhomax and rhomin, according to
                    # an exponential distribution
                    scale = 1 - np.exp(tau*(1.0/rhomax-1.0/rhomin))
                    eta = np.random.rand(1) * scale
                    rhonew = -tau / (np.log(1-eta)-tau/rhomax)

                    newpars[signal['parindex']+jj] = np.log10(rhonew)
            elif signal['pulsarind'] == ii and signal['stype'] == 'dmspectrum':
                nms = likob.npm[ii]
                nfs = likob.npf[ii]
                nfdms = likob.npfdm[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms
                if 'rednoise' in likob.gibbsmodel:
                    ntot += nfs

                # Loop over the frequencies
                for jj in range(signal['ntotpars']):
                    # We can sample directly from this distribution.
                    # Prior domain:
                    pmin = signal['pmin'][jj]
                    pmax = signal['pmax'][jj]
                    rhomin = 10**pmin
                    rhomax = 10**pmax
                    tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)

                    # Draw samples between rhomax and rhomin, according to
                    # an exponential distribution
                    scale = 1 - np.exp(tau*(1.0/rhomax-1.0/rhomin))
                    eta = np.random.rand(1) * scale
                    rhonew = -tau / (np.log(1-eta)-tau/rhomax)

                    newpars[signal['parindex']+jj] = np.log10(rhonew)

    return newpars

def gibbs_sample_loglik_Phi(likob, a, curpars, loglik_PSD):
    """
    Sample the Phi-loglikelihood conditional. Some models can be done
    analytically

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_PSD:  List of prepared likelihood/samplers for non-analytic
                        models
    """
    # First sample from the analytic signals
    newpars = gibbs_sample_loglik_Phi_an(likob, a, curpars)

    # Now do the numerical ones through sampling with an MCMC
    for psd in loglik_PSD:
        # Update the sampler with the new Gibbs-coefficients
        psd.setNewData(a[psd.psrindex][psd.gindices])
        ndim = psd.dimensions()
        p0 = newpars[psd.pindex:psd.pindex+ndim]

        steps = ndim*40
        psd.sampler.sample(p0, steps, thin=1, burn=10)

        newpars[psd.pindex:psd.pindex+ndim] = psd.sampler._chain[steps-1,:]

    return newpars

def gibbs_prepare_loglik_Det(likob, curpars):
    """
    Prepares the likelihood objects for deterministic parameters

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """
    sigList = ['lineartimingmodel', 'nonlineartimingmodel', 'fouriermode', \
            'dmfouriermode', 'jitterfouriermode', \
            'bwm']

    loglik_Det = []

    mask = np.array([0]*likob.dimensions, dtype=np.bool)
    for ss, signal in enumerate(likob.ptasignals):
        if signal['stype'] in sigList:
            # Deterministic source, so include it
            pindex = signal['parindex']
            npars = np.sum(signal['bvary'])
            mask[pindex:pindex+npars] = True

    ndim = np.sum(mask)
    newpars = curpars.copy()

    if ndim > 0:
        # Only sample if we really have something to do
        pmin = likob.pmin[mask]
        pmax = likob.pmax[mask]
        pstart = likob.pstart[mask]
        pwidth = likob.pwidth[mask]
        bvary = np.array([1]*ndim, dtype=np.bool)

        # Prepare the likelihood function
        loglik_Det.append(pulsarDetLL(likob, newpars, mask, pmin, pmax, pstart, pwidth, bvary))
        pdl = loglik_Det[-1]

        cov = np.diag(pdl.pwidth**2)
        p0 = pdl.pstart
        sampler = ptmcmc.PTSampler(ndim, pdl.loglikelihood, pdl.logprior, cov=cov, \
                outDir='./gibbs-chains/', verbose=False, nowrite=True)

        pdl.setSampler(sampler)
        #steps = ndim*20
        #sampler.sample(p0, steps, thin=1, burn=10)

        #newpars[pdl.mask] = sampler._chain[steps-1,:]

        # And we should not forget to re-set the gibbsresiduals
        #for psr in likob.ptapsrs:
        #    psr.gibbsresiduals = psr.detresiduals - psr.gibbssubresiduals

    return loglik_Det


def gibbs_sample_loglik_Det(likob, curpars, loglik_Det):
    """
    Sample the Phi-loglikelihood conditional. Some models can be done
    analytically

    @param likob:       the full likelihood object
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_Det:  List of prepared likelihood/samplers
    """

    # First sample from the analytic signals
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pdl in loglik_Det:
        ndim = pdl.dimensions()
        pdl.pstart = curpars.copy()
        p0 = pdl.pstart[pdl.mask]

        steps = ndim*20
        pdl.sampler.sample(p0, steps, thin=1, burn=10)

        newpars[pdl.mask] = pdl.sampler._chain[steps-1,:]

    return newpars



def gibbs_prepare_loglik_corrPhi(likob, curpars):
    """
    Prepares the likelihood objects for the power-spectra of correlated signals

    @param likob:   the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars: the current value of all parameters
    """

    loglik_corrPSD = []
    for ss, signal in enumerate(likob.ptasignals):
        if signal['stype'] == 'powerlaw' and signal['corr'] != 'single':
            bvary = signal['bvary']
            pindex = signal['parindex']
            pstart = signal['pstart']
            pmin = signal['pmin']
            pmax = signal['pmax']
            pwidth = signal['pwidth']
            Tmax = signal['Tmax']

            b = []
            gindices = []
            for ii, psr in enumerate(likob.ptapsrs):
                nms = likob.npm[ii]
                nfs = likob.npf[ii]
                nfdms = likob.npfdm[ii]
                npus = likob.npu[ii]

                b.append(np.zeros(nfs))

                # Save the indices of the Gibbs parameters
                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms
                if 'rednoise' in likob.gibbsmodel:
                    ntot += nfs
                if 'dm' in likob.gibbsmodel:
                    ntot += nfdms
                if 'jitter' in likob.gibbsmodel:
                    ntot += npus

                gindices.append(np.arange(ntot, ntot+nfs))

            ndim = np.sum(bvary)
            if ndim > 0:
                pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                # Make the b's with only GW gibsscoeffs here

                loglik_corrPSD.append(corrPSDLL(b, \
                        likob.Ffreqs_gw, Tmax, pmin, pmax, pstart, pwidth, \
                        pindex, -1, gindices, bvary))
                psd = loglik_corrPSD[-1]

                cov = np.diag(pwidth[bvary]**2)
                p0 = pstart[bvary]
                sampler = ptmcmc.PTSampler(ndim, psd.loglikelihood, psd.logprior, cov=cov, \
                        outDir='./gibbs-chains/', verbose=False, nowrite=True)

                psd.setSampler(sampler)

    return loglik_corrPSD


def gibbs_sample_loglik_corrPhi(likob, a, curpars, loglik_corrPSD):
    """
    Sample the correlated Phi-loglikelihood conditional. Some models can be done
    analytically (latter not yet implemented)

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_PSD:  List of prepared likelihood/samplers for non-analytic
                        models
    """
    newpars = curpars.copy()

    for psd in loglik_corrPSD:
        b = []
        for ii, ind in enumerate(psd.gindices):
            b.append(a[ii][ind])

        psd.setNewData(b, likob.Scor_inv, likob.Scor_ldet)
        ndim = psd.dimensions()

        p0 = newpars[psd.pindex:psd.pindex+ndim]

        steps = ndim*40
        psd.sampler.sample(p0, steps, thin=1, burn=10)

        newpars[psd.pindex:psd.pindex+ndim] = psd.sampler._chain[steps-1,:]

    return newpars


def gibbs_prepare_correlations(likob):
    """
    Prepare the inverse covariance matrix with correlated signals, and related
    quantities for the Gibbs sampler

    @param likob:   the full likelihood object
    """
    if not np.all(likob.Svec == 0):
        likob.have_gibbs_corr = True

        # There actually is a signal, so invert the correlation covariance
        try:
            U, s, Vt = sl.svd(likob.Scor)

            if not np.all(s > 0):
                raise ValueError("ERROR: WScor singular according to SVD")

            likob.Scor_inv = np.dot(U * (1.0/s), Vt)
            #likob.Scor_Li = U * (1.0 / np.sqrt(s))      # Do we need this?
            likob.Scor_ldet = np.sum(np.log(s))

        except ValueError:
            print "WTF?"
            print "Look in wtf.txt for the Scor matrix"
            np.savetxt("wtf.txt", likob.Scor)
            raise
            
    else:
        likob.have_gibbs_corr = False

def gibbs_psr_corrs(likob, psrindex, a):
    """
    Get the Gibbs coefficient quadratic offsets for the correlated signals, for
    a specific pulsar

    @param likob:       The full likelihood object
    @param psrindex:    Index of the pulsar
    @param a:           List of Gibbs coefficient of all pulsar (of previous step)

    @return:    (pSinv_vec, pPvec), the quadratic offsets
    """
    psr = likob.ptapsrs[psrindex]

    # The inverse of the GWB correlations are easy
    pSinv_vec = (1.0 / likob.Svec[:likob.npf[psrindex]]) * \
            likob.Scor_inv[psrindex,psrindex]

    # For the quadratic offsets, we'll need to do some splice magic
    # First select the slice we'll need from the correlation matrix
    temp = np.arange(len(likob.ptapsrs))
    psrslice = np.delete(temp, psrindex)
    #corr_inv = likob.Scor_inv[:,psrslice]       # Specific slice of inverse

    # The quadratic offset we'll return
    pPvec = np.zeros(psr.Fmat.shape[1])

    # Pre-compute the GWB-index offsets of all the pulsars
    corrmode_offset = []
    for ii in range(len(likob.ptapsrs)):
        nms = likob.npm[ii]
        nfs = likob.npf[ii]
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        # GWs are all the way at the end. Sum all we need
        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms
        if 'rednoise' in likob.gibbsmodel:
            ntot += nfs
        if 'dm' in likob.gibbsmodel:
            ntot += nfdms
        if 'jitter' in likob.gibbsmodel:
            ntot += npus

        corrmode_offset.append(ntot)


    # For every mode, build the b vector
    for ii, freq in enumerate(psr.Ffreqs):
        # We are not even sure if all pulsars have this frequency, so be
        # careful. Just create them on the fly
        b = []
        A = []
        for jj in psrslice:
            if ii < likob.ptapsrs[jj].Fmat.shape[1]:
                # Have it, add to the sum
                b.append(a[jj][corrmode_offset[jj]+ii])
                A.append(likob.Scor_inv[psrindex,jj] / freq)

        # Make numpy arrays
        b = np.array(b)
        A = np.array(A)

        # Ok, we have the two vectors. Now fill the next element of
        pPvec[ii] = np.sum(b * A)

    return (pSinv_vec, pPvec)



def gibbs_sample_a(likob, preva=None):
    """
    Assume that all the noise parameters have been set (N, Phi, Theta). Given
    that, return a sample from the coefficient/timing model parameters

    @param likob:   the full likelihood object
    @param preva:   the previous list of Gibbs coefficients. Defaults to all
                    zeros

    @return: list of coefficients/timing model parameters per pulsar
    """

    if preva is None:
        preva = []
        for ii, psr in enumerate(likob.ptapsrs):
            preva.append(np.zeros(likob.npz[ii]))

    a = preva

    for ii, psr in enumerate(likob.ptapsrs):
        #zindex = np.sum(likob.npz[:ii])
        nzs = likob.npz[ii]
        nms = likob.npm[ii]
        findex = np.sum(likob.npf[:ii])
        nfs = likob.npf[ii]
        fdmindex = np.sum(likob.npfdm[:ii])
        nfdms = likob.npfdm[ii]
        uindex = np.sum(likob.npu[:ii])
        npus = likob.npu[ii]

        # Make ZNZ and Sigma
        ZNZ = np.dot(psr.Zmat.T, ((1.0/psr.Nvec) * psr.Zmat.T).T)
        Sigma = ZNZ.copy()

        # ahat is the slice ML value for the coefficients. Need ENx
        ENx = np.dot(psr.Zmat.T, psr.detresiduals / psr.Nvec)

        # Depending on what signals are in the Gibbs model, we'll have to add
        # prior-covariances to ZNZ
        zindex = 0
        if 'design' in likob.gibbsmodel:
            # Do nothing, she'll be 'right
            zindex += nms

        if 'rednoise' in likob.gibbsmodel:
            ind = range(zindex, zindex+nfs)
            Sigma[ind, ind] += 1.0 / likob.Phivec[findex:findex+nfs]
            zindex += nfs

        if 'dm' in likob.gibbsmodel:
            ind = range(zindex, zindex+nfdms)
            Sigma[ind, ind] += 1.0 / likob.Thetavec[fdmindex:fdmindex+nfdms]
            zindex += nfdms

        if 'jitter' in likob.gibbsmodel:
            ind = range(zindex, zindex+npus)
            Sigma[ind, ind] += 1.0 / psr.Jvec

        if 'corrsig' in likob.gibbsmodel and likob.have_gibbs_corr:
            (pSinv_vec, pPvec) = gibbs_psr_corrs(likob, ii, a)

            ind = range(zindex, zindex + nfs)
            Sigma[ind, ind] += pSinv_vec
            ENx[ind] -= pPvec

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
            try:
                cfL = sl.cholesky(Sigma, lower=True)
                cf = (cfL, True)

                # Calculate the inverse Cholesky factor (can we do this faster?)
                cfLi = sl.cho_factor(cfL, lower=True)
                Li = sl.cho_solve(cfLi, np.eye(Sigma.shape[0]))

                ahat = sl.cho_solve(cf, ENx)
            except np.linalg.LinAlgError:
                U, s, Vt = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                Sigi = np.dot(U * (1.0/s), Vt)
                Li = U * (1.0 / np.sqrt(s))

                ahat = np.dot(Sigi, ENx)
        except ValueError:
            print "WTF?"
            print "Look in wtf.txt for the Sigma matrix"
            np.savetxt("wtf.txt", Sigma)
            raise

        # Get a sample from the coefficient distribution
        aadd = np.dot(Li, np.random.randn(Li.shape[0]))
        addcoefficients = ahat + aadd

        psr.gibbscoefficients = addcoefficients.copy()
        psr.gibbscoefficients[:psr.Mmat.shape[1]] = np.dot(psr.tmpConv, \
                addcoefficients[:psr.Mmat.shape[1]])

        # We really do not care about the tmp's at this point. Save them
        # separately
        a[ii] = psr.gibbscoefficients
        psr.gibbssubresiduals = np.dot(psr.Zmat, addcoefficients)
        psr.gibbsresiduals = psr.detresiduals - psr.gibbssubresiduals

    return a



def gibbsQuantities(likob, parameters):
    """
    Calculate basic Gibbs quantities at least once

    @param likob:       The full likelihood object
    @param parameters:  The current non-Gibbs model parameters
    """
    likob.setPsrNoise(parameters)

    # Place only correlated signals (GWB) in the Phi matrix, and the rest in the
    # noise vectors
    likob.constructPhiAndTheta(parameters, make_matrix=True, \
            noise_vec=True, gibbs_expansion=True)


    if likob.haveDetSources:
        likob.updateDetSources(parameters)



def RunGibbs(likob, steps, chainsdir):
    """
    Run a gibbs sampler on a, for now, simplified version of the likelihood.

    Parameters are grouped into several categories:
    1) a, the Fourier coefficients and timing model parameters
    2) N, the white noise parameters
    3) Phi, the red noise PSD coefficients
    4) Jitter: pulse Jitter. May be included in N later on
    5) Deterministic: all deterministic sources not described elsewhere

    Note that this blocked Gibbs sampler has more parameters than the usual
    MCMC's (except for mark11). This sampler also samples from the timing model
    parameters, and the Fourier coefficients: the Gibbs-only parameters. These
    are all appended in the MCMC chain after the normal parameters. Everything
    is correctly labeled in the 'ptparameters.txt' file.

    The outputted 'chain_1.txt' file in the outputdir has the same format as
    that of the PTMCMC_generic sampler, however the log-likelihood/posterior
    values are not saved. These are set to zero.

    @param likob:       The likelihood object, containing everything
    @param steps:       The number of full-circle Gibbs steps to take
    @param chainsdir:   Where to save the MCMC chain
    """
    if not likob.likfunc in ['gibbs']:
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    # Save the description of all the parameters
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    # Clear the file for writing
    chainfilename = chainsdir + '/chain_1.txt'
    chainfile = open(chainfilename, 'w')
    chainfile.close()

    # Dump samples to file every dumpint steps (no thinning)
    dumpint = 100

    ndim = likob.dimensions         # The non-Gibbs model parameters
    ncoeffs = np.sum(likob.npz)     # The Gibbs-only parameters/coefficients
    pars = likob.pstart.copy()      # The Gibbs

    # Make a list of all the blocked signal samplers (except for the coefficient
    # samplers)
    loglik_N = gibbs_prepare_loglik_N(likob, pars)
    loglik_PSD = gibbs_prepare_loglik_Phi(likob, pars)
    loglik_J = gibbs_prepare_loglik_J(likob, pars)
    loglik_Det = gibbs_prepare_loglik_Det(likob, pars)
    loglik_corrPSD = gibbs_prepare_loglik_corrPhi(likob, pars)

    # The gibbs coefficients will be set by gibbs_sample_a
    a = None

    samples = np.zeros((min(dumpint, steps), ndim+ncoeffs))
    stepind = 0
    for step in range(steps):
        doneIteration = False
        iter = 0

        # Start with calculating the required likelihood quantities
        gibbsQuantities(likob, pars)

        # If necessary, invert the correlation matrix Svec & Scor with Ffreqs_gw
        # and Fmat_gw
        if 'corrsig' in likob.gibbsmodel:
            gibbs_prepare_correlations(likob)

        while not doneIteration:
            try:
                # Generate new coefficients
                a = gibbs_sample_a(likob, a)

                samples[stepind, ndim:] = np.hstack(a)

                doneIteration = True

            except np.linalg.LinAlgError:
                # Why does SVD sometimes not converge?
                # Try different values...
                iter += 1

                if iter > 100:
                    print "WARNING: numpy.linalg problems"

            # Generate new white noise parameers
            #pars = gibbs_sample_N_old(likob, pars)
            pars = gibbs_sample_loglik_N(likob, pars, loglik_N)

            # Generate new red noise parameters
            #pars = gibbs_sample_Phi_old(likob, a, pars)
            pars = gibbs_sample_loglik_Phi(likob, a, pars, loglik_PSD)

            # Generate new correlated equad/jitter parameters
            #pars = gibbs_sample_J_old(likob, a, pars)
            pars = gibbs_sample_loglik_J(likob, a, pars, loglik_J)

            # If we have 'm, sample from the deterministic sources
            #pars = gibbs_sample_Det_old(likob, pars)
            pars = gibbs_sample_loglik_Det(likob, pars, loglik_Det)

            if 'corrsig' in likob.gibbsmodel and likob.have_gibbs_corr:
                # Generate new GWB parameters
                pars = gibbs_sample_loglik_corrPhi(likob, a, pars, loglik_corrPSD)

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

