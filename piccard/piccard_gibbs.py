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
    toaerrs = None          # The uncertainties
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

    def __init__(self, residuals, toaerrs):
        """
        @param residuals:   Initialise the residuals we'll work with
        @param toaerrs:     The original TOA uncertainties
        """
        self.residuals = residuals
        self.toaerrs = toaerrs
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
                np.sum(np.log(self.Nvec))

    def logprior(self, parameters):
        bok = -np.inf
        if np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax):
            bok = 0

        return bok

    def dimensions(self):
        if len(self.vNvec) != len(self.vis_efac):
            raise ValueError("dimensions not set correctly")

        return len(self.vNvec)


class singleFreqLL(object):
    """
    For now, this class only does a single frequency for a single pulsar. Later
    on we'll expand it
    """
    a = None
    pmin = None
    pmax = None
    pstart = None
    pwidth = None

    def __init__(self, a, pmin, pmax, pstart, pwidth):
        self.a = a
        self.pmin = np.array([pmin])
        self.pmax = np.array([pmax])
        self.pstart = np.array([pstart])
        self.pwidth = np.array([pwidth])

    def loglikelihood(self, parameters):
        phivec = np.array([parameters[0], parameters[0]])

        return -0.5 * np.sum(self.a**2/phivec) - 0.5*np.sum(np.log(phivec))

    def logprior(self, parameters):
        bok = -np.inf
        if np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax):
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
    # Make ZNZ

    a = []
    for ii, psr in enumerate(self.ptapsrs):
        ZNZ = np.dot(psr.Zmat.T, ((1.0/psr.Nvec) * psr.Zmat.T).T)

        di = np.diag_indices(ZNZ.shape[0])

        # Construct the covariance matrix
        Sigma = ZNZ.copy()
        Sigma[di] += 1.0/psr.Phivec

        cfL = sl.cholesky(Sigma, lower=True)
        cf = (cfL, True)

        # ahat is the slice ML value for the coefficients
        ENx = np.dot(psr.Zmat, psr.detresiduals / psr.Nvec)
        ahat = sl.cho_solve(cf, ENx)

        # Calculate the inverse Cholesky factor (can we do this faster?)
        cfLi = sl.cho_factor(cfL)
        Li = sl.cho_solve(cfLi, np.eye(Sigma.shape[0]))

        # Get a sample from the coefficient distribution
        psr.gibbscoefficients = ahat + np.dot(Li, np.random.randn(Li.shape[0]))
        a.append(psr.gibbscoefficients[psr.Mmat.shape[1]:])

        # We really do not care about the tmp's at this point
        psr.gibbsresiduals = psr.detresiduals - np.dot(psr.Zmat, a[-1])

    return a


def gibbs_sample_N(self, curpars):
    # Given the values for a, which have been subtracted from the residuals, we
    # now need to find the PSD and noise coefficients.
    newpars = curpars.copy()

    for ii, psr in enumerate(self.ptapsrs):
        pnl = pulsarNoiseLL(psr.residuals, psr.toaerrs)

        # Add the signals
        for ss, signal in enumerate(self.ptasignals):
            pstart = curpars[signal['parindex']]
            if signal['pulsarind'] == ii and signal['stype'] == 'efac':
                # We have a winner: add this signal
                pnl.addSignal(signal['Nvec'], True, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        fixed=(not signal['bvary'][0]))
            elif signal['pulsarind'] == ii and signal['stype'] == 'equad':
                # We have a winner: add this signal
                pnl.addSignal(signal['Nvec'], False, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        fixed=(not signal['bvary'][0]))

        # Run a tiny MCMC of one correlation length, and return the parameters
        ndim = pnl.dimensions()
        cov = np.diag(pnl.pwidth**2)
        p0 = pnl.pstart
        sampler = ptmcmc.PTSampler(ndim, pnl.loglikelihood, pnl.logprior, cov=cov, \
                outDir='./gibbs-chains/', verbose=True, nowrite=True)

        sampler.sample(p0, ndim*100, thin=1, burn=0)

        newpars[pnl.pindex] = sampler._chain[-1,:]

    return newpars



def gibbs_sample_Phi(self, a, curpars):
    """
    Same as gibbs_sample_N, but for the phi frequency components

    """
    newpars = curpars.copy()

    for ii, psr in enumerate(self.ptapsrs):
        for ss, signal in enumerate(self.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'spectrum':
                pstart = curpars[signal['parindex']]

                # Loop over the frequencies
                for jj in range(signal['ntotpars']):
                    # Warning: does not take into account non-varying parameters
                    sfl = singleFreqLL(a[ii][2*jj:2*jj+2], signal['pmin'][jj], \
                            signal['pmax'][jj], pstart[jj], signal['pwidth'][jj])

                    ndim = sfl.dimensions()

                    cov = np.diag(sfl.pwidth**2)
                    p0 = sfl.pstart
                    sampler = ptmcmc.PTSampler(ndim, sfl.loglikelihood, sfl.logprior, cov=cov, \
                            outDir='./gibbs-chains/', verbose=True, nowrite=True)

                    sampler.sample(p0, ndim*100, thin=1, burn=1)

                    newpars[pnl.pindex] = sampler._chain[-1,:]

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
    if likob.likfunc != 'gibbs1':
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    ndim = likob.dimensions
    pars = likob.pstart.copy()

    for step in range(steps):
        # Start with calculating the required likelihood quantities
        gibbsQuantities(likob, pars)

        # Generate new coefficients
        a = gibbs_sample_a(likob)

        # Generate new white noise parameers
        pars = gibbs_sample_N(likob, pars)

        # Generate new red noise parameters
        gibbs_sample_Phi(likob, a, pars)
