from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os as os
import glob
import sys

from .piccard import *
from .piccard_pso import *
from . import PTMCMC_generic as ptmcmc

"""
This file implements a blocked Gibbs sampler. Gibbs sampling is a special case
of Metropolis-Hastings, and in Pulsar Timing data analysis it can be used to
increase the mixing rate of the chain. We still use the PAL/PAL2 version of
PTMCMC_generic to do the actual MCMC steps, but the steps are performed in
parameter blocks.
"""


def gibbs_loglikelihood(likob, parameters):
    """
    Within the Gibbs sampler, we would still like to have access to the
    loglikelihood value, even though it is not necessarily used for the sampling
    itself. This function evaluates the ll.

    This function does not set any of the noise/correlation auxiliaries. It
    assumes that has been done earlier in the Gibbs step. Also, it assumes the
    Gibbsresiduals have been set properly.

    @param likob:       The full likelihood object
    @param parameters:  All the model parameters
    @param coeffs:      List of all the Gibbs coefficients per pulsar

    @return:            The log-likelihood
    """

    xi2 = 0
    ldet = 0

    for ii, psr in likob.ptapsrs:
        # Before continuing, isn't it just possible to sum up all the
        # conditionals? Must come down to the same thing, right? Just offset
        # with the current values
        pass

    if 'design' in likob.gibbsmodel:
        ntot += nms
    if 'rednoise' in likob.gibbsmodel:
        ntot += nfs
    if 'dm' in likob.gibbsmodel:
        ntot += nfdms
    if 'jitter' in likob.gibbsmodel:
        ntot += npus

    return 0.0



class pulsarNoiseLL(object):
    """
    This class represents the likelihood function in the block with
    white-noise-only parameters.
    """

    def __init__(self, residuals, psrindex, maskJvec=None):
        """
        @param residuals:   Initialise the residuals we'll work with
        @param psrindex:    Index of the pulsar this noise applies to
        """
        self.vNvec = []                             # Mask residuals (varying)
        self.fNvec = []                             # Mask residuals (fixed)
        self.vis_efac = []                          # Is it an efac? (varying)
        self.fis_efac = []                          # Is it an equad (fixed)

        self.residuals = residuals                  # The residuals
        self.Nvec = np.zeros(len(residuals))        # Full noise vector
        self.pmin = np.zeros(0)                     # Minimum of prior domain
        self.pmax = np.zeros(0)                     # Maximum of prior domain
        self.pstart = np.zeros(0)                   # Start position
        self.pwidth = np.zeros(0)                   # Initial step-size
        self.fval = np.zeros(0)                     # Current value (non-varying)
        self.pindex = np.zeros(0, dtype=np.int)     # Index of parameters

        self.psrindex = psrindex                    # Inde xof pulsar
        self.maskJvec = maskJvec                    # Selection mask Jvec

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance
                                    # updates


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

    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate

    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :]

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin, self.pmax, self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx

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

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        if len(self.vNvec) != len(self.vis_efac):
            raise ValueError("dimensions not set correctly")

        return len(self.vNvec)


class pulsarPSDLL(object):
    """
    Like pulsarNoiseLL, but for the power spectrum coefficients. Expects the
    matrix to be diagonal. Is always a function of amplitude and spectral index
    """

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

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance

    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth[self.bvary]**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate


    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :]

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin[self.bvary], self.pmax[self.bvary], self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx



    def setNewData(self, a):
        self.a = a

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

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        return np.sum(self.bvary)



class corrPSDLL(object):
    """
    Like pulsarPSDLL, but now for correlated signals.
    """

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

        self.allPsrSame = False
        self.Scor_inv = None
        self.Scor_ldet = None

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance


    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth[self.bvary]**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate

    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :]

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin[self.bvary], self.pmax[self.bvary], self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx


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

        self.allPsrSame = np.all(freqmask)

    def loglikelihood(self, parameters):
        pars = self.pstart.copy()
        pars[self.bvary] = parameters

        freqpy = self.freqs * pic_spy
        pcdoubled = ((10**(2*pars[0])) * pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-pars[1])

        xi2 = 0
        ldet = 0

        if self.allPsrSame:
            nfreqs = len(pcdoubled)
            dotprod = np.dot(self.freqb[:,0], np.dot(self.Scor_inv, \
                    self.freqb[:,0]))
            xi2 += np.sum(dotprod / pcdoubled)
            ldet += nfreqs * self.Scor_ldet + np.sum(np.log(pcdoubled))
        else:
            # For every frequency, do an H&D matrix inverse inner-product
            for ii, pc in enumerate(pcdoubled):
                msk = self.freqmask[:, ii]
                xi2 += np.dot(self.freqb[msk,ii], np.dot(\
                        self.Scor_inv[msk,:][:,msk], self.freqb[msk,ii])) / pc
                ldet += self.Scor_ldet + np.sum(msk)*np.log(pc)

        return -0.5 * xi2 - 0.5 * ldet


    def logprior(self, parameters):
        bok = -np.inf
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

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

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance

    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth[self.bvary]**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate


    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :]

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin[self.bvary], self.pmax[self.bvary], self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx



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

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

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
        ndim = pnl.dimensions()

        pnl.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                covUpdate=ndim*400)

    return loglik_N

def gibbs_sample_loglik_N(likob, curpars, loglik_N, ml=False):
    """
    @param likob:       the full likelihood object
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_N:    List of prepared likelihood/samplers for the noise
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pnl in loglik_N:
        # Update the sampler with the new Gibbs residuals
        psr = likob.ptapsrs[pnl.psrindex]
        pnl.setNewData(psr.gibbsresiduals)

        if ml:
            newpars[pnl.pindex] = pnl.runPSO()
        else:
            p0 = newpars[pnl.pindex]
            newpars[pnl.pindex] = pnl.runSampler(p0)

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

                pnl.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                        covUpdate=ndim*400)

    return loglik_J

def gibbs_sample_loglik_J(likob, a, curpars, loglik_J, ml=False):
    """
    @param likob:       the full likelihood object
    @param a:       list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_J:    List of prepared likelihood/samplers for the correlated noise
    @param ml:          If True, return ML values (PSO), not a random sample
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

        if ml:
            newpars[pnl.pindex] = pnl.runPSO()
        else:
            p0 = newpars[pnl.pindex]
            newpars[pnl.pindex] = pnl.runSampler(p0)

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


                    psd.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                            covUpdate=ndim*400)

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
                    psd = loglik_Phi[-1]

                    psd.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                            covUpdate=ndim*400)

    return loglik_Phi

def gibbs_sample_loglik_Phi_an(likob, a, curpars, ml=False):
    """
    Sample the Phi-loglikelihood conditional for the analytic parameters. The
    numerical (MCMC) parameters are done elsewhere

    @param likob:   the full likelihood object
    @param a:       list of arrays with all the Gibbs-only parameters
    @param curpars: the current value of all non-Gibbs parameters
    @param ml:      If True, return ML values, not a random sample
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
                    if ml:
                        tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)
                        rhonew = 0.5 * tau
                        newpars[signal['parindex']+jj] = np.log10(rhonew)
                    else:
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
                    if ml:
                        tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)
                        rhonew = 0.5 * tau
                        newpars[signal['parindex']+jj] = np.log10(rhonew)
                    else:
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

def gibbs_sample_loglik_Phi(likob, a, curpars, loglik_PSD, ml=False):
    """
    Sample the Phi-loglikelihood conditional.

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_PSD:  List of prepared likelihood/samplers for non-analytic
                        models
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    # First sample from the analytic signals
    newpars = gibbs_sample_loglik_Phi_an(likob, a, curpars)

    # Now do the numerical ones through sampling with an MCMC
    for psd in loglik_PSD:
        # Update the sampler with the new Gibbs-coefficients
        psd.setNewData(a[psd.psrindex][psd.gindices])
        ndim = psd.dimensions()

        if ml:
            newpars[psd.pindex:psd.pindex+ndim] = pnl.runPSO()
        else:
            p0 = newpars[psd.pindex:psd.pindex+ndim]
            newpars[psd.pindex:psd.pindex+ndim] = psd.runSampler(p0)

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

        pdl.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                covUpdate=ndim*400)

    return loglik_Det


def gibbs_sample_loglik_Det(likob, curpars, loglik_Det, ml=False):
    """
    Sample the Phi-loglikelihood conditional. Some models can be done
    analytically

    @param likob:       the full likelihood object
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_Det:  List of prepared likelihood/samplers
    @param ml:      If True, return ML values, not a random sample
    """

    # First sample from the analytic signals
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pdl in loglik_Det:
        ndim = pdl.dimensions()
        pdl.pstart = curpars.copy()

        if ml:
            newpars[pdl.mask] = pnl.runPSO()
        else:
            p0 = pdl.pstart[pdl.mask]
            newpars[pdl.mask] = pdl.runSampler(p0)

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

                psd.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                        covUpdate=ndim*400)

    return loglik_corrPSD


def gibbs_sample_loglik_corrPhi(likob, a, curpars, loglik_corrPSD, ml=False):
    """
    Sample the correlated Phi-loglikelihood conditional. Some models can be done
    analytically (latter not yet implemented)

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_PSD:  List of prepared likelihood/samplers for non-analytic
                        models
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    newpars = curpars.copy()

    for psd in loglik_corrPSD:
        b = []
        for ii, ind in enumerate(psd.gindices):
            b.append(a[ii][ind])

        psd.setNewData(b, likob.Scor_inv, likob.Scor_ldet)
        ndim = psd.dimensions()

        if ml:
            newpars[pnl.pindex:psd.pindex+ndim] = psd.runPSO()
        else:
            # Use an adaptive MCMC
            p0 = newpars[psd.pindex:psd.pindex+ndim]
            newpars[pnl.pindex:psd.pindex+ndim] = pnl.runSampler(p0)

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



def gibbs_sample_a(likob, preva=None, ml=False):
    """
    Assume that all the noise parameters have been set (N, Phi, Theta). Given
    that, return a sample from the coefficient/timing model parameters

    @param likob:   the full likelihood object
    @param preva:   the previous list of Gibbs coefficients. Defaults to all
                    zeros
    @param ml:      If True, return ML values, not a random sample

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
        if ml:
            addcoefficients = ahat
        else:
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


def MLGibbs(likob, chainsdir, tol=1.0e-3, noWrite=False):
    """
    Run an iterative maximizer, utilising the blocked Gibbs likelihood
    representation, that iteratively uses analytic expressions and a Particle
    Swarm Optimiser.

    @param likob:       The likelihood object, containing everything
    @param chainsdir:   Where to save the MCMC chain
    @param tol:         Fractional tolerance in parameters / stopping criterion
    @param noWrite:     If True, do not write results to file
    """
    if not likob.likfunc in ['gibbs']:
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    if not noWrite:
        mlfilename = chainsdir + '/pso.txt'

    # Save the description of all the parameters
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

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

    mlpars = (1 + 10*tol) * pars
    fullml = np.zeros(ndim + ncoeffs)

    # Keep track of iterations
    iter = 0

    while not np.all( (mlpars - pars) / pars < tol):
        # Not converged yet
        mlpars = pars.copy()
        doneIteration = False

        # At iteration:
        iter += 1
        sys.stdout.write('\rParticle Swarm Optimise iteration: {0}  (max = {1})'.\
                        format(iter, np.max(mlpars - pars)/pars))
        sys.stdout.flush()

        # Calculate the required likelihood quantities
        gibbsQuantities(likob, pars)

        # If necessary, invert the correlation matrix Svec & Scor with Ffreqs_gw
        # and Fmat_gw
        if 'corrsig' in likob.gibbsmodel:
            gibbs_prepare_correlations(likob)

        while not doneIteration:
            try:
                # Generate new coefficients
                a = gibbs_sample_a(likob, a, ml=True)

                fullml[ndim:] = np.hstack(a)

                doneIteration = True

            except np.linalg.LinAlgError:
                # Why does SVD sometimes not converge?
                # Try different values...
                iter += 1

                if iter > 100:
                    print "WARNING: numpy.linalg problems"

            # Generate new white noise parameers
            pars = gibbs_sample_loglik_N(likob, pars, loglik_N, ml=True)

            # Generate new red noise parameters
            pars = gibbs_sample_loglik_Phi(likob, a, pars, loglik_PSD, ml=True)

            # Generate new correlated equad/jitter parameters
            pars = gibbs_sample_loglik_J(likob, a, pars, loglik_J, ml=True)

            # If we have 'm, sample from the deterministic sources
            pars = gibbs_sample_loglik_Det(likob, pars, loglik_Det, ml=True)

            if 'corrsig' in likob.gibbsmodel and likob.have_gibbs_corr:
                # Generate new GWB parameters
                pars = gibbs_sample_loglik_corrPhi(likob, a, pars, loglik_corrPSD, ml=True)

        fullml[:ndim] = pars

    # Great, found a ML value!
    sys.stdout.write("PSO maximum found!\n")

    # Open the file in append mode
    if not noWrite:
        mlfile = open(mlfilename, 'w')

        mlfile.write('\t'.join(["%.17e"%\
                (fullml[kk]) for kk in range(ndim+ncoeffs)]))
        mlfile.write('\n')
        mlfile.close()

    return fullml



def RunGibbs(likob, steps, chainsdir, noWrite=False):
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
    @param noWrite:     If True, do not write results to file
    """
    if not likob.likfunc in ['gibbs']:
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    # Save the description of all the parameters
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    # Clear the file for writing
    if not noWrite:
        chainfilename = chainsdir + '/chain_1.txt'
        chainfile = open(chainfilename, 'w')
        chainfile.close()

        # Also save the residuals for all pulsars
        likob.saveResiduals(chainsdir)

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
            pars = gibbs_sample_loglik_N(likob, pars, loglik_N)

            # Generate new red noise parameters
            pars = gibbs_sample_loglik_Phi(likob, a, pars, loglik_PSD)

            # Generate new correlated equad/jitter parameters
            pars = gibbs_sample_loglik_J(likob, a, pars, loglik_J)

            # If we have 'm, sample from the deterministic sources
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
            if not noWrite:
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

