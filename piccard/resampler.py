#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
Resampling techniques to search for gravitational-waves using the single-pulsar
MCMC chains of the Gibbs sampler
"""

from __future__ import division

import numpy as np
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os as os
import glob
import sys

from piccard import *
from piccard_samplers import *

class GWresampler(object):
    """
    A likelihood resampler class that uses previous MCMC chains
    """
    def __init__(self, hdmat, dirlist, burnin=0, thin=1, sampler='auto', \
            rho_prior_min=-19.0, rho_prior_max=-9.0):
        """
        This init function accepts a list of directories with MCMC chains (most
        likely from the Gibbs sampler). Crucial about the MCMC chains is that
        each includes the Fourier modes, and that the Fourier modes are all at
        the same frequency.

        @param hdmat:       Hellings & Downs matrix
        @param dirlist:     List of directories with MCMC chains
        @param burnin:      How many samples to burn before using them
        @param thin:        Chain thinning length
        @param sampler:     'auto'/'ptmcmc'/'multinest'/'emcee'
        @param rho_prior:   
        """
        self.Npsr = len(dirlist)        # Number of pulsars
        self.freqs = np.zeros(0)        # Spectrum frequencies
        self.nfreqs = 0
        self.l_mcmcdir = dirlist        # MCMC directory list
        self.l_chain = []               # relevant chains
        self.l_oia = []                 # Indices in MCMC chain (Fmat)
        self.l_oirho = []               # Indices in MCMC chain (spectrum)
        self.Tmax = 0.0                 # 1.0/min(freqs)
        self.hdmat = hdmat              # Hellings & Downs matrix
        self.have_spectrum = True       # Whether or not we have a spectrum

        # Set the priors on the noise
        self.rho_prior_min = rho_prior_min * np.ones(self.Npsr)
        self.rho_prior_max = rho_prior_max * np.ones(self.Npsr)

        for ii, direc in enumerate(dirlist):
            (llf, lpf, chainf, labels, pulsarid, pulsarname, stype, mlpso, mlpsopars) = \
                    ReadMCMCFile(direc, sampler=sampler, incextra=True)

            if len(set(pulsarname)) > 1:
                raise ValueError("Resampling requires single-pulsar MCMC chains!")

            # Store the original MCMC indices for the noise/signal
            self.l_oia.append(np.array(stype) == 'Fmat')
            self.l_oirho.append(np.array(stype) == 'spectrum')

            if ii == 0:
                # Store the frequencies
                self.freqs = np.array(labels)[self.l_oia[ii]][::2].astype(np.float)
                self.nfreqs = len(self.freqs)
                self.Tmax = 1.0 / np.min(self.freqs)

                if np.sum(self.l_oirho[ii]) == 0:
                    self.have_spectrum = False
                else:
                    self.have_spectrum = True

            # Check that the labels are consistent with each other so far
            if not self.checkFreqs(np.array(labels), ii):
                raise ValueError("Fmat/spectrum frequencies don't match up")

            # Add the rho and a chain
            localchain_a = chainf[burnin::thin, self.l_oia[ii]]
            localchain_rho = chainf[burnin::thin, self.l_oirho[ii]]
            self.l_chain.append(np.append(localchain_a, localchain_rho, axis=1))

    def checkFreqs(self, labels, psrind):
        """
        Check whether all the provided frequencies match up
        
        @param labels:  List of the MCMC labels for the pulsar
        @param psrind:  Index of the pulsar
        """
        if self.have_spectrum:
            spok = np.all(self.freqs == np.array(labels)[self.l_oirho[psrind]].astype(np.float))
        else:
            spok = True

        return np.all(self.freqs == np.array(labels)[self.l_oia[psrind]][::2].astype(np.float)) and \
                np.all(self.freqs == np.array(labels)[self.l_oia[psrind]][1::2].astype(np.float)) and \
                spok

    def randomSample_indiv(self, psrind):
        """
        Given a pulsar number, draw a random sample of a/rho from the noise-only
        MCMC chain

        @param psrind:  pulsar index

        @return:        np.append(a, rho)
        """
        return self.l_chain[psrind][np.random.randint(0, len(self.l_chain[psrind]))]

    def randomSample_freq(self, psrind, ifreq):
        """
        Like randomSample_indiv, but now do it for a single frequency. Using it
        this way effectively decouples all frequencies, which might be a good
        approximation.

        @param psrind:  pulsar index
        @param ifreq:   frequency index

        @return:        np.append(a, rho)
        """
        return self.l_chain[psrind][np.random.randint(0, len(self.l_chain[psrind])), [2*ifreq, 2*ifreq+1, 2*self.nfreqs+ifreq]]


    def randomSample(self):
        """
        Draw a random sample of a/rho for all pulsars simultaneously

        @return:    Matrix with N_pulsar rows of np.append(a, rho)
        """
        rv = np.zeros((self.Npsr, 3*self.nfreqs))
        for ii in range(self.Npsr):
            rv[ii,:] = self.randomSample_indiv(ii)

        return rv

    def gwPSD(self, Agw, gammagw):
        """
        Given a GWB amplitude and spectral index, calculate the PSD of that GWB
        at the frequencies of self.freqs

        @param Agw:     log10(GW amplitude)
        @param gammagw: GWB spectral index

        @return:        PSDgwb [s^2 / bin]  (one bin has size 1/Tmax)
        """
        freqpy = self.freqs * pic_spy
        return (10**(2*Agw) * pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-gammagw)

    def transformParsFreqs_indiv(self, psrind, rho, Agw, gammagw):
        """
        Given the combined noise+GW values rho for a pulsar, calculate the
        noise amplitudes with a GWB modelled by A and gamma

        @param psrind:  pulsar index
        @param rho:     log10(Noise+GW) PSD values
        @param Agw:     log10(GW amplitude)
        @param gammagw: GWB spectral index

        @return:        rho_noise
        """
        pc_gw = self.gwPSD(Agw, gammagw)
        if not np.all(10**rho - pc_gw > 10**self.rho_prior_min[psrind]) or \
                not np.all(10**rho - pc_gw < 10**self.rho_prior_max[psrind]):
            raise ValueError("outofbound")

        return np.log10(10**rho - pc_gw)

    def transformParsFreqs_freq(self, psrind, ifreq, rho, Agw, gammagw):
        """
        Like transformParsFreqs_indiv, but now for a single frequency only

        @param psrind:  pulsar index
        @param ifreq:   Index of frequency of the transformation
        @param rho:     log10(Noise+GW) PSD value
        @param Agw:     log10(GW amplitude)
        @param gammagw: GWB spectral index

        @return:        rho_noise
        """
        pc_gw = self.gwPSD(Agw, gammagw)[ifreq]
        if not 10**rho - pc_gw > 10**self.rho_prior_min[psrind] or \
                not 10**rho - pc_gw < 10**self.rho_prior_max[psrind]:
            raise ValueError("Outside of prior")

        return np.log10(10**rho - pc_gw)

    def transformJacobian_indiv(self, rho_full, rho_noise):
        """
        Return the transformation Jacobian: log(J) (single pulsar)

        @param l_rho_full:  scalar/array of full PSD coefficients
        @param l_rho_noise: scalar/array of noise PSD coefficients

        @return:            log(|J|)
        """
        return np.sum(np.atleast_1d(rho_full - rho_noise)) * np.log(10)

    def transformJacobian(self, l_rho_full, l_rho_noise):
        """
        Return the transformation Jacobian: log(J)

        @param rho_full:    scalar/array of full PSD coefficients
        @param rho_noise:   scalar/array of noise PSD coefficients

        @return:            log(|J|)
        """
        return np.sum([self.transformJacobian_indiv(l_rho_full[ii], \
                l_rho_noise[ii]) for ii in range(len(l_rho_full))])

    def conditionalSample_indiv(self, psrind, Agw, gammagw, \
            decouple=True, usetransform=True):
        """
        Given the GWB parameters, draw a random sample of GW+noise parameters
        (rho), and transform these to our new parameterization.

        @param psrind:          pulsar index
        @param Agw:             log10(GW amplitude)
        @param gammagw:         GWB spectral index
        @param decouple:        Decouple frequencies True/False. Likely better
                                for high S/N
        @param usetransform:    Use a coordinate transform to high S/N noise
                                region

        @return:                np.append(a, rho_full, rho_noise)
        """
        if decouple:
            newpars = np.zeros(3*self.nfreqs)
            rho_noise = np.zeros(self.nfreqs)

            for ii in range(self.nfreqs):
                no_sample = True
                iter = 0
                while(no_sample and iter <= 10000):
                    try:
                        newpars[[2*ii, 2*ii+1, 2*self.nfreqs+ii]] = \
                            self.randomSample_freq(psrind, ii)
                        newrho = newpars[2*self.nfreqs+ii]
                        if usetransform:
                            rho_noise[ii] = self.transformParsFreqs_freq(psrind, ii, newrho, Agw, gammagw)
                        else:
                            rho_noise[ii] = newrho[ii]
                        no_sample = False
                    except ValueError:
                        pass

                    iter += 1
                    if iter == 1000:
                        print("Jumping outside of prior a lot!")
        else:
            no_sample = True
            iter = 0
            while(no_sample and iter <= 10000):
                # TODO: Is there any other way to do this sampling? Strong S/N seems
                # inefficient/biased now
                # Question: if we assume the frequencies to be not covariant, can we
                # decouple them from the chain? Most probably not...  --- RvH
                try:
                    newpars = self.randomSample_indiv(psrind)
                    newrho = newpars[2*self.nfreqs:]
                    if usetransform:
                        rho_noise = self.transformParsFreqs_indiv(psrind, newrho, Agw, gammagw)
                    else:
                        rho_noise = newrho
                    no_sample = False
                except ValueError:
                    pass

                iter += 1
                if iter == 1000:
                    print("Many iterations necessary for sampler!")

        return np.append(newpars, rho_noise)

    def conditionalSample(self, Agw, gammagw, decouple=True, usetransform=True):
        """
        Given the GWB parameters, draw a random sample of GW+noise parameters
        for all pulsars, and transform these to our new parameterization

        @param Agw:             log10(GW amplitude)
        @param gammagw:         GWB spectral index
        @param decouple:        Decouple frequencies True/False. Likely better
                                for high S/N
        @param usetransform:    Use a coordinate transform to high S/N noise
                                region

        @return:                Matrix with Npsr rows [a, rho_full, rho_noise]
        """
        rv = np.zeros((self.Npsr, 4*self.nfreqs))
        for ii in range(self.Npsr):
            rv[ii,:] = self.conditionalSample_indiv(ii, Agw, gammagw, \
                    decouple=decouple, usetransform=usetransform)

        return rv

    def loglik_orig_indiv(self, a, rho):
        """
        Given the a and rho values for a single pulsar, calculate the original likelihood

        @param a:   The Fourier coefficients for a single pulsar
        @param rho: The log10(PSD) amplitudes for a single pulsar

        @return:    Log-likelihood
        """
        pc = 10**np.array([rho, rho]).T.flatten()       # PSD coefficients
        return -0.5 * np.sum(a**2 / pc) - len(pc)*np.log(2*np.pi) - 0.5 * np.sum(np.log(pc))

    def loglik_orig(self, l_a, l_rho):
        """
        Given a list of a and rho values for all pulsars, calculate the full
        original likelihood

        @param l_a:     List of Fourier coefficient arrays for all pulsars
        @param l_rho:   List of arrays of log10(PSD) amplitudes for all pulsars

        @return:        Log-likelihood
        """
        if len(l_a) != len(l_rho):
            raise ValueError("l_a and l_rho not the same length")

        rv = 0.0
        for ii in range(len(l_a)):
            rv += self.loglik_orig_indiv(l_a[ii], l_rho[ii])

        return rv

    def loglik_full(self, l_a, l_rho, Agw, gammagw):
        """
        Given all these parameters, calculate the full likelihood

        @param l_a:     List of Fourier coefficient arrays for all pulsars
        @param l_rho:   List of arrays of log10(PSD) amplitudes for all pulsars
        @param Agw:     log10(GW amplitude)
        @param gammagw: GWB spectral index

        @return:        Log-likelihood
        """
        # Transform the GWB parameters to PSD coefficients (pc)
        pc_gw = self.gwPSD(Agw, gammagw)
        
        rv = 0.0
        for ii, freq in enumerate(self.freqs):
            a_cos = l_a[:,2*ii]         # Cosine modes for f=freq
            a_sin = l_a[:,2*ii+1]       # Sine modes for f=freq
            rho = l_rho[:,ii]           # PSD amp for f=freq

            # Covariance matrix is the same for sine and cosine modes
            cov = np.diag(10**rho) + self.hdmat * pc_gw[ii]
            cf = sl.cho_factor(cov)
            logdet = 2*np.sum(np.log(np.diag(cf[0])))
            
            # Add the log-likelihood for the cosine and the sine modes
            rv += -0.5 * np.dot(a_cos, sl.cho_solve(cf, a_cos)) - \
                   0.5 * np.dot(a_sin, sl.cho_solve(cf, a_sin)) - \
                   2*self.Npsr*np.log(2*np.pi) - logdet

        return rv

    def loglik(self, Agw, gammagw, n=1, decouple=True, usetransform=True):
        """
        Given Agw and gammagw, calculate the likelihood marginalized over all
        the noise parameters by averaging over n samples.

        @param Agw:             log10(GW amplitude)
        @param gammagw:         GWB spectral index
        @param n:               Number of repetitions to average over
        @param decouple:        Decouple frequencies True/False. Likely better
                                for high S/N
        @param usetransform:    Use a coordinate transform to high S/N noise
                                region

        @return:        Log-likelihood
        """
        rv = 0.0
        jc = 0.0
        for ii in range(n):
            newpars = self.conditionalSample(Agw, gammagw, \
                    decouple=decouple, usetransform=usetransform)
            l_a = newpars[:, :2*self.nfreqs]
            l_rho_full = newpars[:, 2*self.nfreqs:3*self.nfreqs]
            l_rho_noise = newpars[:, 3*self.nfreqs:]

            ll_orig = self.loglik_orig(l_a, l_rho_full)
            ll_full = self.loglik_full(l_a, l_rho_noise, Agw, gammagw)

            rv += np.exp(ll_full - ll_orig) / n

            if usetransform:
                jc += np.exp(self.transformJacobian(l_rho_full, l_rho_noise)) / n

        return np.log(rv) + np.log(jc)

    def loglik_gwonly(self, Agw, gammagw):
        """
        Given Agw and gammagw, calculate the likelihood for a no-noise model

        @param Agw:             log10(GW amplitude)
        @param gammagw:         GWB spectral index
        @param n:               Number of repetitions to average over

        @return:        Log-likelihood
        """
        rv = 0.0
        newpars = self.conditionalSample(Agw, gammagw, \
                decouple=False, usetransform=False)

        l_a = newpars[:, :2*self.nfreqs]
        l_rho_full = 0*newpars[:, 2*self.nfreqs:3*self.nfreqs] - 20
        l_rho_noise = 0*newpars[:, 3*self.nfreqs:] - 20

        return self.loglik_full(l_a, l_rho_noise, Agw, gammagw)
