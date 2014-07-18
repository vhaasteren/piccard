#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
seplik.py

Created by vhaasteren on 2013-08-06.
Copyright (c) 2013 Rutger van Haasteren

Create very simple separable likelihood functions, using individual likelihood
objects similar to the optimal statistic code

"""

from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import h5py as h5
import matplotlib.pyplot as plt
import os as os
import sys
import json
import tempfile

from .constant import *
from .datafile import *


class sepLikelihood(object):
    """
    A separable likelihood function, for use with for instance deterministic
    sources.

    Subsequent versions will be able to use previous MCMC noise chains for
    importance sampling
    """

    def __init__(self, likobs, numcpars, chaintuples=None):
        """
        Initialize the full likelihood with a list of ptaLikelihood objects. For
        the moment, assume that only the last couple of parameters belong to the
        common signal

        @param likobs:      List of likelihood objects
        @param numcpars:    Number of common parameters (common signal)
        @param chaintuples: If not None, then contains the MCMC chains of
                            individual pulsar runs. List of tuples with
                            chaintuples[ii] = (lp, ll, chain, labels)

        """
        self.likobs = likobs
        self.numcpars = numcpars
        self.single_lp = []
        self.single_ll = []
        self.single_chain = []
        self.single_labels = []
        self.likfunc = 'sep'

        if chaintuples is not None:
            for ct in chaintuples:
                self.single_lp.append(ct[0])
                self.single_ll.append(ct[1])
                self.single_chain.append(ct[2])
                self.single_labels.append(ct[3])
            

    @property
    def dimensions(self):
        """
        Return the number of parameters of the combined likelihood
        """
        return self.numcpars + np.sum(np.array([lo.dimensions for lo in self.likobs])-self.numcpars)

    def saveResiduals(self, chainsdir):
        """
        Saving the residuals for all likobs
        """
        for likob in self.likobs:
            likob.saveResiduals(chainsdir)

    def saveModelParameters(self, ptfile):
        """
        Do what the usual saveModelParameters does, but now for the full
        complete model.

        @param ptfile:  Name of the file with the descriptions
        """
        fil = open(filename, "w")

        parind = 0
        psrind = 0
        # THIS IS NOT YET FULLY IMPLEMENTED!!!
        for ll, likob in enumerate(self.likobs):
            for ii in range(len(likob.pardes)):
                fil.write("{0:d} \t{1:d} \t{2:s} \t{3:s} \t{4:s} \t{5:s} \t{6:s}\n".format(\
                        likob.pardes[ii]['index'],
                        likob.pardes[ii]['pulsar'],
                        likob.pardes[ii]['sigtype'],
                        likob.pardes[ii]['correlation'],
                        likob.pardes[ii]['name'],
                        likob.pardes[ii]['id'],
                        likob.pardes[ii]['pulsarname']))

    def logprior(self, pars):
        """
        Based on the combined parameters, calculate the log-prior using the
        individual pulsar objects

        @param pars:    Parameters of the combined prior

        @return:        Log-prior
        """

        cpars = pars[-self.numcpars:]

        lpr = 0.0   # Log-prior

        pind = 0
        for ii, likob in enumerate(self.likobs):
            nppars = likob.dimensions - self.numcpars
            psrpars = np.append(pars[pind:pind+nppars], cpars)
            pind += nppars

            p_lpr = likob.logprior(psrpars)

            lpr += p_lpr

        return lpr

    def loglikelihood(self, pars):
        """
        Based on the combined parameters, calculate the log-likelihood using the
        individual pulsar objects

        @param pars:    Parameters of the combined prior

        @return:        Log-likelihood
        """

        cpars = pars[-self.numcpars:]

        ll = 0.0   # Log-likelihood

        pind = 0
        for ii, likob in enumerate(self.likobs):
            nppars = likob.dimensions - self.numcpars
            psrpars = np.append(pars[pind:pind+nppars], cpars)
            pind += nppars

            p_ll = likob.loglikelihood(psrpars)

            ll += p_ll

        return ll

    def logposterior(self, pars):
        """
        Based on the combined parameters, calculate the log-posterior using the
        individual pulsar objects

        @param pars:    Parameters of the combined prior

        @return:        Log-posterior
        """
        return self.logprior(pars) + self.loglikelihood(pars)

def RunImportance_SepLik_MCMC(seplik, steps, chainsdir, burnin=1000):
    """
    Using the MCMC chains that have already been loaded into the separable
    likelihood object, run an importance sampling method for 'steps' steps. 
    """
    if len(seplik.likobs) != len(spelik.single_lp):
        raise IOError("Separable likelihood object does not contain MCMC chains")
    
    # Save the parameters to file
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')
