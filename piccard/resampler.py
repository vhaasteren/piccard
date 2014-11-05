#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
Resampling techniques to search for gravitational-waves using the single-pulsar
MCMC chains of the Gibbs sampler
"""

from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import h5py as h5
import matplotlib.pyplot as plt
import os as os
import glob
import sys
import json
import tempfile

from .piccard import *
from . import pytwalk                  # Internal module
from . import pydnest                  # Internal module
from . import rjmcmchammer as rjemcee  # Internal module
from . import PTMCMC_generic as ptmcmc
from .triplot import *

def resampler_scanGW(dirlist, burnin=0, thin=1, sampler='auto'):
    """
    This function accepts a list of directories with MCMC chains (most likely
    from the Gibbs sampler). Crucial about the MCMC chains is that each includes
    the Fourier modes, and that the Fourier modes are all at the same frequency.

    @param dirlist:     List of directories with MCMC chains
    """
    Npsr = len(dirlist)

    l_ll, l_lp, l_chain, l_labels, l_pulsarid, l_pulsarname, l_stype, l_mlpso, \
            l_mlpsopars) = [], [], [], [], [], [], [], [], []
    l_Mmatind = []

    for ii, direc in enumerate(dirlist):
        (llf, lpf, chainf, labels, pulsarid, pulsarname, stype, mlpso, mlpsopars) = \
                ReadMCMCFile(direc, sampler=sampler, incextra=True)

        if len(set(pulsarname)) > 0:
            raise ValueError("Resampling requires single-pulsar MCMC chains!")

        l_ll.append(llf[burnin::thin])
        l_lp.append(lpf[burnin::thin])
        l_chain.append(chainf[burnin::thin, :])
        l_labels.append(labels)
        l_pulsarid.append(pulsarid)
        l_pulsarname.append(pulsarname)
        l_stype.append(stype)
        l_mlpso.append(mlpso)
        l_mlpsopars.append(mlpsopars)

        # Find the indices for red noise
        l_Mmatind.append(np.array(stype) == 'Fmat')
