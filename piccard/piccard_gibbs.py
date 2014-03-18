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
        a.append = ahat + np.dot(Li, np.random.randn(Li.shape[0]))

    return a

def gibbs_sample_N(self):
    pass



# For constructPhiAndTheta, use phimat=False
