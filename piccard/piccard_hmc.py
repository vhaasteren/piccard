#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss

from piccard import *

class hmcLikelihood(object):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all parameters from an interval to all real
    numbers.
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the hmcLikelihood with a ptaLikelihood object"""
        self.likob = ptaLikelihood(h5filename, jsonfilename, **kwargs)

        self.a, self.b = self.likob.pmin, self.likob.pmax
        self.msk = self.likob.interval

    def initFromFile(self, filename, **kwargs):
        self.likob.initFromFile(filename, **kwargs)

        self.a, self.b = self.likob.pmin, self.likob.pmax
        self.msk = self.likob.interval

    def getPulsarNames(self, **kwargs):
        return self.likob.getPulsarNames(**kwargs)

    def makeModelDict(self, **kwargs):
        return self.likob.makeModelDict(**kwargs)

    def initModelFromFile(self, filename, **kwargs):
        self.likob.initModelFromFile(filename, **kwargs)

        self.a, self.b = self.likob.pmin, self.likob.pmax
        self.msk = self.likob.interval

    def writeModelToFile(self, filename):
        self.likob.writeModelToFile(filename)

    def initModel(self, fullmodel, **kwargs):
        self.likob.initModel(fullmodel, **kwargs)

        self.a, self.b = self.likob.pmin, self.likob.pmax
        self.msk = self.likob.interval

    def getGibbsModelParameterList(self):
        return self.likob.getGibbsModelParameterList()

    def saveModelParameters(self, filename):
        self.likob.saveModelParameters(filename)

    def saveResiduals(self, outputdir):
        self.likob.saveResiduals(outputdir)

    def forward(self, x):
        """Forward transform the real coordinates (on the interval) to the
        transformed coordinates (on all real numbers)
        """
        p = np.atleast_2d(x.copy())
        m = self.msk
        p[:,m] = np.log((p[:,m] - self.a[m]) / (self.b[m] - p[:,m]))
        return p.reshape(x.shape)

    def backward(self, p):
        """Backward transform the transformed coordinates (on all real numbers)
        to the real coordinates (on the interval)
        """
        x = np.atleast_2d(p.copy())
        m = self.msk
        x[:,m] = (self.b[m] - self.a[m]) * np.exp(x[:,m]) / (1 +
                np.exp(x[:,m])) + self.a[m]
        return x.reshape(p.shape)
    
    def logjacobian(self, p):
        """Return the log of the Jacobian at point p"""
        m = self.msk
        lj = np.sum( np.log(self.b[m]-self.a[m]) + p[m] -
                2*np.log(1.0+np.exp(p[m])) )

        # Alternate version as a function of x:
        #x = self.backward(p)
        #frac = 1.0/(x-self.a) + 1.0/(self.b-x)
        #lj = -np.sum(np.log(frac))

        lj_grad = np.zeros_like(p)
        lj_grad[m] = (1 - np.exp(p[m])) / (1 + np.exp(p[m]))
        return lj, lj_grad

    def dxdp(self, p):
        """Derivative of x wrt p (jacobian for chain-fule)"""
        pp = np.atleast_2d(p)
        m = self.msk
        d = np.ones_like(pp)
        d[:,m] = (self.b[m]-self.a[m])*np.exp(pp[:,m])/(1+np.exp(pp[:,m]))**2
        return d.reshape(p.shape)

    def logposterior_grad(self, p):
        """The log-posterior in the new coordinates"""
        lpr, lpr_grad = self.logprior_grad(p)
        ll, ll_grad = self.loglikelihood_grad(p)
        return ll+lpr, ll_grad+lpr_grad

    def loglikelihood_grad(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)
        ll, ll_grad = self.likob.mark13loglikelihood(x)
        lj, lj_grad = self.logjacobian(p)
        return ll+lj, ll_grad*self.dxdp(p)+lj_grad

    def logprior_grad(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        x = self.backward(p)
        lp, lp_grad = self.likob.mark13logprior(x)
        return lp, lp_grad*self.dxdp(p)

    def logposterior(self, p):
        """The log-posterior in the new coordinates"""
        lpr, lpr_grad = self.logprior(p)
        ll, ll_grad = self.loglikelihood(p)
        return ll+lpr

    def loglikelihood(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)

        ll, ll_grad = self.likob.mark13loglikelihood(x)
        lj, lj_grad = self.logjacobian(p)
        return ll+lj

    def logprior(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        x = self.backward(p)
        lp, lp_grad = self.likob.mark13logprior(x)
        return lp
    
    def addPriorDraws(self, which='hyper'):
        self.likob.addPriorDraws(which=which)

    @property
    def _prior_draw_signal(self):
        return self.likob._prior_draw_signal

    def drawFromPrior(self, p, iter, beta):
        parameters = self.backward(p)
        pq, qxy = self.likob.drawFromPrior(parameters, iter, beta)
        q = self.forward(pq)
        return q, qxy

    @property
    def pmin(self):
        pm
        return np.ones_like(self.a)*-np.inf

    @property
    def pmax(self):
        return np.ones_like(self.a)*np.inf

    @property
    def pstart(self):
        return self.forward(self.likob.pstart)

    @property
    def pwidth(self):
        x0 = self.forward(self.likob.pstart)
        x1 = self.forward(self.likob.pstart+self.likob.pwidth)
        return x1-x0

    @property
    def dimensions(self):
        return self.likob.dimensions

    @property
    def pardes(self):
        return self.likob.pardes
    
    @property
    def likfunc(self):
        return self.likob.likfunc
