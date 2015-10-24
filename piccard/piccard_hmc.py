#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss

from piccard import *


class likelihoodWrapper_old(object):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This is an
    abstract class that gives the basic functionality for any likelihood wrapper

    Used for coordinate transformations
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """
        Initialize the likelihoodWrapper, just like ptaLikelihood

        Extra kwarg is 'wrapperclass'. If not present, default is ptaLikelihood.
        Otherwise, this transform can be placed on top of another transform.

        """
        if not 'wrapperclass' in kwargs:
            self.likob = ptaLikelihood(h5filename, jsonfilename, **kwargs)
        else:
            # TODO: by allowing wrapperclass to be a list, we could chain
            #       multiple transformations on top of each other
            wc = kwargs.pop('wrapperclass')
            self.likob = wc(h5filename, jsonfilename, **kwargs)
        
        self.initBounds()
        self._cached = False        # We'll use a simple type of caching
        self._p = None              # Cached values for coordinates
        self._x = None              # Cached values for coordinates

    def initBounds(self):
        """Initialize the parameter bounds"""
        self.a, self.b = self.likob.pmin, self.likob.pmax

    def initFromFile(self, filename, **kwargs):
        self.likob.initFromFile(filename, **kwargs)
        
        self.initBounds()

    def getPulsarNames(self, **kwargs):
        return self.likob.getPulsarNames(**kwargs)

    def makeModelDict(self, **kwargs):
        return self.likob.makeModelDict(**kwargs)

    def initModelFromFile(self, filename, **kwargs):
        self.likob.initModelFromFile(filename, **kwargs)
        
        self.initBounds()

    def writeModelToFile(self, filename):
        self.likob.writeModelToFile(filename)

    def initModel(self, fullmodel, **kwargs):
        self.likob.initModel(fullmodel, **kwargs)
        
        self.initBounds()

    def getGibbsModelParameterList(self):
        return self.likob.getGibbsModelParameterList()

    def saveModelParameters(self, filename):
        self.likob.saveModelParameters(filename)

    def saveResiduals(self, outputdir):
        self.likob.saveResiduals(outputdir)

    def forward(self, x):
        """Forward transform
        """
        raise NotImplementedError("Create this method in derived class!")
        return None

    def backward(self, p):
        """Backward transform
        """
        raise NotImplementedError("Create this method in derived class!")
        return None
    
    def logjacobian_grad(self, p):
        """Return the log of the Jacobian at point p"""
        raise NotImplementedError("Create this method in derived class!")
        return None

    def dxdp(self, p):
        """Derivative of x wrt p (jacobian for chain-rule) - diagonal"""
        raise NotImplementedError("Create this method in derived class!")
        return None

    def logposterior_grad(self, p):
        """The log-posterior in the new coordinates"""
        lpr, lpr_grad = self.logprior_grad(p)
        ll, ll_grad = self.loglikelihood_grad(p)
        return ll+lpr, ll_grad+lpr_grad

    def loglikelihood_grad(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)
        ll, ll_grad = self.likob.mark13loglikelihood_old(x)
        lj, lj_grad = self.logjacobian_grad(p)
        return ll+lj, ll_grad*self.dxdp(p)+lj_grad

    def logprior_grad(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        x = self.backward(p)
        lp, lp_grad = self.likob.mark13logprior_fast(x)
        return lp, lp_grad*self.dxdp(p)

    def logposterior(self, p):
        """The log-posterior in the new coordinates"""
        lpr, lpr_grad = self.logprior_grad(p)
        ll, ll_grad = self.loglikelihood_grad(p)
        return ll+lpr

    def loglikelihood(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)

        ll, ll_grad = self.likob.mark13loglikelihood_old(x)
        lj, lj_grad = self.logjacobian_grad(p)
        return ll+lj

    def logprior(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        x = self.backward(p)
        lp, lp_grad = self.likob.mark13logprior_fast(x)
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
        if self.likob.pmin is not None:
            return self.forward(self.likob.pmin)
        return None

    @pmin.setter
    def pmin(self, value):
        self.likob.pmin = self.backward(value)

    @property
    def pmax(self):
        if self.likob.pmax is not None:
            return self.forward(self.likob.pmax)
        return None

    @pmax.setter
    def pmax(self, value):
        self.likob.pmax = self.backward(value)

    @property
    def pstart(self):
        if self.likob.pstart is not None:
            return self.forward(self.likob.pstart)
        return None

    @pstart.setter
    def pstart(self, value):
        self.likob.pstart = self.backward(value)

    @property
    def pwidth(self):
        if self.likob.pwidth is not None:
            x0 = self.forward(self.likob.pstart)
            x1 = self.forward(self.likob.pstart+self.likob.pwidth)
            return x1-x0
        return None

    @pwidth.setter
    def pwidth(self, value):
        self.likob.pwidth = self.backward(value)

    @property
    def dimensions(self):
        return self.likob.dimensions

    @property
    def pardes(self):
        return self.likob.pardes
    
    @property
    def likfunc(self):
        return self.likob.likfunc

    @property
    def ptapsrs(self):
        return self.likob.ptapsrs

    @property
    def ptasignals(self):
        return self.likob.ptasignals

    @property
    def npf(self):
        return self.likob.npf

    @property
    def npfdm(self):
        return self.likob.npfdm

    @property
    def npu(self):
        return self.likob.npu

    @property
    def Phivec(self):
        return self.likob.Phivec

    @property
    def Svec(self):
        return self.likob.Svec

    @property
    def d_Phivec_d_param(self):
        return self.likob.d_Phivec_d_param

    @property
    def d_Svec_d_param(self):
        return self.likob.d_Svec_d_param

    @property
    def d_Thetavec_d_param(self):
        return self.likob.d_Thetavec_d_param


class hmcLikelihood_old(likelihoodWrapper_old):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for some/all parameters from an interval to all
    real numbers.
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the hmcLikelihood_old with a ptaLikelihood object"""
        super(hmcLikelihood_old, self).__init__(h5filename, jsonfilename, **kwargs)

    def initBounds(self):
        """Initialize the parameter bounds"""
        super(hmcLikelihood_old, self).initBounds()
        self.msk = self.likob.interval

        self.setLowLevelStart()

    def setLowLevelStart(self):
        """Set the starting-point of the low-level parameters, now that we have
        a stingray transform"""
        low_level_pars = ['timingmodel_xi', 'fouriermode_xi',
                'dmfouriermode_xi', 'jittermode_xi']

        if self.pstart is None:
            return

        pstart = self.pstart

        for ii, m2signal in enumerate(self.ptasignals):
            if m2signal['stype'] in low_level_pars:
                # We have parameters that need to be re-set
                sind = m2signal['parindex']
                msk = m2signal['bvary']
                npars = np.sum(msk)
                #sigstart = m2signal['pstart'][msk]  # Original pars

                # We use random starting positions, close to zero
                sigstart = np.ones(npars) * (0.1+0.05*np.random.randn(npars))

                pstart[sind:sind+npars] = sigstart

        # We cannot do in-place because of possible lower-level transformations
        self.pstart = pstart

    def forward(self, x):
        """Forward transform the real coordinates (on the interval) to the
        transformed coordinates (on all real numbers)
        """
        p = np.atleast_2d(x.copy())
        posinf, neginf = (self.a == x), (self.b == x)
        m = self.msk & ~(posinf | neginf)
        p[:,m] = np.log((p[:,m] - self.a[m]) / (self.b[m] - p[:,m]))
        p[:,posinf] = np.inf
        p[:,neginf] = -np.inf
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
    
    def logjacobian_grad(self, p):
        """Return the log of the Jacobian at point p"""
        m = self.msk
        lj = np.sum( np.log(self.b[m]-self.a[m]) + p[m] -
                2*np.log(1.0+np.exp(p[m])) )

        lj_grad = np.zeros_like(p)
        lj_grad[m] = (1 - np.exp(p[m])) / (1 + np.exp(p[m]))
        return lj, lj_grad

    def dxdp(self, p):
        """Derivative of x wrt p (jacobian for chain-rule) - diagonal"""
        pp = np.atleast_2d(p)
        m = self.msk
        d = np.ones_like(pp)
        d[:,m] = (self.b[m]-self.a[m])*np.exp(pp[:,m])/(1+np.exp(pp[:,m]))**2
        return d.reshape(p.shape)


