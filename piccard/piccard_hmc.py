#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss

from piccard import *


class likelihoodWrapper(object):
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

    # Idea for Caching: always work with the self._x, and self._p
    #                   in the beginning, do the calculations, which set those
    #                   values. After any function, de-cache the function
    #                   Can be done with a decorator?

    def logposterior_grad(self, p):
        """The log-posterior in the new coordinates"""
        lpr, lpr_grad = self.logprior_grad(p)
        ll, ll_grad = self.loglikelihood_grad(p)
        return ll+lpr, ll_grad+lpr_grad

    def loglikelihood_grad(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)
        ll, ll_grad = self.likob.loglikelihood_grad(x)
        lj, lj_grad = self.logjacobian_grad(p)
        return ll+lj, ll_grad*self.dxdp(p)+lj_grad

    def logprior_grad(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        x = self.backward(p)
        lp, lp_grad = self.likob.logprior_grad(x)
        return lp, lp_grad*self.dxdp(p)

    def logposterior(self, p):
        """The log-posterior in the new coordinates"""
        lpr, lpr_grad = self.logprior_grad(p)
        ll, ll_grad = self.loglikelihood_grad(p)
        return ll+lpr

    def loglikelihood(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)

        ll, ll_grad = self.likob.loglikelihood_grad(x)
        lj, lj_grad = self.logjacobian_grad(p)
        return ll+lj

    def logprior(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        x = self.backward(p)
        lp, lp_grad = self.likob.logprior_grad(x)
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
        return self.forward(self.likob.pmin)

    @property
    def pmax(self):
        return self.forward(self.likob.pmax)

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

    @property
    def ptapsrs(self):
        return self.likob.ptapsrs

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


class hmcLikelihood(likelihoodWrapper):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for some/all parameters from an interval to all
    real numbers.
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the hmcLikelihood with a ptaLikelihood object"""
        super(hmcLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

    def initBounds(self):
        """Initialize the parameter bounds"""
        super(hmcLikelihood, self).initBounds()
        self.msk = self.likob.interval

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



class stingrayLikelihood(likelihoodWrapper):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the stingrayLikelihood with a ptaLikelihood object"""
        super(stingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

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
    









class hmcLikelihood_old(object):
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
    
    def logjacobian_grad(self, p):
        """Return the log of the Jacobian at point p"""
        m = self.msk
        lj = np.sum( np.log(self.b[m]-self.a[m]) + p[m] -
                2*np.log(1.0+np.exp(p[m])) )

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
        lpr, lpr_grad = self.logprior(p)
        ll, ll_grad = self.loglikelihood(p)
        return ll+lpr

    def loglikelihood(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)

        ll, ll_grad = self.likob.mark13loglikelihood(x)
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

    @property
    def ptapsrs(self):
        return self.likob.ptapsrs

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
