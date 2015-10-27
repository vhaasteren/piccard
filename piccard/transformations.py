#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
from functools import partial

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

        # We use a simplistic form of caching in stingrayLikeli8hood
        # TODO: use descriptor decorators
        self._cachefunc = None      # For caching tracking
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

    def fullforward(self, x):
        """Assume these are 'original' parameters, and transform them to this
        level of transformation by applying all the transformations in the
        chain"""
        curob = self
        obj_list = []
        p = x.copy()

        while isinstance(curob, likelihoodWrapper):
            obj_list.append(curob)
            curob = curob.likob

        while len(obj_list):
            curob = obj_list.pop()
            p = curob.forward(p)

        return p

    def fullbackward(self, p):
        """Transform these parameters to 'original' parameters, all the way down
        the transformation chain"""
        curob = self
        x = p.copy()

        while isinstance(curob, likelihoodWrapper):
            x = curob.backward(x)
            curob = curob.likob

        return x

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

    def dxdp_nondiag(self, p, ll_grad):
        """Non-diagonal derivative of x wrt p (jacobian for chain-rule)

        Dealt with (optionally) separately, for efficiency

        Can be overwritten in a subclass
        """
        return np.zeros_like(p)

    def logposterior_grad(self, p):
        """The log-posterior in the new coordinates"""
        """ # Old version:
        lpr, lpr_grad = self.logprior_grad(p)
        ll, ll_grad = self.loglikelihood_grad(p)
        return ll+lpr, ll_grad+lpr_grad
        """
        # We need the prior, and the likelihood, but all from one function
        x = self.backward(p)
        lp, lp_grad = self.likob.logposterior_grad(x)
        lj, lj_grad = self.logjacobian_grad(p)
        return lp+lj, lp_grad*self.dxdp(p)+lj_grad+self.dxdp_nondiag(p, lp_grad)

    def loglikelihood_grad(self, p):
        """The log-likelihood in the new coordinates"""
        x = self.backward(p)
        ll, ll_grad = self.likob.loglikelihood_grad(x)
        lj, lj_grad = self.logjacobian_grad(p)
        return ll+lj, ll_grad*self.dxdp(p)+lj_grad+self.dxdp_nondiag(p, ll_grad)

    def logprior_grad(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        x = self.backward(p)
        lp, lp_grad = self.likob.logprior_grad(x)
        return lp, lp_grad*self.dxdp(p)+self.dxdp_nondiag(p, lp_grad)

    def logposterior(self, p):
        """The log-posterior in the new coordinates"""
        #lpr, lpr_grad = self.logprior_grad(p)
        #ll, ll_grad = self.loglikelihood_grad(p)
        #return ll+lpr
        lp, lp_grad = self.logposterior_grad(p)
        return lp

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

    def set_hyper_pars(self, p, calc_gradient=True):
        """Set all the hyper-parameter dependents"""
        self.likob.set_hyper_pars(p, calc_gradient=calc_gradient)

    def set_det_sources(self, p, calc_gradient=True):
        """Set all the deterministic source dependents"""
        self.likob.set_det_sources(p, calc_gradient=calc_gradient)
    
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
    def interval(self):
        return self.likob.interval

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


class intervalLikelihood(likelihoodWrapper):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for some/all parameters from an interval to all
    real numbers.
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the intervalLikelihood with a ptaLikelihood object"""
        super(intervalLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

    def initBounds(self):
        """Initialize the parameter bounds"""
        super(intervalLikelihood, self).initBounds()
        self.msk = self.interval

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

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the stingrayLikelihood with a ptaLikelihood object"""
        super(stingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

    def initBounds(self):
        """Initialize the parameter bounds"""
        super(stingrayLikelihood, self).initBounds()

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
                #sigstart = np.ones(npars) * (0.1+0.05*np.random.randn(npars))
                sigstart = np.ones(npars) * 0.1

                pstart[sind:sind+npars] = sigstart

        # We cannot do in-place because of possible lower-level transformations
        # Need to set the hyper parameters in some way
        self.pstart = pstart

    def cache(self, pars, func, direction='backward', calc_gradient=True):
        """Perform the forward coordinate transformation, and cache the result
        
        TODO: Make this functionality into a descriptor decorator
        TODO: Keep track of whether or not we have gradients
        """
        if self._cachefunc is None:
            self._cachefunc = func

            # Set hyper parameters
            self.stingray_transformation(pars, calc_gradient=calc_gradient,
                    set_hyper_pars=True)

            if direction == 'forward':
                self._x = np.copy(pars)
                self._p = self.forward(self._x)
            elif direction == 'backward':
                self._p = np.copy(pars)
                self._x = self.backward(self._p)
            else:
                raise ValueError("Direction of transformation unknown")

            # Update all the deterministic sources
            self.set_det_sources(self._x)

    def uncache(self, func):
        """Perform the forward coordinate transformation, and cache the result
        
        TODO: Make this functionality into a descriptor decorator
        """
        if self._cachefunc == func:
            self._cachefunc = None
            #self._x = None
            #self._p = None

    def have_cache(self):
        """Whether or not we have values cache already"""
        return self._cachefunc is not None

    def stingray_transformation(self, p, calc_gradient=True, set_hyper_pars=True):
        """Perform the stingray transformation

        NOTE: exactly as was done in ptaLikelihood

        :param p:
            Stingray parameters, to be transformed (full array)

        :param transform:
            'forward', 'backward', 'none'
            What direction to perfomr the transform in

        :param calc_gradient:
            When True, calculate the gradients
        """
        if set_hyper_pars:
            self.set_hyper_pars(p, calc_gradient=calc_gradient)

        log_jacob = 0.0
        gradient = np.zeros_like(p)
        d_b_d_xi = np.ones_like(p)      # If nothing else, it's one
        d_b_d_B = np.zeros_like(p)
        mu = np.zeros_like(p)           # Mean of transformation
        sigma = np.ones_like(p)         # Sigma of transformation

        for ii, psr in enumerate(self.ptapsrs):
            if psr.timingmodelind is not None:
                # Have a timing model parameter stingray transformation
                Sigmavec = 1.0/psr.ZNZ_srvec[psr.Zmask_M_only]  # No hyper pars
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.ZNy_srvec[psr.Zmask_M_only]
                index = psr.timingmodelind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                log_jacob += np.sum(np.log(std))
                d_b_d_xi[slc] = std

            if psr.fourierind is not None:
                # Have a red noise stingray transformation
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc = slice(findex, findex+nfs)
                phivec = self.Phivec[fslc] + self.Svec[:nfs]

                Sigmavec = 1.0/(psr.ZNZ_srvec[psr.Zmask_F_only] + 1.0 / phivec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.ZNy_srvec[psr.Zmask_F_only]
                index = psr.fourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                if calc_gradient:
                    # Red noise
                    for key, value in self.d_Phivec_d_param.iteritems():
                        # Problem here is that value is length of full Phivec.
                        # The rest is shorter.
                        # Should we slice value, or just store less?
                        d_lj_d_phi = 0.5 * Sigmavec / phivec**2
                        gradient[key] += np.sum(d_lj_d_phi * value[fslc])

                    # GW signals
                    for key, value in self.d_Svec_d_param.iteritems():
                        d_lj_d_phi = 0.5 * Sigmavec / phivec**2
                        gradient[key] += np.sum(d_lj_d_phi * value[fslc])

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / phivec**2
                d_mean_d_B = mean * Sigmavec / phivec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

            if psr.dmfourierind is not None:
                # Have a dm noise stingray transformation
                fdmindex = np.sum(self.npfdm[:ii])
                nfdms = self.npfdm[ii]
                thetavec = self.Thetavec[fdmindex:fdmindex+nfdms]

                Sigmavec = 1.0/(psr.ZNZ_srvec[psr.Zmask_D_only] + 1.0 / thetavec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.ZNy_srvec[psr.Zmask_D_only]
                index = psr.dmfourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                if calc_gradient:
                    for key, value in self.d_Thetavec_d_param.iteritems():
                        d_lj_d_theta = 0.5 * Sigmavec / thetavec**2
                        gradient[key] += np.sum(d_lj_d_theta * value)

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / thetavec**2
                d_mean_d_B = mean * Sigmavec / thetavec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

            if psr.jitterind is not None:
                # Have an ecor stingray transformation
                uindex = np.sum(self.npu[:ii])
                nus = self.npu[ii]

                Sigmavec = 1.0/(psr.ZNZ_srvec[psr.Zmask_U_only] + \
                        1.0 / psr.Jvec)
                std = np.sqrt(Sigmavec)        # No hyper pars
                mean = Sigmavec * psr.ZNy_srvec[psr.Zmask_U_only]
                index = psr.jitterind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                if calc_gradient:
                    for key, value in psr.d_Jvec_d_param.iteritems():
                        d_lj_d_J = 0.5 * Sigmavec / psr.Jvec**2
                        gradient[key] += np.sum(d_lj_d_J * value)

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / psr.Jvec**2
                d_mean_d_B = mean * Sigmavec / psr.Jvec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

        self._mu = mu                   # Mean of stingray transform
        self._sigma = sigma             # Slope of stingray transform
        self._log_jacob = log_jacob     # Log-jacobian of transform
        self._gradient = gradient       # Gradient of log-jacobian
        self._d_b_d_xi = d_b_d_xi       # d_x_d_p
        self._d_b_d_B = d_b_d_B         # d_x_d_B, with B hyper-pars

    def dxdp_nondiag(self, p, ll_grad):
        """Non-diagonal derivative of x wrt p (jacobian for chain-rule)

        Dealt with (optionally) separately, for efficiency, since this would
        otherwise be an O(p^2) operation
        """
        # TODO: speed increase for multiple pulsars by not looping over all
        #       pulsars AND all d_Phivec_d_p simultaneously. Wastes lots of time
        extra_grad = np.zeros_like(p)

        for ii, psr in enumerate(self.ptapsrs):
            if psr.fourierind is not None:
                findex = np.sum(self.npf[:ii])
                ind = psr.fourierind
                nfreq = self.npf[ii]
                pslc = slice(ind, ind+nfreq)
                fslc = slice(findex, findex+nfreq)

                # Hyper parameters on Phi
                for key, d_Phivec_d_p in self.d_Phivec_d_param.iteritems():
                    extra_grad[key] += np.sum(ll_grad[pslc] * 
                            self._d_b_d_B[pslc] * d_Phivec_d_p[fslc])

                for key, d_Svec_d_p in self.d_Svec_d_param.iteritems():
                    extra_grad[key] += np.sum(ll_grad[pslc] * 
                            self._d_b_d_B[pslc] * d_Svec_d_p[fslc])

            if psr.dmfourierind is not None:
                ind = psr.dmfourierind
                nfreqdm = self.npfdm[ii]
                pslc = slice(ind, ind+nfreqdm)
                # Hyper parameters on Theta
                for key, d_Thetavec_d_p in self.d_Thetavec_d_param.iteritems():
                    extra_grad[key] += np.sum(ll_grad[pslc] * 
                            self._d_b_d_B[pslc] * d_Thetavec_d_p)

            if psr.jitterind is not None:
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                # Hyper parameters on Jvec
                for key, d_Jvec_d_p in psr.d_Jvec_d_param.iteritems():
                    extra_grad[key] += np.sum(ll_grad[pslc] * 
                            self._d_b_d_B[pslc] * d_Jvec_d_p)

        return extra_grad

    def forward(self, x):
        """Forward transform the real coordinates (with Stingray) to the
        transformed coordinates (more Gaussian-like)

        NOTE: forward transformation cannot yield proper gradients. But those
              would not be necessary anyways.

        """
        p = np.atleast_2d(x.copy())

        if p.shape[0] == 1:
            self.cache(p[0,:], self.forward, 
                    direction='forward', calc_gradient=False)
            p[0,:] = (p[0,:] - self._mu) / self._sigma
            self.uncache(self.forward)
        else:
            if self.have_cache():
                raise RuntimeError("Invalid transform from caching function")

            for ii, pc in enumerate(p):
                self.cache(pc, self.forward, 
                        direction='forward', calc_gradient=False)
                p[ii,:] = (pc - self._mu) / self._sigma
                self.uncache(self.forward)

        return p.reshape(x.shape)

    def backward(self, p):
        """Backward transform the transformed coordinates (Gaussian-like) to the
        real coordinates (with Stingray)

        NOTE: forward transformation cannot yield proper gradients. But those
              would not be necessary anyways.

        """
        x = np.atleast_2d(p.copy())

        if x.shape[0] == 1:
            self.cache(x[0,:], self.backward, 
                    direction='backward', calc_gradient=False)
            x[0,:] = x[0,:] * self._sigma +  self._mu
            self.uncache(self.backward)
        else:
            if self.have_cache():
                raise RuntimeError("Invalid transform from caching function")

            for ii, xc in enumerate(x):
                self.cache(xc, self.backward, 
                        direction='backward', calc_gradient=False)
                x[ii,:] = xc * self._sigma + self._mu
                self.uncache(self.backward)

        return x.reshape(p.shape)
    
    def logjacobian_grad(self, p):
        """Return the log of the Jacobian at point p"""
        #self.stingray_transformation(p, calc_gradient=True)
        self.cache(p, self.logjacobian_grad, direction='backward',
                calc_gradient=True)
        self.uncache(self.logjacobian_grad)

        return self._log_jacob, self._gradient

    def dxdp(self, p):
        #self.stingray_transformation(p, calc_gradient=False)
        self.cache(p, self.dxdp, direction='backward',
                calc_gradient=False)
        self.uncache(self.dxdp)

        return self._d_b_d_xi

    def logposterior_grad(self, p):
        """The log-posterior in the new coordinates"""
        self.cache(p, self.logposterior_grad, direction='backward',
                calc_gradient=True)
        lpr, lpr_grad = self.logprior_grad(self._p)
        ll, ll_grad = self.loglikelihood_grad(self._p)
        self.uncache(self.logposterior_grad)
        return ll+lpr, ll_grad+lpr_grad

    def loglikelihood_grad(self, p):
        """The log-likelihood in the new coordinates"""
        #x = self.backward(p)
        self.cache(p, self.loglikelihood_grad, direction='backward',
                calc_gradient=True)

        ll, ll_grad = self.likob.loglikelihood_grad(self._x, set_hyper_pars=False)
        lj, lj_grad = self.logjacobian_grad(self._p)

        lp = ll+lj
        lp_grad = ll_grad*self.dxdp(self._p) + \
                lj_grad + self.dxdp_nondiag(self._p, ll_grad)

        self.uncache(self.loglikelihood_grad)
        #return ll+lj, ll_grad*self.dxdp(p)+lj_grad+self.dxdp_nondiag(p, ll_grad)
        return lp, lp_grad

    def logprior_grad(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        self.cache(p, self.logprior_grad, direction='backward',
                calc_gradient=True)
        #x = self.backward(p)

        lp, lp_grad = self.likob.logprior_grad(self._x)
        lpt = lp
        lpt_grad = lp_grad*self.dxdp(self._p) + \
                self.dxdp_nondiag(self._p, lp_grad)

        self.uncache(self.logprior_grad)
        #return lp, lp_grad*self.dxdp(p)+self.dxdp_nondiag(p, lp_grad)
        return lpt, lpt_grad

    def logposterior(self, p):
        """The log-posterior in the new coordinates"""
        self.cache(p, self.logposterior, direction='backward',
                calc_gradient=False)

        lpr, lpr_grad = self.logprior_grad(self._p)
        ll, ll_grad = self.loglikelihood_grad(self._p)

        self.uncache(self.logposterior)
        return ll+lpr

    def loglikelihood(self, p):
        """The log-likelihood in the new coordinates"""
        #x = self.backward(p)
        self.cache(p, self.loglikelihood, direction='backward',
                calc_gradient=False)

        ll, ll_grad = self.likob.loglikelihood_grad(self._x)
        lj, lj_grad = self.logjacobian_grad(self._p)

        self.uncache(self.loglikelihood)
        return ll+lj

    def logprior(self, p):
        """The log-prior in the new coordinates
        
        Note: to preserve logposterior = loglikelihood + logprior, this term
              does not include the jacobian transformation
        """
        #x = self.backward(p)
        self.cache(p, self.logprior, direction='backward', calc_gradient=False)

        lp, lp_grad = self.likob.logprior_grad(self._x)

        self.uncache(self.logprior)
        return lp
    

def hmcLikelihood(h5filename=None, jsonfilename=None, **kwargs):
    """Wrapper for the compound of the stingray transformation and the interval
    transformation
    """
    if 'wrapperclass' in kwargs:
        raise ValueError("hmcLikelihood already pre-sets wrapperclass")

    return intervalLikelihood(h5filename=h5filename,
            jsonfilename=jsonfilename,
            wrapperclass=stingrayLikelihood,
            **kwargs)
    
