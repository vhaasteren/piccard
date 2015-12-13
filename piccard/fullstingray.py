#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
from functools import partial

from transformations import *
from choleskyext import *

# This file contains the class for the full stingray transformation. We'll use
# the Cholesky derivative algorithm from the Cholesky extension

class fullStingrayLikelihood(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """

    # TODO: The linear transformation is now a matrix equation.
    #       What we need to redefine:
    #           - self.forward  (see below)
    #           - self.backward (see below)
    #           - self._sigma (no longer used)
    #           - self._d_b_d_xi (now 2D: psr.Li)
    #
    # NOTE: Most of these per PTA, but transforms defined per pulsar
    #
    # Detailed plan
    # =============
    #
    # 1) The transforms (forward & backward)
    #    Currently we have x = p * self._sigma + self._mu
    #    Make sure we have a psr.sr_llslice (low-level-parameter slice -- pars)
    #    Calculated in get_par_psr_sigma_inds (just store it I suppose:
    #    slc = psr.sr_llslice)
    #    Change to: self._mu[slc] = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
    #    So that: x[slc] = np.dot(psr.sr_Li, p[slc]) + self._mu[slc]  (loop psrs)
    #             p[scl] = np.dot(psr.sr_L, x[slc]-self._mu[slc])
    #
    # 2) No longer use self._sigma. We have psr.sr_Sigma, psr.sr_L, psr.sr_Li
    #
    # 3) Remove self._d_b_d_xi (used in dxdp)
    #
    # 4) Re-implement self.dxdp. Make it all zeros. We'll implement it in
    #    dxdp_nondiag
    #
    # 5) Re-implement self.logposterior_grad, since it calls dxdp_nondiag twice.
    #    Change it so the call is only made once. In there, we'll have a call to
    #    np.dot(psr.Li, ll_grad2)
    #
    # 6) The log-jacobian is np.sum(np.log(np.diag(psr.sr_Li)))
    #
    # 7) Implement all the hyper-parameter derivatives using the Cython code.
    #    This includes the gradient of the Jacobian
    #    a) d_mu_d_hp  (fancy slicing)
    #    b) d_lj_d_hp  (Cython code - tj)
    #    c) d_L_d_hp   (Cython code - M)

    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the fullStingrayLikelihood with a ptaLikelihood object"""
        super(fullStingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

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
            # TODO: Change this for the real transform
            p[0,:] = (p[0,:] - self._mu) / self._sigma
            self.uncache(self.forward)
        else:
            if self.have_cache():
                raise RuntimeError("Invalid transform from caching function")

            for ii, pc in enumerate(p):
                self.cache(pc, self.forward, 
                        direction='forward', calc_gradient=False)
                # TODO: Change this for the real transform
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
            # TODO: Change this for the real transform
            x[0,:] = x[0,:] * self._sigma +  self._mu
            self.uncache(self.backward)
        else:
            if self.have_cache():
                raise RuntimeError("Invalid transform from caching function")

            for ii, xc in enumerate(x):
                self.cache(xc, self.backward, 
                        direction='backward', calc_gradient=False)
                # TODO: Change this for the real transform
                x[ii,:] = xc * self._sigma + self._mu
                self.uncache(self.backward)

        return x.reshape(p.shape)

    def get_psr_Beta(self, ii, psr):
        """For the mean-part of the stingray transform, we need the full Sigma,
        and therefore the full Beta matrix"""
        Beta_inv_diag = np.zeros(len(psr.sr_ZNZ))

        if psr.fourierind is not None:
            # Have a red noise stingray transformation
            findex = np.sum(self.npf[:ii])
            nfs = self.npf[ii]
            fslc_phi = slice(findex, findex+nfs)
            phivec = self.Phivec[fslc_phi] + self.Svec[:nfs]

            # The diagonal of the Sigma_inv needs to include the hyper parameter
            # constraints
            Beta_inv_diag[psr.Zmask_F_only] = 1.0 / phivec

        if psr.dmfourierind is not None:
            # Have a dm noise stingray transformation
            fdmindex = np.sum(self.npfdm[:ii])
            nfdms = self.npfdm[ii]
            fslc_theta = slice(fdmindex, fdmindex+nfdms)
            thetavec = self.Thetavec[fslc_theta]

            # The diagonal of the Sigma_inv needs to include the hyper parameter
            # constraints
            Beta_inv_diag[psr.Zmask_D_only] = 1.0 / thetavec

        if psr.jitterind is not None:
            # The diagonal of the Sigma_inv needs to include the hyper parameter
            # constraints
            Beta_inv_diag[psr.Zmask_U_only] = 1.0 / psr.Jvec

        return Beta_inv_diag

    def get_psr_Sigma(self, ii, psr, Beta_inv_diag):
        """Obtain Sigma, L, and Li, with Sigma=Li Li^T"""
        Sigma_inv = np.copy(psr.sr_ZNZ)
        Sigma_inv_diag = np.diag(Sigma_inv)

        # Construct the full Sigma matrix
        np.fill_diagonal(Sigma_inv, Sigma_inv_diag + Beta_inv_diag)

        # NOTE: Sigma = L_inv L_inv^T    (just easier that way)
        L = sl.cholesky(Sigma_inv, lower=True)
        Li = sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True)
        cf = (Li, True)

        return sl.cho_solve(cf, np.eye(len(Sigma_inv))), L, Li

    def get_par_psr_sigma_inds(self, ii, psr):
        """Given a pulsar, get a slice object (numpy int array) that contains
        _all_ the indices of the low-level parameters"""
        slc = np.array([], dtype=np.int)

        if psr.timingmodelind is not None:
            slc = np.append(slc, np.arange(psr.timingmodelind,
                    psr.timingmodelind + self.npm[ii]))

        if psr.fourierind is not None:
            slc = np.append(slc, np.arange(psr.fourierind,
                    psr.fourierind + self.npf[ii]))

        if psr.dmfourierind is not None:
            slc = np.append(slc, np.arange(psr.dmfourierind,
                    psr.dmfourierind + self.npfdm[ii]))

        if psr.jitterind is not None:
            slc = np.append(slc, np.arange(psr.jitterind,
                    psr.jitterind + self.npu[ii]))

        return slc

    def stingray_transformation(self, p, calc_gradient=True,
            set_hyper_pars=True):
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
            # For the timing-model stingray of this pulsar, we will be filling
            # some quantities for the low-level parameters, so we can collect
            # results later on
            sr_gamma_frac_den = np.zeros(psr.sr_gamma.shape[1])
            fslc_phi = slice(0)         # Initialize later
            fslc_theta = slice(0)       # Initialize later
            fslc_J = slice(0)           # Initialize later
            mslc = slice(np.sum(self.npm[:ii]), np.sum(self.npm[:ii+1]))
            m_index = psr.timingmodelind
            mslc_par = slice(m_index, m_index+self.npm[ii])
            ntmpars = self.npm[ii]

            psr.sr_Beta_inv = self.get_psr_Beta(ii, psr)
            psr.sr_Sigma, psr.sr_L, psr.sr_Li = \
                    self.get_psr_Sigma(ii, psr, psr.sr_Beta_inv)
            psr.sr_mu = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
            #psr.sr_sigma = np.sqrt(np.diag(psr.sr_Sigma))
            psr.diagSBS = dict()

            # TODO: The linear transformation is now a matrix equation.
            #       CONTINUE EDITING STUFF HERE

            if psr.fourierind is not None:
                # Have a red noise stingray transformation
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc_phi = slice(findex, findex+nfs)
                phivec = self.Phivec[fslc_phi] + self.Svec[:nfs]

                #Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_F_only] + 1.0 / phivec)
                Sigmavec = psr.sr_sigma[psr.Zmask_F_only]**2
                std = np.sqrt(Sigmavec)
                mean = psr.sr_mu[psr.Zmask_F_only]

                index = psr.fourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))


            if psr.dmfourierind is not None:
                # Have a dm noise stingray transformation
                fdmindex = np.sum(self.npfdm[:ii])
                nfdms = self.npfdm[ii]
                fslc_theta = slice(fdmindex, fdmindex+nfdms)
                thetavec = self.Thetavec[fslc_theta]

                Sigmavec = psr.sr_sigma[psr.Zmask_D_only]**2
                std = np.sqrt(Sigmavec)
                mean = psr.sr_mu[psr.Zmask_D_only]

                index = psr.dmfourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))


            if psr.jitterind is not None:
                # Have an ecor stingray transformation
                uindex = np.sum(self.npu[:ii])
                nus = self.npu[ii]
                fslc_J = slice(uindex, uindex+nus)

                Sigmavec = psr.sr_sigma[psr.Zmask_U_only]**2
                std = np.sqrt(Sigmavec)        # No hyper pars
                mean = psr.sr_mu[psr.Zmask_U_only]

                index = psr.jitterind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))


            if psr.timingmodelind is not None:
                # Now have all we need for the Stingray transformation
                Sigmavec = psr.sr_sigma[psr.Zmask_M_only]**2

                std = np.sqrt(Sigmavec)

                # We keep the old estimate of the mean, because is it less
                # biased
                mean = psr.sr_mu[psr.Zmask_M_only]

                index = psr.timingmodelind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                log_jacob += np.sum(np.log(std))

                d_b_d_xi[slc] = std

        if calc_gradient:

            for ii, psr in enumerate(self.ptapsrs):

                # Red noise
                if psr.fourierind is not None:
                    fslc_phi = slice(np.sum(self.npf[:ii]),
                            np.sum(self.npf[:ii+1]))

                    for key, value in self.d_Phivec_d_param.iteritems():
                        # We need to remember diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_F_only] = \
                                psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                                value[fslc_phi]

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for red noise Fourier terms
                        gradient[key] = 0.5 * np.sum(psr.diagSBS[key] /
                                psr.sr_sigma**2)

                    # GW signals
                    for key, value in self.d_Svec_d_param.iteritems():
                        # We need to remember diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_F_only] = \
                                psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                                value[fslc_phi]

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for red noise Fourier terms
                        gradient[key] = 0.5 * np.sum(psr.diagSBS[key] /
                                psr.sr_sigma**2)

                # DM variations
                if psr.dmfourierind is not None:
                    fslc_theta = slice(np.sum(self.npfdm[:ii]),
                            np.sum(self.npfdm[:ii+1]))

                    for key, value in self.d_Thetavec_d_param.iteritems():
                        # We need to remember diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_D_only] = \
                                psr.sr_Beta_inv[psr.Zmask_D_only]**2 * \
                                value[fslc_theta]

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for DM variation Fourier terms
                        gradient[key] = 0.5 * np.sum(psr.diagSBS[key] /
                                psr.sr_sigma**2)

                # ECORR
                if psr.jitterind is not None:
                    #fslc_J = slice(np.sum(self.npu[:ii]),
                    #        np.sum(self.npu[:ii+1]))

                    for key, value in psr.d_Jvec_d_param.iteritems():
                        # We need to remember diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_U_only] = \
                                psr.sr_Beta_inv[psr.Zmask_U_only]**2 * \
                                value

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for DM variation Fourier terms
                        gradient[key] = 0.5 * np.sum(psr.diagSBS[key] /
                                psr.sr_sigma**2)

        self._mu = mu                   # Mean of stingray transform
        self._sigma = sigma             # Slope of stingray transform
        self._log_jacob = log_jacob     # Log-jacobian of transform
        self._gradient = gradient       # Gradient of log-jacobian
        self._d_b_d_xi = d_b_d_xi       # d_x_d_p
        self._d_b_d_B = d_b_d_B         # d_x_d_B, with B hyper-pars


    def stingray_hessian_quants(self, p, set_hyper_pars=True):
        """Calculate quantities necessary for the Hessian calculations"""

        raise NotImplementedError("Hessian not implemented for this class (yet)!")

        # p should not be more than one-dimensional
        assert len(p.shape) == 1

        # Obtain the gradient of the original posterior.
        # No chaching, as we are not optimizing for speed
        x = self.backward(p)
        lp, lp_grad = self.likob.logposterior_grad(x)

        hessian = np.zeros((len(p), len(p)))

        for ii, psr in enumerate(self.ptapsrs):

            if psr.fourierind is not None:
                # Have a red noise stingray transformation
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc = slice(findex, findex+nfs)
                phivec = self.Phivec[fslc] + self.Svec[:nfs]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_F_only] + 1.0 / phivec)
                v = psr.sr_ZNyvec[psr.Zmask_F_only]
                std = np.sqrt(Sigmavec)
                index = psr.fourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # Pre-compute coefficients for the non-tensor components
                p_coeff1 = 0.75 * Sigmavec**2.5 / phivec**4
                p_coeff2 = -1.0 * Sigmavec**1.5 / phivec**3
                p_coeff = p[slc] * (p_coeff1 + p_coeff2)
                v_coeff1 = 2.0 * Sigmavec**3 / phivec**4
                v_coeff2 = -2.0 * Sigmavec**2 / phivec**3
                v_coeff = v * (v_coeff1 + v_coeff2)
                snt_coeff = p_coeff + v_coeff
                fnt_coeff = 0.5 * Sigmavec**1.5 / phivec**2
                d2_sigma_d2_phi = 0.5 * Sigmavec**2 / phivec**4
                d2_phi2_d2_phi = -Sigmavec / phivec**3

                # INEFFICIENT: slicing [flc] all the time, below

                # Only red-noise -- red-noise crosses
                for key1, d_Phivec_d_p1 in self.d_Phivec_d_param.iteritems():
                    # First derivatives of non-tensor component
                    hessian[slc, key1] += fnt_coeff * d_Phivec_d_p1[fslc] * lp_grad[slc]
                    hessian[key1, slc] += fnt_coeff * d_Phivec_d_p1[fslc] * lp_grad[slc]

                    # Don't symmetrize, because we loop over both keys
                    for key2, d_Phivec_d_p2 in self.d_Phivec_d_param.iteritems():
                        # First term (log-jacobian): d_sigma_d_phi
                        hessian[key1, key2] += np.sum(d2_sigma_d2_phi *
                                d_Phivec_d_p1[fslc] * d_Phivec_d_p2[fslc])

                        # Second term (log-jacobian): d_phi2_d_phi
                        hessian[key1, key2] += np.sum(d2_phi2_d2_phi *
                                d_Phivec_d_p1[fslc] * d_Phivec_d_p2[fslc])

                        # Non-tensor component
                        hessian[key1, key2] += np.sum(snt_coeff * d_Phivec_d_p1[fslc] *
                                d_Phivec_d_p2[fslc] * lp_grad[slc])

                # Pre-compute coefficients for the non-tensor components
                p_coeff = 0.5 * p[slc] * Sigmavec**1.5 / phivec**2
                v_coeff = 1.0 * v * Sigmavec**2 / phivec**2
                snt_coeff = p_coeff + v_coeff
                d2_phi_d2_phi = 0.5 * Sigmavec / phivec**2

                # Only red-noise -- second derivatives
                for key, d2_Phivec_d2_p in self.d2_Phivec_d2_param.iteritems():

                    if key[0] == key[1]:
                        # Log-jacobian
                        hessian[key[0], key[1]] += np.sum(d2_phi_d2_phi *
                                d2_Phivec_d2_p[fslc])

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Phivec_d2_p[fslc] * lp_grad[slc])
                    else:
                        # Log-jacobian
                        hessian[key[0], key[1]] += np.sum(d2_phi_d2_phi *
                                d2_Phivec_d2_p[fslc])
                        hessian[key[1], key[0]] += np.sum(d2_phi_d2_phi *
                                d2_Phivec_d2_p[fslc])

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Phivec_d2_p[fslc] * lp_grad[slc])
                        hessian[key[1], key[0]] += np.sum(snt_coeff *
                                d2_Phivec_d2_p[fslc] * lp_grad[slc])

                # INEFFICIENT: Merge these with the memorized ones above? (Not
                #              very inefficient though)
                # INEFFICIENT: slicing [flc] all the time, below
                # Pre-compute coefficients for the non-tensor components
                fnt_coeff = 0.5 * Sigmavec**1.5 / phivec**2
                d2_sigma_d2_phi = 0.5 * Sigmavec**2 / phivec**4
                d2_phi2_d2_phi = -Sigmavec / phivec**3
                p_coeff1 = 0.75 * Sigmavec**2.5 / phivec**4
                p_coeff2 = -1.0 * Sigmavec**1.5 / phivec**3
                p_coeff = p[slc] * (p_coeff1 + p_coeff2)
                v_coeff1 = 2.0 * Sigmavec**3 / phivec**4
                v_coeff2 = -2.0 * Sigmavec**2 / phivec**3
                v_coeff = v * (v_coeff1 + v_coeff2)
                snt_coeff = p_coeff + v_coeff

                # GW and red-noise crosses
                for key1, d_Svec_d_p1 in self.d_Svec_d_param.iteritems():
                    # First derivatives of non-tensor component
                    hessian[slc, key1] += fnt_coeff * d_Svec_d_p1[fslc] * lp_grad[slc]
                    hessian[key1, slc] += fnt_coeff * d_Svec_d_p1[fslc] * lp_grad[slc]

                    # Do symmetrize, because key1 != key2
                    for key2, d_Phivec_d_p2 in self.d_Phivec_d_param.iteritems():
                        # First term (log-jacobian): d_sigma_d_phi
                        hessian[key1, key2] += np.sum(d2_sigma_d2_phi *
                                d_Svec_d_p1[fslc] * d_Phivec_d_p2[fslc])
                        hessian[key2, key1] += np.sum(d2_sigma_d2_phi *
                                d_Svec_d_p1[fslc] * d_Phivec_d_p2[fslc])

                        # Second term (log-jacobian): d_phi2_d_phi
                        hessian[key1, key2] += np.sum(d2_phi2_d2_phi *
                                d_Svec_d_p1[fslc] * d_Phivec_d_p2[fslc])
                        hessian[key2, key1] += np.sum(d2_phi2_d2_phi *
                                d_Svec_d_p1[fslc] * d_Phivec_d_p2[fslc])

                        # Non-tensor component
                        hessian[key1, key2] += np.sum(snt_coeff * d_Svec_d_p1[fslc] *
                                d_Phivec_d_p2[fslc] * lp_grad[slc])
                        hessian[key2, key1] += np.sum(snt_coeff * d_Svec_d_p1[fslc] *
                                d_Phivec_d_p2[fslc] * lp_grad[slc])

                    # Don't symmetrize, because we loop over both keys
                    for key2, d_Svec_d_p2 in self.d_Svec_d_param.iteritems():
                        # First term (log-jacobian): d_sigma_d_phi
                        hessian[key1, key2] += np.sum(d2_sigma_d2_phi *
                                d_Svec_d_p1[fslc] * d_Svec_d_p2[fslc])

                        # Second term (log-jacobian): d_phi2_d_phi
                        hessian[key1, key2] += np.sum(d2_phi2_d2_phi *
                                d_Svec_d_p1[fslc] * d_Svec_d_p2[fslc])

                        # Non-tensor component
                        hessian[key1, key2] += np.sum(snt_coeff * d_Svec_d_p1[fslc] *
                                d_Svec_d_p2[fslc] * lp_grad[slc])

                # Pre-compute coefficients for the non-tensor components
                p_coeff = 0.5 * p[slc] * Sigmavec**1.5 / phivec**2
                v_coeff = 1.0 * v * Sigmavec**2 / phivec**2
                snt_coeff = p_coeff + v_coeff
                d2_phi_d2_phi = 0.5 * Sigmavec / phivec**2

                # Only GWs -- second derivatives
                for key, d2_Svec_d2_p in self.d2_Svec_d2_param.iteritems():

                    if key[0] == key[1]:
                        hessian[key[0], key[1]] += np.sum(d2_phi_d2_phi *
                                d2_Svec_d2_p[fslc])

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Svec_d2_p[fslc] * lp_grad[slc])
                    else:
                        hessian[key[0], key[1]] += np.sum(d2_phi_d2_phi *
                                d2_Svec_d2_p[fslc])
                        hessian[key[1], key[0]] += np.sum(d2_phi_d2_phi *
                                d2_Svec_d2_p[fslc])

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Svec_d2_p[fslc] * lp_grad[slc])
                        hessian[key[1], key[0]] += np.sum(snt_coeff *
                                d2_Svec_d2_p[fslc] * lp_grad[slc])

            if psr.dmfourierind is not None:
                # Have a dm noise stingray transformation
                fdmindex = np.sum(self.npfdm[:ii])
                nfdms = self.npfdm[ii]
                thetavec = self.Thetavec[fdmindex:fdmindex+nfdms]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_D_only] + 1.0 / thetavec)
                v = psr.sr_ZNyvec[psr.Zmask_D_only]
                std = np.sqrt(Sigmavec)
                index = psr.dmfourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # Pre-compute coefficients for the non-tensor components
                fnt_coeff = 0.5 * Sigmavec**1.5 / thetavec**2
                d2_sigma_d2_theta = 0.5 * Sigmavec**2 / thetavec**4
                d2_theta2_d2_theta = -Sigmavec / thetavec**3
                p_coeff1 = 0.75 * Sigmavec**2.5 / thetavec**4
                p_coeff2 = -1.0 * Sigmavec**1.5 / thetavec**3
                p_coeff = p[slc] * (p_coeff1 + p_coeff2)
                v_coeff1 = 2.0 * Sigmavec**3 / thetavec**4
                v_coeff2 = -2.0 * Sigmavec**2 / thetavec**3
                v_coeff = v * (v_coeff1 + v_coeff2)
                snt_coeff = p_coeff + v_coeff

                # First derivatives
                for key1, d_Thetavec_d_p1 in self.d_Thetavec_d_param.iteritems():
                    # First derivatives of non-tensor component
                    hessian[slc, key1] += fnt_coeff * d_Thetavec_d_p1[fslc] * lp_grad[slc]
                    hessian[key1, slc] += fnt_coeff * d_Thetavec_d_p1[fslc] * lp_grad[slc]

                    # Don't symmetrize, because we loop over both keys
                    for key2, d_Thetavec_d_p2 in self.d_Thetavec_d_param.iteritems():
                        # First term (log-jacobian): d_sigma_d_theta
                        hessian[key1, key2] += np.sum(d2_sigma_d2_theta *
                                d_Thetavec_d_p1 * d_Thetavec_d_p2)

                        # Second term (log-jacobian): d_theta2_d_theta
                        hessian[key1, key2] += np.sum(d2_theta2_d2_theta *
                                d_Thetavec_d_p1 * d_Thetavec_d_p2)

                        # Non-tensor component
                        hessian[key1, key2] += np.sum(snt_coeff * d_Thetavec_d_p1[fslc] *
                                d_Thetavec_d_p2[fslc] * lp_grad[slc])

                # Pre-compute coefficients for the non-tensor components
                p_coeff = 0.5 * p[slc] * Sigmavec**1.5 / thetavec**2
                v_coeff = 1.0 * v * Sigmavec**2 / thetavec**2
                snt_coeff = p_coeff + v_coeff
                d2_theta_d2_theta = 0.5 * Sigmavec / thetavec**2

                # Second derivatives
                for key, d2_Thetavec_d2_p in self.d2_Thetavec_d2_param.iteritems():
                    if key[0] == key[1]:
                        # Log-jacobian
                        hessian[key[0], key[1]] += np.sum(d2_theta_d2_theta *
                                d2_Thetavec_d2_p)

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Thetavec_d2_p[fslc] * lp_grad[slc])
                    else:
                        # Log-jacobian
                        hessian[key[0], key[1]] += np.sum(d2_theta_d2_theta *
                                d2_Thetavec_d2_p)
                        hessian[key[1], key[0]] += np.sum(d2_theta_d2_theta *
                                d2_Thetavec_d2_p)

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Thetavec_d2_p[fslc] * lp_grad[slc])
                        hessian[key[1], key[0]] += np.sum(snt_coeff *
                                d2_Thetavec_d2_p[fslc] * lp_grad[slc])

            if psr.jitterind is not None:
                # Have an ecor stingray transformation
                uindex = np.sum(self.npu[:ii])
                nus = self.npu[ii]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_U_only] + \
                        1.0 / psr.Jvec)
                v = psr.sr_ZNyvec[psr.Zmask_U_only]
                std = np.sqrt(Sigmavec)        # No hyper pars
                index = psr.jitterind
                npars = len(std)
                slc = slice(index, index+npars)

                # Pre-compute coefficients for the non-tensor components
                fnt_coeff = 0.5 * Sigmavec**1.5 / psr.Jvec**2
                d2_sigma_d2_j = 0.5 * Sigmavec**2 / psr.Jvec**4
                d2_j2_d2_j = -Sigmavec / psr.Jvec**3
                p_coeff1 = 0.75 * Sigmavec**2.5 / psr.Jvec**4
                p_coeff2 = -1.0 * Sigmavec**1.5 / psr.Jvec**3
                p_coeff = p[slc] * (p_coeff1 + p_coeff2)
                v_coeff1 = 2.0 * Sigmavec**3 / psr.Jvec**4
                v_coeff2 = -2.0 * Sigmavec**2 / psr.Jvec**3
                v_coeff = v * (v_coeff1 + v_coeff2)
                snt_coeff = p_coeff + v_coeff

                # First derivatives
                for key1, d_Jvec_d_p1 in psr.d_Jvec_d_param.iteritems():
                    # First derivatives of non-tensor component
                    hessian[slc, key1] += fnt_coeff * d_Jvec_d_p1 * lp_grad[slc]
                    hessian[key1, slc] += fnt_coeff * d_Jvec_d_p1 * lp_grad[slc]

                    # Don't symmetrize, because we loop over both keys
                    for key2, d_Jvec_d_p2 in psr.d_Jvec_d_param.iteritems():
                        # First term (log-jacobian): d_sigma_d_j
                        hessian[key1, key2] += np.sum(d2_sigma_d2_j *
                                d_Jvec_d_p1 * d_Jvec_d_p2)

                        # Second term (log-jacobian): d_j2_d_j
                        hessian[key1, key2] += np.sum(d2_j2_d2_j *
                                d_Jvec_d_p1 * d_Jvec_d_p2)

                        # Non-tensor component
                        hessian[key1, key2] += np.sum(snt_coeff * d_Jvec_d_p1 *
                                d_Jvec_d_p2 * lp_grad[slc])

                # Pre-compute coefficients for the non-tensor components
                p_coeff = 0.5 * p[slc] * Sigmavec**1.5 / psr.Jvec**2
                v_coeff = 1.0 * v * Sigmavec**2 / psr.Jvec**2
                snt_coeff = p_coeff + v_coeff
                d2_j_d2_j = 0.5 * Sigmavec / psr.Jvec**2

                # Second derivatives
                for key, d2_Jvec_d2_p in psr.d2_Jvec_d2_param.iteritems():
                    if key[0] == key[1]:
                        # Log-jacobian
                        hessian[key[0], key[1]] += np.sum(d2_j_d2_j *
                                d2_Jvec_d2_p)

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Jvec_d2_p * lp_grad[slc])
                    else:
                        hessian[key[0], key[1]] += np.sum(d2_j_d2_j *
                                d2_Jvec_d2_p)
                        hessian[key[1], key[0]] += np.sum(d2_j_d2_j *
                                d2_Jvec_d2_p)

                        # Non-tensor
                        hessian[key[0], key[1]] += np.sum(snt_coeff *
                                d2_Jvec_d2_p * lp_grad[slc])
                        hessian[key[1], key[0]] += np.sum(snt_coeff *
                                d2_Jvec_d2_p * lp_grad[slc])

        return hessian

    def dxdp_nondiag(self, p, ll_grad):
        """Non-diagonal derivative of x wrt p (jacobian for chain-rule)

        Dealt with (optionally) separately, for efficiency, since this would
        otherwise be an O(p^2) operation

        ll_grad is a vector, or a 2D array, with the columns of equal
        dimensionality as p.
        """
        # p should not be more than one-dimensional
        assert len(p.shape) == 1

        ll_grad2 = np.atleast_2d(ll_grad)
        extra_grad = np.zeros_like(ll_grad2)
        (a, b) = extra_grad.shape

        for ii, psr in enumerate(self.ptapsrs):
            pslc_tot = self.get_par_psr_sigma_inds(ii, psr)
            ll_grad2_psr = ll_grad2[:, pslc_tot]
            pars_psr = p[pslc_tot]
            Wv = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)

            if psr.fourierind is not None:
                fslc_phi = slice(np.sum(self.npf[:ii]), np.sum(self.npf[:ii+1]))
                slc_sig = psr.Zmask_F_only

                for key, d_Phivec_d_p in self.d_Phivec_d_param.iteritems():
                    # dxdp for Sigma
                    extra_grad[:, key] += 0.5 * np.sum(ll_grad2_psr *
                            pars_psr[None, :] * psr.diagSBS[key][None, :] / 
                            psr.sr_sigma[None, :], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Phivec_d_p[fslc_phi] * Wv[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)


                for key, d_Svec_d_p in self.d_Svec_d_param.iteritems():
                    # dxdp for Sigma
                    extra_grad[:, key] += 0.5 * np.sum(ll_grad2_psr *
                            pars_psr[None, :] * psr.diagSBS[key][None, :] / 
                            psr.sr_sigma[None, :], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Svec_d_p[fslc_phi] * Wv[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)

            if psr.dmfourierind is not None:
                fslc_theta = slice(np.sum(self.npfdm[:ii]), np.sum(self.npfdm[:ii+1]))
                slc_sig = psr.Zmask_D_only

                for key, d_Thetavec_d_p in self.d_Thetavec_d_param.iteritems():
                    # dxdp for Sigma
                    extra_grad[:, key] += 0.5 * np.sum(ll_grad2_psr *
                            pars_psr[None, :] * psr.diagSBS[key][None, :] / 
                            psr.sr_sigma[None, :], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Thetavec_d_p[fslc_theta] * Wv[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)

            if psr.jitterind is not None:
                slc_sig = psr.Zmask_U_only

                for key, d_Jvec_d_p in psr.d_Jvec_d_param.iteritems():
                    # dxdp for Sigma
                    extra_grad[:, key] += 0.5 * np.sum(ll_grad2_psr *
                            pars_psr[None, :] * psr.diagSBS[key][None, :] / 
                            psr.sr_sigma[None, :], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Jvec_d_p * Wv[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)


        return extra_grad.reshape(ll_grad.shape)

