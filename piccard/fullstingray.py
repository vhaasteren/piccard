#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss

from transformations import *
from choleskyext import *
from choleskyext_omp import *

# This file contains the class for the full stingray transformation. We'll use
# the Cholesky derivative algorithm from the Cholesky extension

def get_tril(X):
    """Helper function to get lower-triangular P, with P+P.T=C, C Symmetric """
    Y = np.tril(X, k=-1)
    di = np.diag_indices(len(X))
    Y[di] = 0.5 * np.diag(X)
    return Y


class fullStingrayLikelihood(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent

    NOTE: INCORRECT. The L matrix needs to be transposed for this to be correct.
          Included here for reference.
    """

    # TODO: The linear transformation is now a matrix equation.
    #       What we need to redefine:
    #           - self.forward  (see below)
    #           - self.backward (see below)
    #           - self._sigma (no longer used)
    #           - self._d_b_d_xi (now 2D: psr.sr_Li.T)
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
    #    np.dot(psr.sr_Li, ll_grad2)
    #
    # 6) The log-jacobian is np.sum(np.log(np.diag(psr.sr_Li)))
    #
    # 7) Implement all the hyper-parameter derivatives using the Cython code.
    #    This includes the gradient of the Jacobian
    #    a) d_mu_d_hp  (fancy slicing)
    #    b) d_lj_d_hp  (Cython code - tj)
    #    c) d_L_d_hp   (Cython code - M)
    #
    #
    # Most of this is done, except for number (5)!!!!!!!!!!!!!!!!!!!!
    # First, however, make it work with the transposition...

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

            for pp, psr in enumerate(self.ptapsrs):
                # We have to apply the transformation per-pulsar
                p[0,psr.sr_pslc] = np.dot(psr.sr_L.T, p[0,psr.sr_pslc] - psr.sr_mu)

            self.uncache(self.forward)
        else:
            if self.have_cache():
                raise RuntimeError("Invalid transform from caching function")

            for ii, pc in enumerate(p):
                self.cache(pc, self.forward, 
                        direction='forward', calc_gradient=False)

                for pp, psr in enumerate(self.ptapsrs):
                    # We have to apply the transformation per-pulsar
                    p[ii,psr.sr_pslc] = np.dot(psr.sr_L.T, p[ii,psr.sr_pslc] - psr.sr_mu)
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
            for pp, psr in enumerate(self.ptapsrs):
                # We have to apply the transformation per-pulsar
                x[0,psr.sr_pslc] = np.dot(psr.sr_Li.T, x[0,psr.sr_pslc]) + psr.sr_mu
            self.uncache(self.backward)
        else:
            if self.have_cache():
                raise RuntimeError("Invalid transform from caching function")

            for ii, xc in enumerate(x):
                self.cache(xc, self.backward, 
                        direction='backward', calc_gradient=False)

                for pp, psr in enumerate(self.ptapsrs):
                    # We have to apply the transformation per-pulsar
                    x[ii,psr.sr_pslc] = np.dot(psr.sr_Li.T, x[ii,psr.sr_pslc]) + psr.sr_mu
                self.uncache(self.backward)

        return x.reshape(p.shape)

    def dxdp(self, p):
        """We do all the dxdp action in dxdp_nondiag"""
        self.cache(p, self.dxdp, direction='backward',
                calc_gradient=False)
        self.uncache(self.dxdp)

        return np.zeros_like(p)

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

        # NOTE: Sigma = L_inv^T L_inv    (just easier that way)
        L = sl.cholesky(Sigma_inv, lower=True)
        Li = sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True)
        cf = (L, True)

        # Turns out that matrix multiplication is faster than cho_solve
        # However, cho_solve is numerically more stable
        return sl.cho_solve(cf, np.eye(len(Sigma_inv))), L, Li
        #return np.dot(Li, Li.T), L, Li

    def get_par_psr_sigma_inds(self, ii, psr):
        """Given a pulsar, get a slice object (numpy int array) that contains
        _all_ the indices of the low-level parameters"""
        # TODO: This should be functionality of the Pulsar class
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

        for ii, psr in enumerate(self.ptapsrs):
            psr.sr_diagSBS = dict()     # For d_mu_d_par
            psr.sr_pslc = self.get_par_psr_sigma_inds(ii, psr)

            # Define the Stingray transformation
            psr.sr_Beta_inv = self.get_psr_Beta(ii, psr)
            psr.sr_Sigma, psr.sr_L, psr.sr_Li = \
                    self.get_psr_Sigma(ii, psr, psr.sr_Beta_inv)
            psr.sr_mu = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)

            # Undo the Stingray
            #psr.sr_mu *= 0.0
            #psr.sr_Sigma = np.eye(len(psr.sr_Sigma))
            #psr.sr_L = psr.sr_Li = np.eye(len(psr.sr_Sigma))

            # Quantities we need to take derivatives of the Cholesky factor
            lowlevel_pars = np.dot(psr.sr_Li.T, p[psr.sr_pslc])
            psr.sr_dL_M, psr.sr_dL_tj = \
                    cython_dL_update_omp(psr.sr_L, psr.sr_Li, lowlevel_pars)
            #        cython_dL_update(psr.sr_L, psr.sr_Li, lowlevel_pars)

            # The log(det(Jacobian)), for all low-level parameters
            log_jacob += np.sum(np.log(np.diag(psr.sr_Li)))

        if calc_gradient:

            for ii, psr in enumerate(self.ptapsrs):

                # Red noise
                if psr.fourierind is not None:
                    fslc_phi = slice(np.sum(self.npf[:ii]),
                            np.sum(self.npf[:ii+1]))

                    for key, value in psr.d_Phivec_d_param.iteritems():
                        # We need to remember sr_diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_F_only] = \
                                psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                                value[fslc_phi]

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.sr_diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for red noise Fourier terms
                        gradient[key] += np.sum(
                                psr.sr_dL_tj[psr.Zmask_F_only] *
                                BdB[psr.Zmask_F_only])

                    # GW signals
                    for key, value in self.d_Svec_d_param.iteritems():
                        # We need to remember sr_diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_F_only] = \
                                psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                                value[fslc_phi]

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.sr_diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for red noise Fourier terms
                        gradient[key] += np.sum(
                                psr.sr_dL_tj[psr.Zmask_F_only] *
                                BdB[psr.Zmask_F_only])

                # DM variations
                if psr.dmfourierind is not None:
                    fslc_theta = slice(np.sum(self.npfdm[:ii]),
                            np.sum(self.npfdm[:ii+1]))

                    for key, value in psr.d_Thetavec_d_param.iteritems():
                        # We need to remember sr_diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_D_only] = \
                                psr.sr_Beta_inv[psr.Zmask_D_only]**2 * \
                                value[fslc_theta]

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.sr_diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for DM variation Fourier terms
                        gradient[key] += np.sum(
                                psr.sr_dL_tj[psr.Zmask_D_only] *
                                BdB[psr.Zmask_D_only])

                # ECORR
                if psr.jitterind is not None:
                    #fslc_J = slice(np.sum(self.npu[:ii]),
                    #        np.sum(self.npu[:ii+1]))

                    for key, value in psr.d_Jvec_d_param.iteritems():
                        # We need to remember sr_diagSBS for dxdp_nondiag
                        BdB = np.zeros(len(psr.sr_Sigma))
                        BdB[psr.Zmask_U_only] = \
                                psr.sr_Beta_inv[psr.Zmask_U_only]**2 * \
                                value

                        # Do some slicing magic to get the diagonals of the
                        # matrix product in O(n^2) time
                        BS = psr.sr_Sigma * BdB[None, :]
                        psr.sr_diagSBS[key] = np.sum(psr.sr_Sigma * BS, axis=1)

                        # Log-jacobian for DM variation Fourier terms
                        gradient[key] += np.sum(
                                psr.sr_dL_tj[psr.Zmask_U_only] *
                                BdB[psr.Zmask_U_only])

        self._log_jacob = log_jacob     # Log-jacobian of transform
        self._gradient = gradient       # Gradient of log-jacobian


    def stingray_hessian_quants(self, p, set_hyper_pars=True):
        """Calculate quantities necessary for the Hessian calculations"""

        # p should not be more than one-dimensional
        assert len(p.shape) == 1

        # Obtain the gradient of the original posterior.
        # No chaching, as we are not optimizing for speed
        # We need this for the components that do not transform as a tensor
        x = self.backward(p)
        lp, lp_grad = self.likob.logposterior_grad(x)

        hessian = np.zeros((len(p), len(p)))

        for ii, psr in enumerate(self.ptapsrs):
            psr.sr_pslc = self.get_par_psr_sigma_inds(ii, psr)

            # Define the Stingray transformation
            psr.sr_Beta_inv = self.get_psr_Beta(ii, psr)
            psr.sr_Sigma, psr.sr_L, psr.sr_Li = \
                    self.get_psr_Sigma(ii, psr, psr.sr_Beta_inv)
            psr.sr_mu = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)

            # Red noise
            if psr.fourierind is not None:
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc_phi = slice(findex, findex+nfs)
                #v = psr.sr_ZNyvec[psr.Zmask_F_only]

                # Loop over first derivatives
                for key1, d_Phivec_d_p1 in psr.d_Phivec_d_param.iteritems():
                    BdB = np.zeros(len(psr.sr_Sigma))
                    BdB[psr.Zmask_F_only] = \
                            psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                            d_Phivec_d_p1[fslc_phi]
                    d_W_d_ea = -np.copy(BdB)                            # OK

                    # Derivatives wrt eta_a (ea)
                    d_Sigma_d_ea = -np.dot(psr.sr_Sigma * d_W_d_ea, psr.sr_Sigma)
                    LWaL = np.dot(psr.sr_Li * d_W_d_ea, psr.sr_Li.T)    # OK
                    PhiLa = get_tril(LWaL)                              # OK
                    d_L_d_ea = np.dot(psr.sr_L, PhiLa)                  # OK

                    # Combination of dL that we'll need more later
                    LdLTa = np.dot(d_L_d_ea.T, psr.sr_Li.T)             # OK
                    d_LmT_d_ea = -np.dot(psr.sr_Li.T, LdLTa)            # OK

                    #######################################################
                    # Some actual Hessian elements. Non-tensor first order
                    non_tensor =  np.dot(d_LmT_d_ea.T, lp_grad[psr.sr_pslc])
                    hessian[psr.sr_pslc, key1] += non_tensor            # OK
                    hessian[key1, psr.sr_pslc] += non_tensor            # OK

                    # Don't symmetrize, because we loop over both keys
                    for key2, d_Phivec_d_p2 in psr.d_Phivec_d_param.iteritems():
                        # Find the key for the second derivative
                        if (key1, key2) in psr.d2_Phivec_d2_param:
                            dkey = (key1, key2)
                        elif (key2, key1) in psr.d2_Phivec_d2_param:
                            dkey = (key2, key1)
                        else:
                            raise ValueError("key not in d2_Phivec_d2_param!")
                        d2_Phivec_d_p1p2 = psr.d2_Phivec_d2_param[dkey]

                        # The derivatives of W (=Sigma^{-1})
                        d_W_d_eb = np.zeros(len(psr.sr_Sigma))
                        d2_W_d_eab = np.zeros(len(psr.sr_Sigma))
                        d_W_d_eb[psr.Zmask_F_only] = \
                                -psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                                d_Phivec_d_p2[fslc_phi]
                        d2_W_d_eab[psr.Zmask_F_only] = \
                                -psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                                d2_Phivec_d_p1p2[fslc_phi] + \
                                2*psr.sr_Beta_inv[psr.Zmask_F_only]**3 * \
                                d_Phivec_d_p1[fslc_phi] * \
                                d_Phivec_d_p2[fslc_phi]

                        # Derivatives wrt eta_b (eb)
                        d_Sigma_d_eb = -np.dot(psr.sr_Sigma * d_W_d_eb, psr.sr_Sigma)
                        LWbL = np.dot(psr.sr_Li * d_W_d_eb, psr.sr_Li.T)
                        PhiLb = get_tril(LWbL)                          # OK
                        d_L_d_eb = np.dot(psr.sr_L, PhiLb)              # OK

                        # Get the Phi's for the cross-dL second derivatives
                        Phi_first_ba = get_tril(
                                np.dot(psr.sr_Li, np.dot(d_L_d_eb, LWaL)))
                        Phi_first_ab = get_tril(
                                np.dot(LWaL, np.dot(d_L_d_eb.T, psr.sr_Li.T)))
                        Phi_second_ab = get_tril(
                                np.dot(psr.sr_Li * d2_W_d_eab, psr.sr_Li.T))

                        # Second derivatives of the Cholesky factor
                        d2_L_deadeb = np.dot(d_L_d_eb, PhiLa) - \
                                np.dot(psr.sr_L, Phi_first_ba) - \
                                np.dot(psr.sr_L, Phi_first_ab) + \
                                np.dot(psr.sr_L, Phi_second_ab)             # OK

                        # For second derivatives of b, we need the
                        # p_coefficients
                        p_vals = p[psr.sr_pslc]
                        p_coeff_first_ab = np.dot(psr.sr_Li.T,
                                np.dot(d_L_d_ea.T, np.dot(psr.sr_Li.T,
                                        np.dot(d_L_d_eb.T, np.dot(psr.sr_Li.T,
                                                p_vals)))))
                        p_coeff_first_ba = np.dot(psr.sr_Li.T,
                                np.dot(d_L_d_eb.T, np.dot(psr.sr_Li.T,
                                        np.dot(d_L_d_ea.T, np.dot(psr.sr_Li.T,
                                                p_vals)))))
                        p_coeff_second_ab = np.dot(psr.sr_Li.T,
                                np.dot(d2_L_deadeb.T, np.dot(psr.sr_Li.T,
                                        p_vals)))

                        # Second derivatives of mu
                        Sigmav = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
                        dSv = np.dot(d_Sigma_d_ea, psr.sr_ZNyvec)
                        d2_mu_d2_eab = np.dot(d_Sigma_d_ea * d_W_d_eb, Sigmav) \
                                - np.dot(psr.sr_Sigma * d2_W_d_eab, Sigmav) \
                                + np.dot(psr.sr_Sigma * d_W_d_eb, dSv)

                        # Second derivatives of b
                        d2_b_d_eab = d2_mu_d2_eab - p_coeff_second_ab + \
                                p_coeff_first_ab + p_coeff_first_ba
                        
                        # Non-tensor Hessian components
                        hessian[key1, key2] += np.sum(d2_b_d_eab *
                                lp_grad[psr.sr_pslc])

                        # Log-jacobian Hessian components                   # OK
                        LdLa = np.dot(psr.sr_Li, d_L_d_ea)
                        LdLb = np.dot(psr.sr_Li, d_L_d_eb)
                        LdLab = np.dot(psr.sr_Li, d2_L_deadeb)
                        hessian[key1, key2] += np.trace(np.dot(LdLa, LdLb)) - \
                                               np.trace(LdLab)
            
            # Do Theta and J components here....

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
        extra_grad[:, :] = np.copy(ll_grad2)

        for ii, psr in enumerate(self.ptapsrs):
            #pslc_tot = self.get_par_psr_sigma_inds(ii, psr)
            pslc_tot = psr.sr_pslc
            ll_grad2_psr = ll_grad2[:, pslc_tot]
            pars_psr = p[pslc_tot]

            # We have to do the 'regular' dxdp here as well, since in this full
            # Stingray transform, that is a full 2D matrix for the low-level
            # parameters
            extra_grad[:, pslc_tot] = np.dot(psr.sr_Li, ll_grad2_psr.T).T

            if psr.fourierind is not None:
                fslc_phi = slice(np.sum(self.npf[:ii]), np.sum(self.npf[:ii+1]))
                slc_sig = psr.Zmask_F_only

                for key, d_Phivec_d_p in psr.d_Phivec_d_param.iteritems():
                    BdB = np.zeros(len(psr.sr_Sigma))
                    BdB[psr.Zmask_F_only] = \
                            psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                            d_Phivec_d_p[fslc_phi]

                    # dxdp for Sigma
                    dxdhp = np.dot(psr.sr_Li.T, np.dot(psr.sr_dL_M[:,slc_sig],
                            BdB[psr.Zmask_F_only]))
                    extra_grad[:, key] += np.sum(
                            dxdhp[None,:] * ll_grad2_psr[:,:], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Phivec_d_p[fslc_phi] * psr.sr_mu[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)


                for key, d_Svec_d_p in self.d_Svec_d_param.iteritems():
                    BdB = np.zeros(len(psr.sr_Sigma))
                    BdB[psr.Zmask_F_only] = \
                            psr.sr_Beta_inv[psr.Zmask_F_only]**2 * \
                            d_Svec_d_p[fslc_phi]

                    # dxdp for Sigma
                    dxdhp = np.dot(psr.sr_Li.T, np.dot(psr.sr_dL_M[:,slc_sig],
                            BdB[psr.Zmask_F_only]))
                    extra_grad[:, key] += np.sum(
                            dxdhp[None,:] * ll_grad2_psr[:,:], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Svec_d_p[fslc_phi] * psr.sr_mu[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)

            if psr.dmfourierind is not None:
                fslc_theta = slice(np.sum(self.npfdm[:ii]), np.sum(self.npfdm[:ii+1]))
                slc_sig = psr.Zmask_D_only

                for key, d_Thetavec_d_p in psr.d_Thetavec_d_param.iteritems():
                    BdB = np.zeros(len(psr.sr_Sigma))
                    BdB[psr.Zmask_D_only] = \
                            psr.sr_Beta_inv[psr.Zmask_D_only]**2 * \
                            d_Thetavec_d_p[fslc_theta]
                    # dxdp for Sigma
                    dxdhp = np.dot(psr.sr_Li.T, np.dot(psr.sr_dL_M[:,slc_sig],
                            BdB[psr.Zmask_D_only]))
                    extra_grad[:, key] += np.sum(
                            dxdhp[None,:] * ll_grad2_psr[:,:], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Thetavec_d_p[fslc_theta] * psr.sr_mu[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)

            if psr.jitterind is not None:
                slc_sig = psr.Zmask_U_only

                for key, d_Jvec_d_p in psr.d_Jvec_d_param.iteritems():
                    BdB = np.zeros(len(psr.sr_Sigma))
                    BdB[psr.Zmask_U_only] = \
                            psr.sr_Beta_inv[psr.Zmask_U_only]**2 * \
                            d_Jvec_d_p
                    # dxdp for Sigma
                    dxdhp = np.dot(psr.sr_Li.T, np.dot(psr.sr_dL_M[:,slc_sig],
                            BdB[psr.Zmask_U_only]))
                    extra_grad[:, key] += np.sum(
                            dxdhp[None,:] * ll_grad2_psr[:,:], axis=1)

                    # dxdp for mu
                    WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                            psr.sr_Beta_inv[slc_sig]**2 *
                            d_Jvec_d_p * psr.sr_mu[slc_sig])
                    extra_grad[:, key] += np.sum(ll_grad2_psr * 
                            WBWv[None, :], axis=1)


        return extra_grad.reshape(ll_grad.shape)

