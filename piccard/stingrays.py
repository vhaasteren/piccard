#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
from functools import partial

from transformations import *


class hpStingrayLikelihood(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the hpStingrayLikelihood with a ptaLikelihood object"""
        super(hpStingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

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

        # For efficiency in calculating gradients, we'll have to store some
        # matrix-vector products
        d_lj_d_phi = np.zeros_like(self.Phivec)
        d_lj_d_theta = np.zeros_like(self.Thetavec)

        for ii, psr in enumerate(self.ptapsrs):
            if psr.timingmodelind is not None:
                # Have a timing model parameter stingray transformation
                Sigmavec = psr.sr_Sigma1[psr.Zmask_M_only]  # No hyper pars
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_M_only]
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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_F_only] + 1.0 / phivec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_F_only]
                index = psr.fourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_phi for down below
                d_lj_d_phi[fslc] = 0.5 * Sigmavec / phivec**2

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
                fslc = slice(fdmindex, fdmindex+nfdms)
                thetavec = self.Thetavec[fdmindex:fdmindex+nfdms]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_D_only] + 1.0 / thetavec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_D_only]
                index = psr.dmfourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_theta for down below
                d_lj_d_theta[fslc] = 0.5 * Sigmavec / thetavec**2

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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_U_only] + \
                        1.0 / psr.Jvec)
                std = np.sqrt(Sigmavec)        # No hyper pars
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_U_only]
                index = psr.jitterind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Do jitter gradients here, since they are kept track of on a
                # per-pulsar basis
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

        if calc_gradient:
            # Red noise
            for key, value in self.d_Phivec_d_param.iteritems():
                gradient[key] += np.sum(d_lj_d_phi * value)

            # GW signals
            for key, value in self.d_Svec_d_param.iteritems():
                gradient[key] += np.sum(d_lj_d_phi * value)

            # DM variations
            for key, value in self.d_Thetavec_d_param.iteritems():
                gradient[key] += np.sum(d_lj_d_theta * value)

        self._mu = mu                   # Mean of stingray transform
        self._sigma = sigma             # Slope of stingray transform
        self._log_jacob = log_jacob     # Log-jacobian of transform
        self._gradient = gradient       # Gradient of log-jacobian
        self._d_b_d_xi = d_b_d_xi       # d_x_d_p
        self._d_b_d_B = d_b_d_B         # d_x_d_B, with B hyper-pars (non-diag)

    def stingray_hessian_quants(self, p, set_hyper_pars=True):
        """Calculate quantities necessary for the Hessian calculations"""
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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_F_only] + 1.0 / phivec)
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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_D_only] + 1.0 / thetavec)
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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_U_only] + \
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

        # For computational efficiency, we'll have to memorize the
        # ll_grad * d_b_d_B vectors
        d_b_d_B_rn = np.zeros_like(self.Phivec)
        ll_grad2_rn = np.zeros( (ll_grad2.shape[0], len(self.Phivec)) )
        d_b_d_B_dm = np.zeros_like(self.Phivec)
        ll_grad2_dm = np.zeros( (ll_grad2.shape[0], len(self.Thetavec)) )

        for ii, psr in enumerate(self.ptapsrs):
            if psr.fourierind is not None:
                findex = np.sum(self.npf[:ii])
                ind = psr.fourierind
                nfreq = self.npf[ii]
                pslc = slice(ind, ind+nfreq)
                fslc = slice(findex, findex+nfreq)

                ll_grad2_rn[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_rn[fslc] = self._d_b_d_B[pslc]

            if psr.dmfourierind is not None:
                fdmindex = np.sum(self.npfdm[:ii])
                ind = psr.dmfourierind
                nfreqdm = self.npfdm[ii]
                pslc = slice(ind, ind+nfreqdm)
                fslc = slice(fdmindex, fdmindex+nfreqdm)

                ll_grad2_dm[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_dm[fslc] = self._d_b_d_B[pslc]

            if psr.jitterind is not None:
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                # Hyper parameters on Jvec
                for key, d_Jvec_d_p in psr.d_Jvec_d_param.iteritems():
                    for aa in range(a):
                        extra_grad[aa, key] += np.sum(ll_grad2[aa, pslc] * 
                                self._d_b_d_B[pslc] * d_Jvec_d_p)

        # Now that we have that stuff memorized, multiply that stuff
        for key, d_Phivec_d_p in self.d_Phivec_d_param.iteritems():
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Phivec_d_p, axis=1)

        for key, d_Svec_d_p in self.d_Svec_d_param.iteritems():
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Svec_d_p, axis=1)

        for key, d_Thetavec_d_p in self.d_Thetavec_d_param.iteritems():
            extra_grad[:, key] += np.sum(ll_grad2_dm * d_b_d_B_dm *
                    d_Thetavec_d_p, axis=1)

        return extra_grad.reshape(ll_grad.shape)



class tmStingrayLikelihood(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the tmStingrayLikelihood with a ptaLikelihood object"""
        super(tmStingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

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

        # For efficiency in calculating gradients, we'll have to store some
        # matrix-vector products
        d_lj_d_phi = np.zeros_like(self.Phivec)
        d_lj_d_theta = np.zeros_like(self.Thetavec)

        # For efficiency, calculate the following for the timing-model stingray
        # stuff
        ntmptot = np.sum(self.npm)
        d_Sigma_d_phi_tm = np.zeros((len(self.Phivec), ntmptot))
        d_Sigma_d_theta_tm = np.zeros((len(self.Thetavec), ntmptot))
        d_Sigma_d_J_tm = np.zeros((np.sum(self.npu), ntmptot))

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

            if psr.fourierind is not None:
                # Have a red noise stingray transformation
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc_phi = slice(findex, findex+nfs)
                phivec = self.Phivec[fslc_phi] + self.Svec[:nfs]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_F_only] + 1.0 / phivec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_F_only]
                index = psr.fourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_phi for down below
                d_lj_d_phi[fslc_phi] = 0.5 * Sigmavec / phivec**2

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / phivec**2
                d_mean_d_B = mean * Sigmavec / phivec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

                # Fill the quantities for the timing-model stingray
                gslc = psr.Zmask_F_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / phivec
                d_Sigma_d_phi_tm[fslc_phi, mslc] = (psr.sr_gamma[:,gslc] / \
                        (phivec**2*(psr.sr_A[gslc] + 1.0/phivec)**2)).T

            if psr.dmfourierind is not None:
                # Have a dm noise stingray transformation
                fdmindex = np.sum(self.npfdm[:ii])
                nfdms = self.npfdm[ii]
                fslc_theta = slice(fdmindex, fdmindex+nfdms)
                thetavec = self.Thetavec[fslc_theta]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_D_only] + 1.0 / thetavec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_D_only]
                index = psr.dmfourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_theta for down below
                d_lj_d_theta[fslc_theta] = 0.5 * Sigmavec / thetavec**2

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / thetavec**2
                d_mean_d_B = mean * Sigmavec / thetavec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

                # Fill the quantities for the timing-model stingray
                gslc = psr.Zmask_D_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / thetavec
                d_Sigma_d_theta_tm[fslc_theta, mslc] = -(psr.sr_gamma[:,gslc] / \
                            (thetavec**2*(psr.sr_A[gslc] + 1.0/thetavec)**2)).T

            if psr.jitterind is not None:
                # Have an ecor stingray transformation
                uindex = np.sum(self.npu[:ii])
                nus = self.npu[ii]
                fslc_J = slice(uindex, uindex+nus)

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_U_only] + \
                        1.0 / psr.Jvec)
                std = np.sqrt(Sigmavec)        # No hyper pars
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_U_only]
                index = psr.jitterind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                ######################################
                # Timing model Stingray calculations #
                ######################################
                gslc = psr.Zmask_U_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / psr.Jvec
                d_Sigma_d_J_tm[fslc_J, mslc] = -(psr.sr_gamma[:,gslc] / \
                            (psr.Jvec**2*(psr.sr_A[gslc] + 1.0/psr.Jvec)**2)).T

                # Do jitter gradients here, since they are kept track of on a
                # per-pulsar basis
                if calc_gradient:
                    for key, value in psr.d_Jvec_d_param.iteritems():
                        d_lj_d_J = 0.5 * Sigmavec / psr.Jvec**2
                        gradient[key] += np.sum(d_lj_d_J * value)

                        # Do the rest of the gradient stuff below...

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / psr.Jvec**2
                d_mean_d_B = mean * Sigmavec / psr.Jvec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

            if psr.timingmodelind is not None:
                # Now have all we need for the Stingray transformation
                Sigmavec = psr.sr_alpha / (psr.sr_beta - np.sum(
                        psr.sr_gamma[:,ntmpars:] /
                        sr_gamma_frac_den[ntmpars:], axis=1))

                Sigmavec_old = psr.sr_Sigma1[psr.Zmask_M_only]  # No hyper pars
                std = np.sqrt(Sigmavec)
                std_old = np.sqrt(Sigmavec_old)

                # We keep the old estimate of the mean, because is it less
                # biased
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_M_only]
                mean = Sigmavec_old * psr.sr_ZNyvec[psr.Zmask_M_only]
                index = psr.timingmodelind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                log_jacob += np.sum(np.log(std))

                d_b_d_xi[slc] = std

                # Multiply the d_Sigma_d_xxx_tm with a coefficient that depends
                # on all the hyper parameters
                d_Sigma_tm_coeff = psr.sr_alpha / (psr.sr_beta - np.sum(
                        psr.sr_gamma[:,ntmpars:] /
                        sr_gamma_frac_den[None, ntmpars:], axis=1))**2
                d_Sigma_d_phi_tm[fslc_phi,mslc] = (
                        d_Sigma_d_phi_tm[fslc_phi,mslc] *
                        d_Sigma_tm_coeff[None,:])
                d_Sigma_d_theta_tm[fslc_theta,mslc] = (
                        d_Sigma_d_theta_tm[fslc_theta,mslc] *
                        d_Sigma_tm_coeff)
                d_Sigma_d_J_tm[fslc_J,mslc] = (
                        d_Sigma_d_J_tm[fslc_J,mslc] *
                        d_Sigma_tm_coeff)

        ##############################################################
        ##############################################################
        ### ERROR BELOW: We need to loop over psr again ##############
        ##############################################################
        ##############################################################

        if calc_gradient:
            # For the timing model Stingray log-jacobian gradient, we need to
            # multiply the up-to-now recorded values with the full coefficient,
            # summed over the noise low-level parameters
            #d_Sigma_tm_coeff = psr.sr_alpha / (psr.sr_beta - np.sum(
            #        psr.sr_gamma / sr_gamma_frac_den, axis=1))**2

            # Red noise
            TMsigma = sigma[mslc_par]**2
            if psr.fourierind is not None:
                # TODO: Use proper broadcasting
                d_Jac_d_B = 0.5 * d_Sigma_d_phi_tm / TMsigma

                for key, value in self.d_Phivec_d_param.iteritems():
                    # Log-jacobian for red noise Fourier terms
                    gradient[key] += np.sum(d_lj_d_phi * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

                # GW signals
                for key, value in self.d_Svec_d_param.iteritems():
                    # Log-jacobian for red noise Fourier terms
                    gradient[key] += np.sum(d_lj_d_phi * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

            # DM variations
            if psr.dmfourierind is not None:
                # TODO: Use proper broadcasting like right above here
                #d_Jac_d_B = (d_Sigma_d_theta_tm.T * (0.5 / d_Sigma_d_theta_std**2) ).T
                d_Jac_d_B = 0.5 * d_Sigma_d_theta_tm / TMsigma
                for key, value in self.d_Thetavec_d_param.iteritems():
                    # Log-jacobian for DM variation Fourier terms
                    gradient[key] += np.sum(d_lj_d_theta * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

            # ECORR
            if psr.jitterind is not None:
                #d_Jac_d_B = (d_Sigma_d_J_tm.T * (0.5 / d_Sigma_d_J_std**2) ).T
                d_Jac_d_B = 0.5 * d_Sigma_d_J_tm / TMsigma
                for ii, psr in enumerate(self.ptapsrs):
                    fslc_J = slice(self.npu[:ii], self.npu[:ii+1])
                    for key, value in psr.d_Jvec_d_param.iteritems():
                        # The Log-jacobian gradient was already calculated above
                        # (TODO: change to here, now that we are here?)

                        # Log-jacobian for timing model (Just add the whole shebang)
                        gradient[key] += np.sum(np.sum(d_Jac_d_B[fslc_J,:].T * value, axis=0))

        self._mu = mu                   # Mean of stingray transform
        self._sigma = sigma             # Slope of stingray transform
        self._log_jacob = log_jacob     # Log-jacobian of transform
        self._gradient = gradient       # Gradient of log-jacobian
        self._d_b_d_xi = d_b_d_xi       # d_x_d_p
        self._d_b_d_B = d_b_d_B         # d_x_d_B, with B hyper-pars

        # For the timing-model Stingray, we'll have to save some more quantities
        self._d_Sigma_d_phi_tm = d_Sigma_d_phi_tm
        self._d_Sigma_d_theta_tm = d_Sigma_d_theta_tm
        self._d_Sigma_d_J_tm = d_Sigma_d_J_tm 

    def stingray_hessian_quants(self, p, set_hyper_pars=True):
        """Calculate quantities necessary for the Hessian calculations"""
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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_F_only] + 1.0 / phivec)
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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_D_only] + 1.0 / thetavec)
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

                Sigmavec = 1.0/(1.0/psr.sr_Sigma1[psr.Zmask_U_only] + \
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

        ntmptot = np.sum(self.npm)

        # For computational efficiency, we'll have to memorize the
        # ll_grad * d_b_d_B vectors
        d_b_d_B_rn = np.zeros_like(self.Phivec)
        ll_grad2_rn = np.zeros( (ll_grad2.shape[0], len(self.Phivec)) )
        d_b_d_B_dm = np.zeros_like(self.Phivec)
        ll_grad2_dm = np.zeros( (ll_grad2.shape[0], len(self.Thetavec)) )
        ll_grad2_tm = np.zeros( (ll_grad2.shape[0], np.sum(self.npm)) )

        # And these, for the timing model Stingray
        d_b_d_Sigma_rn = np.zeros((len(self.Phivec), ntmptot))
        d_b_d_Sigma_dm = np.zeros((len(self.Thetavec), ntmptot))
        d_b_d_Sigma_J = np.zeros((np.sum(self.npu), ntmptot))

        for ii, psr in enumerate(self.ptapsrs):
            mslc = slice(np.sum(self.npm[:ii]), np.sum(self.npm[:ii+1]))
            m_index = psr.timingmodelind        # Assuming this exists
            mslc_par = slice(m_index, m_index+self.npm[ii])
            if psr.timingmodelind is not None:
                ind = psr.timingmodelind
                fslc = slice(np.sum(self.npm[:ii]), np.sum(self.npm[:ii+1]))
                pslc = slice(ind, ind+self.npm[ii])
                ll_grad2_tm[:, fslc] = ll_grad2[:, pslc]

            if psr.fourierind is not None:
                findex = np.sum(self.npf[:ii])
                ind = psr.fourierind
                nfreq = self.npf[ii]
                pslc = slice(ind, ind+nfreq)
                fslc = slice(findex, findex+nfreq)

                ll_grad2_rn[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_rn[fslc] = self._d_b_d_B[pslc]

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_rn[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[None, psr.Zmask_M_only] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]


            if psr.dmfourierind is not None:
                fdmindex = np.sum(self.npfdm[:ii])
                ind = psr.dmfourierind
                nfreqdm = self.npfdm[ii]
                pslc = slice(ind, ind+nfreqdm)
                fslc = slice(fdmindex, fdmindex+nfreqdm)

                ll_grad2_dm[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_dm[fslc] = self._d_b_d_B[pslc]

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_dm[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[psr.Zmask_M_only][None, :] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]

            if psr.jitterind is not None:
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                fslc = slice(self.npu[:ii], self.npu[:ii+1])

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_J[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[psr.Zmask_M_only][None, :] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]

        # Now that we have that stuff memorized, multiply that stuff
        # Red noise
        for key, d_Phivec_d_p in self.d_Phivec_d_param.iteritems():
            # Extra grad for RN low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Phivec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_rn * self._d_Sigma_d_phi_tm *
                        d_Phivec_d_p[:,None], axis=0)[None, :], axis=1)

        # GW signals
        for key, d_Svec_d_p in self.d_Svec_d_param.iteritems():
            # Extra grad for RN low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Svec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_rn * self._d_Sigma_d_phi_tm *
                        d_Svec_d_p[:,None], axis=0)[None, :], axis=1)

        # DM variations
        for key, d_Thetavec_d_p in self.d_Thetavec_d_param.iteritems():
            # Extra grad for DM low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_dm * d_b_d_B_dm *
                    d_Thetavec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_dm * self._d_Sigma_d_theta_tm *
                        d_Thetavec_d_p[:,None], axis=0)[None, :], axis=1)

        # ECORR
        if psr.jitterind is not None:
            for ii, psr in enumerate(self.ptapsrs):
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                fslc = slice(self.npu[:ii], self.npu[:ii+1])

                for key, d_Jvec_d_p in psr.d_Jvec_d_param.iteritems():
                    for aa in range(a):
                        # Extra grad for ECORR low-level pars
                        extra_grad[aa, key] += \
                                np.sum(ll_grad2[aa, pslc] * 
                                self._d_b_d_B[pslc] * d_Jvec_d_p)

                        # Extra grad for timing model parameters
                        extra_grad[aa, key] += np.sum(
                            ll_grad2_tm[aa, pslc] * np.sum(
                                d_b_d_Sigma_J[fslc,:] *
                                self._d_Sigma_d_J_tm[pslc,:] *
                                d_Jvec_d_p[:,None], axis=0) )

        return extra_grad.reshape(ll_grad.shape)



class tmStingrayLikelihood2(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the tmStingrayLikelihood2 with a ptaLikelihood object"""
        super(tmStingrayLikelihood2, self).__init__(h5filename, jsonfilename, **kwargs)

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

        # For efficiency in calculating gradients, we'll have to store some
        # matrix-vector products
        d_lj_d_phi = np.zeros_like(self.Phivec)
        d_lj_d_theta = np.zeros_like(self.Thetavec)

        # For efficiency, calculate the following for the timing-model stingray
        # stuff
        ntmptot = np.sum(self.npm)
        d_Sigma_d_phi_tm = np.zeros((len(self.Phivec), ntmptot))
        d_Sigma_d_theta_tm = np.zeros((len(self.Thetavec), ntmptot))
        d_Sigma_d_J_tm = np.zeros((np.sum(self.npu), ntmptot))

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

            if psr.fourierind is not None:
                # Have a red noise stingray transformation
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc_phi = slice(findex, findex+nfs)
                phivec = self.Phivec[fslc_phi] + self.Svec[:nfs]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_F_only] + 1.0 / phivec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_F_only]
                index = psr.fourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_phi for down below
                d_lj_d_phi[fslc_phi] = 0.5 * Sigmavec / phivec**2

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / phivec**2
                d_mean_d_B = mean * Sigmavec / phivec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

                # Fill the quantities for the timing-model stingray
                gslc = psr.Zmask_F_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / phivec
                d_Sigma_d_phi_tm[fslc_phi, mslc] = (psr.sr_gamma[:,gslc] / \
                        (phivec**2*(psr.sr_A[gslc] + 1.0/phivec)**2)).T

            if psr.dmfourierind is not None:
                # Have a dm noise stingray transformation
                fdmindex = np.sum(self.npfdm[:ii])
                nfdms = self.npfdm[ii]
                fslc_theta = slice(fdmindex, fdmindex+nfdms)
                thetavec = self.Thetavec[fslc_theta]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_D_only] + 1.0 / thetavec)
                std = np.sqrt(Sigmavec)
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_D_only]
                index = psr.dmfourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_theta for down below
                d_lj_d_theta[fslc_theta] = 0.5 * Sigmavec / thetavec**2

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / thetavec**2
                d_mean_d_B = mean * Sigmavec / thetavec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

                # Fill the quantities for the timing-model stingray
                gslc = psr.Zmask_D_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / thetavec
                d_Sigma_d_theta_tm[fslc_theta, mslc] = -(psr.sr_gamma[:,gslc] / \
                            (thetavec**2*(psr.sr_A[gslc] + 1.0/thetavec)**2)).T

            if psr.jitterind is not None:
                # Have an ecor stingray transformation
                uindex = np.sum(self.npu[:ii])
                nus = self.npu[ii]
                fslc_J = slice(uindex, uindex+nus)

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_U_only] + \
                        1.0 / psr.Jvec)
                std = np.sqrt(Sigmavec)        # No hyper pars
                mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_U_only]
                index = psr.jitterind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                ######################################
                # Timing model Stingray calculations #
                ######################################
                gslc = psr.Zmask_U_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / psr.Jvec
                d_Sigma_d_J_tm[fslc_J, mslc] = -(psr.sr_gamma[:,gslc] / \
                            (psr.Jvec**2*(psr.sr_A[gslc] + 1.0/psr.Jvec)**2)).T

                # Do jitter gradients here, since they are kept track of on a
                # per-pulsar basis
                if calc_gradient:
                    for key, value in psr.d_Jvec_d_param.iteritems():
                        d_lj_d_J = 0.5 * Sigmavec / psr.Jvec**2
                        gradient[key] += np.sum(d_lj_d_J * value)

                        # Do the rest of the gradient stuff below...

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / psr.Jvec**2
                d_mean_d_B = mean * Sigmavec / psr.Jvec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

            if psr.timingmodelind is not None:
                # Now have all we need for the Stingray transformation
                Sigmavec = psr.sr_alpha / (psr.sr_beta - np.sum(
                        psr.sr_gamma[:,ntmpars:] /
                        sr_gamma_frac_den[ntmpars:], axis=1))

                Sigmavec_old = psr.sr_Sigma1[psr.Zmask_M_only]  # No hyper pars
                std = np.sqrt(Sigmavec)
                std_old = np.sqrt(Sigmavec_old)

                # We keep the old estimate of the mean, because is it less
                # biased
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_M_only]
                mean = Sigmavec_old * psr.sr_ZNyvec[psr.Zmask_M_only]
                index = psr.timingmodelind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                log_jacob += np.sum(np.log(std))

                d_b_d_xi[slc] = std

                # Multiply the d_Sigma_d_xxx_tm with a coefficient that depends
                # on all the hyper parameters
                d_Sigma_tm_coeff = psr.sr_alpha / (psr.sr_beta - np.sum(
                        psr.sr_gamma[:,ntmpars:] /
                        sr_gamma_frac_den[None, ntmpars:], axis=1))**2
                d_Sigma_d_phi_tm[fslc_phi,mslc] = (
                        d_Sigma_d_phi_tm[fslc_phi,mslc] *
                        d_Sigma_tm_coeff[None,:])
                d_Sigma_d_theta_tm[fslc_theta,mslc] = (
                        d_Sigma_d_theta_tm[fslc_theta,mslc] *
                        d_Sigma_tm_coeff)
                d_Sigma_d_J_tm[fslc_J,mslc] = (
                        d_Sigma_d_J_tm[fslc_J,mslc] *
                        d_Sigma_tm_coeff)

        ##############################################################
        ##############################################################
        ### ERROR BELOW: We need to loop over psr again ##############
        ##############################################################
        ##############################################################

        if calc_gradient:
            # For the timing model Stingray log-jacobian gradient, we need to
            # multiply the up-to-now recorded values with the full coefficient,
            # summed over the noise low-level parameters
            #d_Sigma_tm_coeff = psr.sr_alpha / (psr.sr_beta - np.sum(
            #        psr.sr_gamma / sr_gamma_frac_den, axis=1))**2

            # Red noise
            TMsigma = sigma[mslc_par]**2
            if psr.fourierind is not None:
                # TODO: Use proper broadcasting
                d_Jac_d_B = 0.5 * d_Sigma_d_phi_tm / TMsigma

                for key, value in self.d_Phivec_d_param.iteritems():
                    # Log-jacobian for red noise Fourier terms
                    gradient[key] += np.sum(d_lj_d_phi * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

                # GW signals
                for key, value in self.d_Svec_d_param.iteritems():
                    # Log-jacobian for red noise Fourier terms
                    gradient[key] += np.sum(d_lj_d_phi * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

            # DM variations
            if psr.dmfourierind is not None:
                # TODO: Use proper broadcasting like right above here
                #d_Jac_d_B = (d_Sigma_d_theta_tm.T * (0.5 / d_Sigma_d_theta_std**2) ).T
                d_Jac_d_B = 0.5 * d_Sigma_d_theta_tm / TMsigma
                for key, value in self.d_Thetavec_d_param.iteritems():
                    # Log-jacobian for DM variation Fourier terms
                    gradient[key] += np.sum(d_lj_d_theta * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

            # ECORR
            if psr.jitterind is not None:
                #d_Jac_d_B = (d_Sigma_d_J_tm.T * (0.5 / d_Sigma_d_J_std**2) ).T
                d_Jac_d_B = 0.5 * d_Sigma_d_J_tm / TMsigma
                for ii, psr in enumerate(self.ptapsrs):
                    fslc_J = slice(self.npu[:ii], self.npu[:ii+1])
                    for key, value in psr.d_Jvec_d_param.iteritems():
                        # The Log-jacobian gradient was already calculated above
                        # (TODO: change to here, now that we are here?)

                        # Log-jacobian for timing model (Just add the whole shebang)
                        gradient[key] += np.sum(np.sum(d_Jac_d_B[fslc_J,:].T * value, axis=0))

        self._mu = mu                   # Mean of stingray transform
        self._sigma = sigma             # Slope of stingray transform
        self._log_jacob = log_jacob     # Log-jacobian of transform
        self._gradient = gradient       # Gradient of log-jacobian
        self._d_b_d_xi = d_b_d_xi       # d_x_d_p
        self._d_b_d_B = d_b_d_B         # d_x_d_B, with B hyper-pars

        # For the timing-model Stingray, we'll have to save some more quantities
        self._d_Sigma_d_phi_tm = d_Sigma_d_phi_tm
        self._d_Sigma_d_theta_tm = d_Sigma_d_theta_tm
        self._d_Sigma_d_J_tm = d_Sigma_d_J_tm 

    def stingray_hessian_quants(self, p, set_hyper_pars=True):
        """Calculate quantities necessary for the Hessian calculations"""
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

        ntmptot = np.sum(self.npm)

        # For computational efficiency, we'll have to memorize the
        # ll_grad * d_b_d_B vectors
        d_b_d_B_rn = np.zeros_like(self.Phivec)
        ll_grad2_rn = np.zeros( (ll_grad2.shape[0], len(self.Phivec)) )
        d_b_d_B_dm = np.zeros_like(self.Phivec)
        ll_grad2_dm = np.zeros( (ll_grad2.shape[0], len(self.Thetavec)) )
        ll_grad2_tm = np.zeros( (ll_grad2.shape[0], np.sum(self.npm)) )

        # And these, for the timing model Stingray
        d_b_d_Sigma_rn = np.zeros((len(self.Phivec), ntmptot))
        d_b_d_Sigma_dm = np.zeros((len(self.Thetavec), ntmptot))
        d_b_d_Sigma_J = np.zeros((np.sum(self.npu), ntmptot))

        for ii, psr in enumerate(self.ptapsrs):
            mslc = slice(np.sum(self.npm[:ii]), np.sum(self.npm[:ii+1]))
            m_index = psr.timingmodelind        # Assuming this exists
            mslc_par = slice(m_index, m_index+self.npm[ii])
            if psr.timingmodelind is not None:
                ind = psr.timingmodelind
                fslc = slice(np.sum(self.npm[:ii]), np.sum(self.npm[:ii+1]))
                pslc = slice(ind, ind+self.npm[ii])
                ll_grad2_tm[:, fslc] = ll_grad2[:, pslc]

            if psr.fourierind is not None:
                findex = np.sum(self.npf[:ii])
                ind = psr.fourierind
                nfreq = self.npf[ii]
                pslc = slice(ind, ind+nfreq)
                fslc = slice(findex, findex+nfreq)

                ll_grad2_rn[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_rn[fslc] = self._d_b_d_B[pslc]

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_rn[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[None, psr.Zmask_M_only] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]


            if psr.dmfourierind is not None:
                fdmindex = np.sum(self.npfdm[:ii])
                ind = psr.dmfourierind
                nfreqdm = self.npfdm[ii]
                pslc = slice(ind, ind+nfreqdm)
                fslc = slice(fdmindex, fdmindex+nfreqdm)

                ll_grad2_dm[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_dm[fslc] = self._d_b_d_B[pslc]

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_dm[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[psr.Zmask_M_only][None, :] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]

            if psr.jitterind is not None:
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                fslc = slice(self.npu[:ii], self.npu[:ii+1])

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_J[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[psr.Zmask_M_only][None, :] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]

        # Now that we have that stuff memorized, multiply that stuff
        # Red noise
        for key, d_Phivec_d_p in self.d_Phivec_d_param.iteritems():
            # Extra grad for RN low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Phivec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_rn * self._d_Sigma_d_phi_tm *
                        d_Phivec_d_p[:,None], axis=0)[None, :], axis=1)

        # GW signals
        for key, d_Svec_d_p in self.d_Svec_d_param.iteritems():
            # Extra grad for RN low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Svec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_rn * self._d_Sigma_d_phi_tm *
                        d_Svec_d_p[:,None], axis=0)[None, :], axis=1)

        # DM variations
        for key, d_Thetavec_d_p in self.d_Thetavec_d_param.iteritems():
            # Extra grad for DM low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_dm * d_b_d_B_dm *
                    d_Thetavec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_dm * self._d_Sigma_d_theta_tm *
                        d_Thetavec_d_p[:,None], axis=0)[None, :], axis=1)

        # ECORR
        if psr.jitterind is not None:
            for ii, psr in enumerate(self.ptapsrs):
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                fslc = slice(self.npu[:ii], self.npu[:ii+1])

                for key, d_Jvec_d_p in psr.d_Jvec_d_param.iteritems():
                    for aa in range(a):
                        # Extra grad for ECORR low-level pars
                        extra_grad[aa, key] += \
                                np.sum(ll_grad2[aa, pslc] * 
                                self._d_b_d_B[pslc] * d_Jvec_d_p)

                        # Extra grad for timing model parameters
                        extra_grad[aa, key] += np.sum(
                            ll_grad2_tm[aa, pslc] * np.sum(
                                d_b_d_Sigma_J[fslc,:] *
                                self._d_Sigma_d_J_tm[pslc,:] *
                                d_Jvec_d_p[:,None], axis=0) )

        return extra_grad.reshape(ll_grad.shape)



class muStingrayLikelihood(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the muStingrayLikelihood with a ptaLikelihood object"""
        super(muStingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

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
        """For the mean-part of the stingray transform, we need the full Sigma
        matrix"""
        Sigma_inv = np.copy(psr.sr_ZNZ)
        Sigma_inv_diag = np.diag(Sigma_inv)

        #res_rms = np.std(psr.residuals)            # RMS of residuals
        #Zrms = np.std(psr.Zmat, axis=0, ddof=0)         # RMS of Z-column
        #amp_prior = res_rms / Zrms                  # Relative amplitude
        #constraints = 1.0 / amp_prior**2            # Constraint value
        #constraints[:psr.Mmat.shape[1]] = 0.0      # Don't constrain timing model

        # Construct the full Sigma matrix
        np.fill_diagonal(Sigma_inv, Sigma_inv_diag + 1.0*Beta_inv_diag)
        #np.fill_diagonal(Sigma_inv, Sigma_inv_diag + 1.0e-3 * constraints)
        cf = sl.cho_factor(Sigma_inv)

        return sl.cho_solve(cf, np.eye(len(Sigma_inv)))

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

        # For efficiency in calculating gradients, we'll have to store some
        # matrix-vector products
        d_lj_d_phi = np.zeros_like(self.Phivec)
        d_lj_d_theta = np.zeros_like(self.Thetavec)

        # For efficiency, calculate the following for the timing-model stingray
        # stuff
        ntmptot = np.sum(self.npm)
        d_Sigma_d_phi_tm = np.zeros((len(self.Phivec), ntmptot))
        d_Sigma_d_theta_tm = np.zeros((len(self.Thetavec), ntmptot))
        d_Sigma_d_J_tm = np.zeros((np.sum(self.npu), ntmptot))

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
            psr.sr_Sigma = self.get_psr_Sigma(ii, psr, psr.sr_Beta_inv)
            psr.sr_mu = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)

            if psr.fourierind is not None:
                # Have a red noise stingray transformation
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc_phi = slice(findex, findex+nfs)
                phivec = self.Phivec[fslc_phi] + self.Svec[:nfs]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_F_only] + 1.0 / phivec)
                std = np.sqrt(Sigmavec)
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_F_only]
                mean = psr.sr_mu[psr.Zmask_F_only]

                index = psr.fourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_phi for down below
                d_lj_d_phi[fslc_phi] = 0.5 * Sigmavec / phivec**2

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / phivec**2
                d_mean_d_B = mean * Sigmavec / phivec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 0.0 # 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

                # Fill the quantities for the timing-model stingray
                gslc = psr.Zmask_F_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / phivec
                d_Sigma_d_phi_tm[fslc_phi, mslc] = (psr.sr_gamma[:,gslc] / \
                        (phivec**2*(psr.sr_A[gslc] + 1.0/phivec)**2)).T

            if psr.dmfourierind is not None:
                # Have a dm noise stingray transformation
                fdmindex = np.sum(self.npfdm[:ii])
                nfdms = self.npfdm[ii]
                fslc_theta = slice(fdmindex, fdmindex+nfdms)
                thetavec = self.Thetavec[fslc_theta]

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_D_only] + 1.0 / thetavec)
                std = np.sqrt(Sigmavec)
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_D_only]
                mean = psr.sr_mu[psr.Zmask_D_only]

                index = psr.dmfourierind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                # Memorize d_lj_d_theta for down below
                d_lj_d_theta[fslc_theta] = 0.5 * Sigmavec / thetavec**2

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / thetavec**2
                d_mean_d_B = mean * Sigmavec / thetavec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 0.0 # 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

                # Fill the quantities for the timing-model stingray
                gslc = psr.Zmask_D_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / thetavec
                d_Sigma_d_theta_tm[fslc_theta, mslc] = -(psr.sr_gamma[:,gslc] / \
                            (thetavec**2*(psr.sr_A[gslc] + 1.0/thetavec)**2)).T

            if psr.jitterind is not None:
                # Have an ecor stingray transformation
                uindex = np.sum(self.npu[:ii])
                nus = self.npu[ii]
                fslc_J = slice(uindex, uindex+nus)

                Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_U_only] + \
                        1.0 / psr.Jvec)
                std = np.sqrt(Sigmavec)        # No hyper pars
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_U_only]
                mean = psr.sr_mu[psr.Zmask_U_only]

                index = psr.jitterind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                ######################################
                # Timing model Stingray calculations #
                ######################################
                gslc = psr.Zmask_U_only
                sr_gamma_frac_den[gslc] = psr.sr_A[gslc] + 1.0 / psr.Jvec
                d_Sigma_d_J_tm[fslc_J, mslc] = -(psr.sr_gamma[:,gslc] / \
                            (psr.Jvec**2*(psr.sr_A[gslc] + 1.0/psr.Jvec)**2)).T

                # Do jitter gradients here, since they are kept track of on a
                # per-pulsar basis
                if calc_gradient:
                    for key, value in psr.d_Jvec_d_param.iteritems():
                        d_lj_d_J = 0.5 * Sigmavec / psr.Jvec**2
                        gradient[key] += np.sum(d_lj_d_J * value)

                        # Do the rest of the gradient stuff below...

                d_std_d_B = 0.5 * (Sigmavec ** 1.5) / psr.Jvec**2
                d_mean_d_B = mean * Sigmavec / psr.Jvec**2
                d_b_d_std = p[slc]
                d_b_d_mean = 0.0 # 1.0
                d_b_d_B[slc] = d_b_d_std * d_std_d_B + d_b_d_mean * d_mean_d_B

                d_b_d_xi[slc] = std
                log_jacob += np.sum(np.log(std))

            if psr.timingmodelind is not None:
                # Now have all we need for the Stingray transformation
                Sigmavec = psr.sr_alpha / (psr.sr_beta - np.sum(
                        psr.sr_gamma[:,ntmpars:] /
                        sr_gamma_frac_den[ntmpars:], axis=1))

                #Sigmavec_old = psr.sr_Sigma1[psr.Zmask_M_only]  # No hyper pars
                std = np.sqrt(Sigmavec)
                #std_old = np.sqrt(Sigmavec_old)

                # We keep the old estimate of the mean, because is it less
                # biased
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_M_only]
                #mean = Sigmavec_old * psr.sr_ZNyvec[psr.Zmask_M_only]
                mean = psr.sr_mu[psr.Zmask_M_only]

                index = psr.timingmodelind
                npars = len(std)
                slc = slice(index, index+npars)

                # The linear transformation is now defined as:
                mu[slc] = mean
                sigma[slc] = std

                log_jacob += np.sum(np.log(std))

                d_b_d_xi[slc] = std

                # Multiply the d_Sigma_d_xxx_tm with a coefficient that depends
                # on all the hyper parameters
                d_Sigma_tm_coeff = psr.sr_alpha / (psr.sr_beta - np.sum(
                        psr.sr_gamma[:,ntmpars:] /
                        sr_gamma_frac_den[None, ntmpars:], axis=1))**2
                d_Sigma_d_phi_tm[fslc_phi,mslc] = (
                        d_Sigma_d_phi_tm[fslc_phi,mslc] *
                        d_Sigma_tm_coeff[None,:])
                d_Sigma_d_theta_tm[fslc_theta,mslc] = (
                        d_Sigma_d_theta_tm[fslc_theta,mslc] *
                        d_Sigma_tm_coeff)
                d_Sigma_d_J_tm[fslc_J,mslc] = (
                        d_Sigma_d_J_tm[fslc_J,mslc] *
                        d_Sigma_tm_coeff)

        ##############################################################
        ##############################################################
        ### ERROR BELOW: We need to loop over psr again ##############
        ##############################################################
        ##############################################################

        if calc_gradient:
            # For the timing model Stingray log-jacobian gradient, we need to
            # multiply the up-to-now recorded values with the full coefficient,
            # summed over the noise low-level parameters
            #d_Sigma_tm_coeff = psr.sr_alpha / (psr.sr_beta - np.sum(
            #        psr.sr_gamma / sr_gamma_frac_den, axis=1))**2

            # Red noise
            TMsigma = sigma[mslc_par]**2
            if psr.fourierind is not None:
                # TODO: Use proper broadcasting
                d_Jac_d_B = 0.5 * d_Sigma_d_phi_tm / TMsigma

                for key, value in self.d_Phivec_d_param.iteritems():
                    # Log-jacobian for red noise Fourier terms
                    gradient[key] += np.sum(d_lj_d_phi * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

                # GW signals
                for key, value in self.d_Svec_d_param.iteritems():
                    # Log-jacobian for red noise Fourier terms
                    gradient[key] += np.sum(d_lj_d_phi * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

            # DM variations
            if psr.dmfourierind is not None:
                # TODO: Use proper broadcasting like right above here
                #d_Jac_d_B = (d_Sigma_d_theta_tm.T * (0.5 / d_Sigma_d_theta_std**2) ).T
                d_Jac_d_B = 0.5 * d_Sigma_d_theta_tm / TMsigma
                for key, value in self.d_Thetavec_d_param.iteritems():
                    # Log-jacobian for DM variation Fourier terms
                    gradient[key] += np.sum(d_lj_d_theta * value)

                    # Log-jacobian for timing model (Just add the whole shebang)
                    gradient[key] += np.sum(np.sum(d_Jac_d_B.T * value, axis=0))

            # ECORR
            if psr.jitterind is not None:
                #d_Jac_d_B = (d_Sigma_d_J_tm.T * (0.5 / d_Sigma_d_J_std**2) ).T
                d_Jac_d_B = 0.5 * d_Sigma_d_J_tm / TMsigma
                for ii, psr in enumerate(self.ptapsrs):
                    fslc_J = slice(self.npu[:ii], self.npu[:ii+1])
                    for key, value in psr.d_Jvec_d_param.iteritems():
                        # The Log-jacobian gradient was already calculated above
                        # (TODO: change to here, now that we are here?)

                        # Log-jacobian for timing model (Just add the whole shebang)
                        gradient[key] += np.sum(np.sum(d_Jac_d_B[fslc_J,:].T * value, axis=0))

        self._mu = mu                   # Mean of stingray transform
        self._sigma = sigma             # Slope of stingray transform
        self._log_jacob = log_jacob     # Log-jacobian of transform
        self._gradient = gradient       # Gradient of log-jacobian
        self._d_b_d_xi = d_b_d_xi       # d_x_d_p
        self._d_b_d_B = d_b_d_B         # d_x_d_B, with B hyper-pars

        # For the timing-model Stingray, we'll have to save some more quantities
        self._d_Sigma_d_phi_tm = d_Sigma_d_phi_tm
        self._d_Sigma_d_theta_tm = d_Sigma_d_theta_tm
        self._d_Sigma_d_J_tm = d_Sigma_d_J_tm 

    def stingray_hessian_quants(self, p, set_hyper_pars=True):
        """Calculate quantities necessary for the Hessian calculations"""
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

        ntmptot = np.sum(self.npm)

        # For computational efficiency, we'll have to memorize the
        # ll_grad * d_b_d_B vectors
        d_b_d_B_rn = np.zeros_like(self.Phivec)
        ll_grad2_rn = np.zeros( (ll_grad2.shape[0], len(self.Phivec)) )
        d_b_d_B_dm = np.zeros_like(self.Phivec)
        ll_grad2_dm = np.zeros( (ll_grad2.shape[0], len(self.Thetavec)) )
        ll_grad2_tm = np.zeros( (ll_grad2.shape[0], np.sum(self.npm)) )

        # And these, for the timing model Stingray
        d_b_d_Sigma_rn = np.zeros((len(self.Phivec), ntmptot))
        d_b_d_Sigma_dm = np.zeros((len(self.Thetavec), ntmptot))
        d_b_d_Sigma_J = np.zeros((np.sum(self.npu), ntmptot))

        for ii, psr in enumerate(self.ptapsrs):
            mslc = slice(np.sum(self.npm[:ii]), np.sum(self.npm[:ii+1]))
            m_index = psr.timingmodelind        # Assuming this exists
            mslc_par = slice(m_index, m_index+self.npm[ii])

            if psr.timingmodelind is not None:
                ind = psr.timingmodelind
                fslc = slice(np.sum(self.npm[:ii]), np.sum(self.npm[:ii+1]))
                pslc = slice(ind, ind+self.npm[ii])
                ll_grad2_tm[:, fslc] = ll_grad2[:, pslc]

            if psr.fourierind is not None:
                findex = np.sum(self.npf[:ii])
                ind = psr.fourierind
                nfreq = self.npf[ii]
                pslc = slice(ind, ind+nfreq)
                fslc = slice(findex, findex+nfreq)

                ll_grad2_rn[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_rn[fslc] = self._d_b_d_B[pslc]

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_rn[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[None, psr.Zmask_M_only] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]


            if psr.dmfourierind is not None:
                fdmindex = np.sum(self.npfdm[:ii])
                ind = psr.dmfourierind
                nfreqdm = self.npfdm[ii]
                pslc = slice(ind, ind+nfreqdm)
                fslc = slice(fdmindex, fdmindex+nfreqdm)

                ll_grad2_dm[:, fslc] = ll_grad2[:, pslc]
                d_b_d_B_dm[fslc] = self._d_b_d_B[pslc]

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_dm[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[psr.Zmask_M_only][None, :] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]

            if psr.jitterind is not None:
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                fslc = slice(np.sum(self.npu[:ii]), np.sum(self.npu[:ii+1]))

                # For the timing model Stingray, we need d_b_d_Sigma
                d_b_d_Sigma_J[fslc, mslc] = \
                        0.0 / psr.sr_Sigma1[psr.Zmask_M_only][None, :] + \
                        0.5 * p[None, mslc_par] / \
                        self._sigma[None, mslc_par]

        # Now that we have that stuff memorized, multiply that stuff
        # Red noise
        for key, d_Phivec_d_p in self.d_Phivec_d_param.iteritems():
            # Extra grad for RN low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Phivec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_rn * self._d_Sigma_d_phi_tm *
                        d_Phivec_d_p[:,None], axis=0)[None, :], axis=1)

            # Extra grad for the mean
            for ii, psr in enumerate(self.ptapsrs):
                Wv = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
                fslc_phi = slice(np.sum(self.npf[:ii]), np.sum(self.npf[:ii+1]))
                slc_sig = psr.Zmask_F_only
                pslc_tot = self.get_par_psr_sigma_inds(ii, psr)

                WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                        psr.sr_Beta_inv[slc_sig]**2 *
                        d_Phivec_d_p[fslc_phi] * Wv[slc_sig])

                extra_grad[:, key] += np.sum(ll_grad2[:, pslc_tot] * 
                        WBWv[None, :], axis=1)


        # GW signals
        for key, d_Svec_d_p in self.d_Svec_d_param.iteritems():
            # Extra grad for RN low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_rn * d_b_d_B_rn *
                    d_Svec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_rn * self._d_Sigma_d_phi_tm *
                        d_Svec_d_p[:,None], axis=0)[None, :], axis=1)

            # Extra grad for the mean
            for ii, psr in enumerate(self.ptapsrs):
                Wv = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
                fslc_phi = slice(np.sum(self.npf[:ii]), np.sum(self.npf[:ii+1]))
                slc_sig = psr.Zmask_F_only
                pslc_tot = self.get_par_psr_sigma_inds(ii, psr)

                WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                        psr.sr_Beta_inv[slc_sig]**2 *
                        d_Svec_d_p[fslc_phi] * Wv[slc_sig])

                extra_grad[:, key] += np.sum(ll_grad2[:, pslc_tot] * 
                        WBWv[None, :], axis=1)

        # DM variations
        for key, d_Thetavec_d_p in self.d_Thetavec_d_param.iteritems():
            # Extra grad for DM low-level pars
            extra_grad[:, key] += np.sum(ll_grad2_dm * d_b_d_B_dm *
                    d_Thetavec_d_p, axis=1)

            # Extra grad for timing model parameters
            extra_grad[:, key] += np.sum(
                    ll_grad2_tm * np.sum(
                        d_b_d_Sigma_dm * self._d_Sigma_d_theta_tm *
                        d_Thetavec_d_p[:,None], axis=0)[None, :], axis=1)

            # Extra grad for the mean
            for ii, psr in enumerate(self.ptapsrs):
                Wv = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
                fslc_theta = slice(np.sum(self.npfdm[:ii]), np.sum(self.npfdm[:ii+1]))
                slc_sig = psr.Zmask_D_only
                pslc_tot = self.get_par_psr_sigma_inds(ii, psr)

                WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                        psr.sr_Beta_inv[slc_sig]**2 *
                        d_Thetavec_d_p[fslc_theta] * Wv[slc_sig])

                extra_grad[:, key] += np.sum(ll_grad2[:, pslc_tot] * 
                        WBWv[None, :], axis=1)

        # ECORR
        if psr.jitterind is not None:
            for ii, psr in enumerate(self.ptapsrs):
                ind = psr.jitterind
                npus = self.npu[ii]
                pslc = slice(ind, ind+npus)
                fslc = slice(np.sum(self.npu[:ii]), np.sum(self.npu[:ii+1]))

                for key, d_Jvec_d_p in psr.d_Jvec_d_param.iteritems():
                    for aa in range(a):
                        # Extra grad for ECORR low-level pars
                        extra_grad[aa, key] += \
                                np.sum(ll_grad2[aa, pslc] * 
                                self._d_b_d_B[pslc] * d_Jvec_d_p)

                        # Extra grad for timing model parameters
                        extra_grad[aa, key] += np.sum(
                            ll_grad2_tm[aa, pslc] * np.sum(
                                d_b_d_Sigma_J[fslc,:] *
                                self._d_Sigma_d_J_tm[pslc,:] *
                                d_Jvec_d_p[:,None], axis=0) )

                        # Extra grad for the mean
                        Wv = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
                        fslc_J = slice(np.sum(self.npu[:ii], self.npu[:ii+1]))
                        slc_sig = psr.Zmask_U_only
                        #pslc_tot = self.get_par_psr_sigma_inds(ii, psr)

                        WBWv = np.dot(psr.sr_Sigma[:,slc_sig],
                                psr.sr_Beta_inv[slc_sig]**2 *
                                d_Thetavec_d_p[fslc_J] * Wv[slc_sig])

                        extra_grad[aa, key] += np.sum(ll_grad2[aa, pslc] * 
                                WBWv)

        return extra_grad.reshape(ll_grad.shape)


class msStingrayLikelihood(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the muStingrayLikelihood with a ptaLikelihood object"""
        super(msStingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

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
        """For the mean-part of the stingray transform, we need the full Sigma
        matrix"""
        Sigma_inv = np.copy(psr.sr_ZNZ)
        Sigma_inv_diag = np.diag(Sigma_inv)

        # Construct the full Sigma matrix
        np.fill_diagonal(Sigma_inv, Sigma_inv_diag + Beta_inv_diag)
        cf = sl.cho_factor(Sigma_inv)

        return sl.cho_solve(cf, np.eye(len(Sigma_inv)))

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

        # For efficiency in calculating gradients, we'll have to store some
        # matrix-vector products
        #d_lj_d_phi = np.zeros_like(self.Phivec)
        #d_lj_d_theta = np.zeros_like(self.Thetavec)

        # For efficiency, calculate the following for the timing-model stingray
        # stuff
        #ntmptot = np.sum(self.npm)
        #d_Sigma_d_phi_tm = np.zeros((len(self.Phivec), ntmptot))
        #d_Sigma_d_theta_tm = np.zeros((len(self.Thetavec), ntmptot))
        #d_Sigma_d_J_tm = np.zeros((np.sum(self.npu), ntmptot))

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
            psr.sr_Sigma = self.get_psr_Sigma(ii, psr, psr.sr_Beta_inv)
            psr.sr_mu = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
            psr.sr_sigma = np.sqrt(np.diag(psr.sr_Sigma))
            psr.diagSBS = dict()

            if psr.fourierind is not None:
                # Have a red noise stingray transformation
                findex = np.sum(self.npf[:ii])
                nfs = self.npf[ii]
                fslc_phi = slice(findex, findex+nfs)
                phivec = self.Phivec[fslc_phi] + self.Svec[:nfs]

                #Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_F_only] + 1.0 / phivec)
                Sigmavec = psr.sr_sigma[psr.Zmask_F_only]**2
                std = np.sqrt(Sigmavec)
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_F_only]
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

                #Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_D_only] + 1.0 / thetavec)
                Sigmavec = psr.sr_sigma[psr.Zmask_D_only]**2
                std = np.sqrt(Sigmavec)
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_D_only]
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

                #Sigmavec = 1.0/(1.0/psr.sr_Sigma2[psr.Zmask_U_only] + \
                #        1.0 / psr.Jvec)
                Sigmavec = psr.sr_sigma[psr.Zmask_U_only]**2
                std = np.sqrt(Sigmavec)        # No hyper pars
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_U_only]
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
                #Sigmavec = psr.sr_alpha / (psr.sr_beta - np.sum(
                #        psr.sr_gamma[:,ntmpars:] /
                #        sr_gamma_frac_den[ntmpars:], axis=1))
                Sigmavec = psr.sr_sigma[psr.Zmask_M_only]**2

                #Sigmavec_old = psr.sr_Sigma1[psr.Zmask_M_only]  # No hyper pars
                std = np.sqrt(Sigmavec)
                #std_old = np.sqrt(Sigmavec_old)

                # We keep the old estimate of the mean, because is it less
                # biased
                #mean = Sigmavec * psr.sr_ZNyvec[psr.Zmask_M_only]
                #mean = Sigmavec_old * psr.sr_ZNyvec[psr.Zmask_M_only]
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


class fullStingrayLikelihood(stingrayLikelihood):
    """
    Wrapper class of the likelihood for Hamiltonian samplers. This implements a
    coordinate transformation for all low-level parameters that gets rid of the
    stingray continuous phase transition

    NOTE: this transformation automagically sets the start position of the
          low-level parameters to 0.1. Nonzero, but close enough to be decent
    """
    def __init__(self, h5filename=None, jsonfilename=None, **kwargs):
        """Initialize the muStingrayLikelihood with a ptaLikelihood object"""
        super(msStingrayLikelihood, self).__init__(h5filename, jsonfilename, **kwargs)

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
        """For the mean-part of the stingray transform, we need the full Sigma
        matrix"""
        Sigma_inv = np.copy(psr.sr_ZNZ)
        Sigma_inv_diag = np.diag(Sigma_inv)

        # Construct the full Sigma matrix
        np.fill_diagonal(Sigma_inv, Sigma_inv_diag + Beta_inv_diag)
        cf = sl.cho_factor(Sigma_inv)

        return sl.cho_solve(cf, np.eye(len(Sigma_inv)))

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
            psr.sr_Sigma = self.get_psr_Sigma(ii, psr, psr.sr_Beta_inv)
            psr.sr_mu = np.dot(psr.sr_Sigma, psr.sr_ZNyvec)
            psr.sr_sigma = np.sqrt(np.diag(psr.sr_Sigma))
            psr.diagSBS = dict()

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

