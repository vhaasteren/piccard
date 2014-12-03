from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os as os
import glob
import sys

from piccard import *
from piccard_pso import *
from jitterext import *
import PTMCMC_generic as ptmcmc


"""
This file implements a blocked Gibbs sampler. Gibbs sampling implements a
specific type of jump in an MCMC, just like e.g. Metropolis-Hastings, and in
Pulsar Timing data analysis it can be used to increase the mixing rate of the
chain. We still use the PAL/PAL2 version of PTMCMC_generic to do the actual MCMC
steps, but the steps are performed in parameter blocks.
"""

def RunGibbs_mark2(likob, steps, chainsdir, noWrite=False):
    """
    Run a blocked Gibbs sampler on the full likelihood, including all quadratic
    parameters numerically. The hyper-parameters are proposed using adaptive
    Metropolis proposals, the quadratic parameters are proposed by being sampled
    analytically given the hyper-parameters.

    Parameters are grouped into several categories:
    1) a, the Fourier coefficients and timing model parameters
    2) N, the white noise parameters
    3) Phi, the red noise PSD coefficients
    4) Jitter: pulse jitter/ecorr. Unlike in mark1, always included in 'N'
    5) Deterministic: all deterministic sources not described elsewhere

    @param likob:       The likelihood object, containing everything
    @param steps:       The number of full-circle Gibbs steps to take
    @param chainsdir:   Where to save the MCMC chain
    @param noWrite:     If True, do not write results to file
    """
    if not likob.likfunc in ['gibbs']:
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    # Save the description of all the parameters
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    # Clear the file for writing
    if not noWrite:
        chainfilename = chainsdir + '/chain_1.txt'
        chainfile = open(chainfilename, 'w')
        chainfile.close()

        #chainfilename_b = chainsdir + '/chain_1-b.txt'
        #chainfile_b = open(chainfilename_b, 'w')
        #chainfile_b.close()


        # Also save the residuals for all pulsars
        likob.saveResiduals(chainsdir)

    # Dump samples to file every dumpint steps (no thinning)
    dumpint = 100

    ndim = likob.dimensions         # The non-Gibbs model parameters
    ncoeffs = np.sum(likob.npz)     # The Gibbs-only parameters/coefficients

    # The actual (Gibbs) MCMC chain output:
    samples = np.zeros((min(dumpint, steps), ndim+ncoeffs))
    #samples2 = np.zeros((min(dumpint, steps), ncoeffs))
    loglik = np.zeros(min(dumpint, steps))
    logpost = np.zeros(min(dumpint, steps))

    # Hyper parameters, and full parameter array (need to init sub-samplers)
    hpars = likob.pstart.copy()      # The Gibbs hyper-parameters
    width = likob.pwidth.copy()
    apars = np.append(hpars, np.zeros(ncoeffs))     # All pars = hyper+quad pars
    awidth = np.append(width, np.zeros(ncoeffs))

    # Allocate all the sub-samplers we need (the samplers for the conditional
    # probabilities
    sampler_N = []          # White noise (per pulsar)
    sampler_F = None        # Red noise (full array, b/c correlations)
    sampler_D = []          # DM variations (per pulsar)
    sampler_J = []          # ECORR/Jitter (per pulsar)
    for ii, psr in enumerate(likob.ptapsrs):
        # Start with the noise search for pulsar ii
        if 'jitter' in likob.gibbsmodel:
            # Parameter mask for the white noise signals
            Nmask = likob.gibbs_get_signal_mask(ii, ['efac', 'equad'], ncoeffs)
        else:
            # Parameter mask for the white noise + ecorr/jitter signals
            Nmask = likob.gibbs_get_signal_mask(ii, \
                    ['efac', 'equad', 'jitter', 'cequad'], ncoeffs)

        if np.sum(Nmask) > 0:
            # If number of signals > 0, initialize the sub-sampler
            psrNpars = apars[Nmask]
            psrNcov = np.diag(awidth[Nmask]**2)
            Ndim = np.sum(Nmask)
            sampler_N.append(ptmcmc.PTSampler(Ndim, \
                likob.gibbs_psr_noise_loglikelihood_mar, \
                likob.gibbs_psr_noise_logprior, \
                cov=psrNcov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True, \
                loglargs=[ii, Nmask, apars], \
                logpargs=[ii, Nmask, apars]))
        else:
            sampler_N.append(None)

        if 'rednoise' in likob.gibbsmodel:
            # Red noise is done for all pulsars combined, so we have nothing to
            # do here
            pass

        if 'dm' in likob.gibbsmodel:
            # Parameter mask for signals for DM variations
            Dmask = likob.gibbs_get_signal_mask(ii, ['dmpowerlaw', 'dmspectrum'], ncoeffs)
            if np.sum(Dmask) > 0:
                psrDpars = apars[Dmask]
                psrDcov = np.diag(awidth[Dmask]**2)
                Ddim = np.sum(Dmask)
                sampler_D.append(ptmcmc.PTSampler(Ddim, \
                    likob.gibbs_psr_DM_loglikelihood_mar, \
                    likob.gibbs_psr_DM_logprior, \
                    cov=psrDcov, outDir='./gibbs-chains/', \
                    verbose=False, nowrite=True, \
                    loglargs=[ii, Dmask, apars], \
                    logpargs=[ii, Dmask, apars]))
            else:
                sampler_D.append(None)

        if 'jitter' in likob.gibbsmodel:
            # Pulse 'jitter'/ecorr, in case we did not include it together with
            # the white noise parameters above
            Jmask = likob.gibbs_get_signal_mask(ii, ['jitter'], ncoeffs)
            if np.sum(Jmask) > 0:
                psrJpars = apars[Dmask]
                psrJcov = np.diag(awidth[Jmask]**2)
                Jdim = np.sum(Jmask)
                sampler_J.append(ptmcmc.PTSampler(Jdim, \
                    likob.gibbs_psr_J_loglikelihood_mar, \
                    likob.gibbs_psr_J_logprior, \
                    cov=psrJcov, outDir='./gibbs-chains/', \
                    verbose=False, nowrite=True, \
                    loglargs=[ii, Dmask, apars], \
                    logpargs=[ii, Dmask, apars]))
            else:
                sampler_J.append(None)

    if 'rednoise' in likob.gibbsmodel:
        # For the full array (hence the -2 below), get the signal mask for the
        # red noise, and initialize the sub-sampler
        Fmask = likob.gibbs_get_signal_mask(-2, \
                ['powerlaw', 'spectralModel', 'spectrum'], ncoeffs)
        if np.sum(Fmask) > 0:
            Fpars = apars[Fmask]
            Fcov = np.diag(awidth[Fmask]**2)
            Fdim = np.sum(Fmask)
            sampler_F = ptmcmc.PTSampler(Fdim, \
                    likob.gibbs_Phi_loglikelihood_mar, \
                    likob.gibbs_Phi_logprior, \
                    cov=Fcov, outDir='./gibbs-chains/', \
                    verbose=False, nowrite=True, \
                    loglargs=[Fmask, apars], \
                    logpargs=[Fmask, apars])
        else:
            sampler_F = None

    # The gibbs coefficients are initially set to 2ns random each for numerical
    # stability (less for DM variations)
    # We keep track of this in both 'a', and 'likob.gibbs_current_a'.
    # TODO: use only gibbs_current_a.
    a = []
    for ii, psr in enumerate(likob.ptapsrs):
        a.append(likob.gibbs_get_initial_quadratics(ii))

    likob.gibbs_current_a = a

    # 1) Set the hyper-parameter structures for all pulsars to their initial
    # values
    likob.setPsrNoise(hpars)
    likob.constructPhiAndTheta(hpars, make_matrix=False, gibbs_expansion=True)

    # 2) Generate the frequency covariances: the matrix decompositions of the
    # correlated signals, per frequency... Scor_im_cf, Scor_im_inv
    if 'rednoise' in likob.gibbsmodel:
        likob.gibbs_construct_all_freqcov()

    # 3) Set the quadratic parameters in the full parameter array
    apars[ndim:] = np.hstack(a)

    # 4) Generate _all_ quadratic parameters here. The sub-residuals are
    # generated from all the respective (conditional-) likelihood functions
    #b = []
    for pp, psr in enumerate(likob.ptapsrs):
        a, bi, xi2  = likob.gibbs_sample_psr_quadratics(apars[:ndim], a, pp)
    #b.append(bi)
    apars[ndim:] = np.hstack(a)
    likob.gibbs_current_a = a

    # 5) Calculate the full likelihood:
    loglik[0] = likob.gibbs_full_loglikelihood(\
            apars, resetCorInv=False, which='all', pp=-1)

    # Time to do the actual sampling here.
    stepind = 0
    logpost[0] = loglik[0] + likob.mark4logprior(apars[:ndim])
    lp_prev = logpost[0]
    ll_prev = loglik[0]
    lp_cur = logpost[0]
    ll_cur = loglik[0]
    samples[stepind, :] = apars
    for step in range(1, steps):
        for pp, psr in enumerate(likob.ptapsrs):
            # Jump in the white noise parameters
            sampler = sampler_N[pp]
            if sampler is not None:
                if 'jitter' in likob.gibbsmodel:
                    # If jitter explicitly in gibbsmodel, then we don't look for
                    # it here. Instead, we look for efac/equad in the
                    # subresiduals that have everything else subtracted
                    # Subtraction is done in gibbs_psr_noise_mar.
                    # RvH: We can generate jitter here, too!
                    Nmask = likob.gibbs_get_signal_mask(pp, ['efac', 'equad'])
                else:
                    # RvH: where is the subtraction done
                    Nmask = likob.gibbs_get_signal_mask(pp, \
                            ['efac', 'equad', 'jitter', 'cequad'])
                psrNpars = apars[Nmask]

                # Arguments for the log-likelihood function (so we can do
                # subtraction)
                sampler.logl.args = [pp, Nmask, apars]
                sampler.logp.args = [pp, Nmask, apars]
                
                # The sampler always calculates the previous step as well, as it
                # should here
                sampler.sample(psrNpars, step+1, covUpdate=500, burn=2, maxIter=2*steps,
                    i0=step-1, thin=1)

                apars[Nmask] = sampler._chain[step,:]

                # RvH: does this set the Jvec as well?
                likob.setSinglePsrNoise(apars[:ndim], pp=pp)

            ##################################################################
            # We should not be saving the whole chain here. Can we checkpoint
            # the MCMC chains here? RvH -- 20141130
            ##################################################################

            if 'dm' in likob.gibbsmodel and sampler_D[pp] is not None:
                sampler = sampler_D[pp]
                Dmask = likob.gibbs_get_signal_mask(pp, ['dmpowerlaw', 'dmspectrum'])
                psrDpars = apars[Dmask]

                sampler.logl.args = [pp, Dmask, apars]
                sampler.logp.args = [pp, Dmask, apars]

                sampler.sample(psrDpars, step+1, covUpdate=500, burn=2, maxIter=2*steps,
                    i0=step-1, thin=1)

                if not np.all(apars[Dmask] == sampler._chain[step,:]):
                    # NOTE: RvH 20141129: What is the difference between these
                    #       two functions?
                    #
                    # Step accepted
                    #likob.gibbs_current_a, bi, xi2 = \
                    #        likob.gibbs_sample_psr_quadratics(apars[:ndim], \
                    #        likob.gibbs_current_a, pp, which='D')
                    likob.gibbs_current_a = likob.gibbs_sample_Theta_quadratics(\
                            likob.gibbs_current_a, pp)


                    apars[ndim:] = np.hstack(likob.gibbs_current_a)
                else:
                    # Not accepted. Put covariance back in place
                    likob.setTheta(apars[:ndim], pp=pp)

                apars[Dmask] = sampler._chain[step,:]


            if 'jitter' in likob.gibbsmodel and sampler_J[pp] is not None:
                sampler = sampler_J[pp]
                Jmask = likob.gibbs_get_signal_mask(pp, ['jitter'])
                psrJpars = apars[Dmask]

                sampler.logl.args = [pp, Jmask, apars]
                sampler.logp.args = [pp, Jmask, apars]

                sampler.sample(psrJpars, step+1, covUpdate=500, burn=2, maxIter=2*steps,
                    i0=step-1, thin=1)

                if np.all(apars[Jmask] != sampler._chain[step,:]):
                    # Step accepted
                    likob.gibbs_current_a, bi, xi2 = \
                            likob.gibbs_sample_psr_quadratics(apars[:ndim], \
                            likob.gibbs_current_a, pp, which='U')

                    apars[ndim:] = np.hstack(likob.gibbs_current_a)
                else:
                    likob.setPsrNoise(apars[:ndim])

                apars[Jmask] = sampler._chain[step,:]


        if 'rednoise' in likob.gibbsmodel and sampler_F is not None:
            sampler = sampler_F
            Fmask = likob.gibbs_get_signal_mask(-2, \
                    ['powerlaw', 'spectralModel', 'spectrum'])
            psrFpars = apars[Fmask]

            sampler.logl.args = [Fmask, apars]
            sampler.logp.args = [Fmask, apars]

            sampler.sample(psrFpars, step+1, covUpdate=500, burn=2, maxIter=2*steps,
                i0=step-1, thin=1)

            if not np.all(apars[Fmask] == sampler._chain[step,:]):
                # Step accepted
                likob.gibbs_current_a = \
                        likob.gibbs_sample_Phi_quadratics(\
                        likob.gibbs_current_a, sigma_precalc=True)

                apars[ndim:] = np.hstack(likob.gibbs_current_a)
            else:
                # Step not accepted. Put the covariances back to where they were
                # apars still has the old values here
                likob.setPhi(apars[:ndim])
                likob.gibbs_construct_all_freqcov()

            apars[Fmask] = sampler._chain[step,:]

        # Also do a timing-model re-sample, so we also get the parameters we had
        # not had before...
        # DO WE DO THE FULL MODEL, OR JUST THE COMPLEMENTARY MODEL????????????
        for pp, psr in enumerate(likob.ptapsrs):
            # NOTE: RvH 20141129: We definitely do not want to re-calculate all
            #       these quadratics. We can subtract plenty of 'm. Which ones?
            likob.gibbs_current_a, bi, xi2 = \
                    likob.gibbs_sample_psr_quadratics(apars[:ndim], \
                    likob.gibbs_current_a, pp, which='all')
            #likob.gibbs_current_a = likob.gibbs_sample_M_quadratics( \
            #        likob.gibbs_current_a, pp)

        # Finally update the actual log-likelihood/posterior for the full
        # model
        ll_prev = ll_cur
        lp_prev = lp_cur
        ll_cur = likob.gibbs_full_loglikelihood(apars)
        lp_cur = ll_cur + likob.mark4logprior(apars[:ndim])



        samples[stepind, :] = apars
        logpost[stepind] = lp_cur
        loglik[stepind] = ll_cur



        # All the blocks done for this step. Update the track records

        stepind += 1
        # Write to file if necessary
        if (stepind % dumpint == 0 or step == steps-1):
            nwrite = dumpint

            # Check how many samples we are writing
            if step == steps-1:
                nwrite = stepind

            # Open the file in append mode
            if not noWrite:
                chainfile = open(chainfilename, 'a+')
                for jj in range(nwrite):
                    chainfile.write('%.17e\t  %.17e\t  0.0\t' % (logpost[jj], \
                        loglik[jj]))
                    chainfile.write('\t'.join(["%.17e"%\
                            (samples[jj,kk]) for kk in range(ndim+ncoeffs)]))
                    chainfile.write('\n')
                chainfile.close()

                #chainfile_b = open(chainfilename_b, 'a+')
                #for jj in range(nwrite):
                #    chainfile_b.write('%.17e\t  %.17e\t  0.0\t' % (logpost[jj], \
                #        loglik[jj]))
                #    chainfile_b.write('\t'.join(["%.17e"%\
                #            (samples2[jj,kk]) for kk in range(ncoeffs)]))
                #    chainfile_b.write('\n')
                #chainfile_b.close()
            stepind = 0

        percent = (step * 100.0 / steps)
        sys.stdout.write("\rGibbs: %d%%" %percent)
        sys.stdout.flush()


    sys.stdout.write("\n")



###############################################################################
###############################################################################
########## Below here, it is all concerned with mark1 Gibbs sampler ###########
###############################################################################
###############################################################################



def gibbs_prepare_get_quadratic_cov(likob):
    """
    For use in the MCMC exploration of the Gibbs model, we need to have an
    initial guess about the stepsize of the quadratic parameters. So we need a
    typical width. Here we provide a simple estimate, based on a weighted
    least-squares fit

    @param likob:   The full likelihood object

    """
    # The width of the quadratic parameters, made per pulsar
    lz = []
    lcov = []
    for ii, psr in enumerate(likob.ptapsrs):
        Zx = np.dot(psr.Zmat.T, psr.residuals / psr.toaerrs**2)
        ZZ = np.dot(psr.Zmat.T, ( (1.0 / psr.toaerrs**2) * psr.Zmat.T).T)
        try:
            cf = sl.cho_factor(ZZ)

            ZZi = sl.cho_solve(cf, np.eye(ZZ.shape[0]))
        except np.linalg.LinAlgError:
            try:
                Qs,Rs = sl.qr(ZZ) 
                ZZi = sl.solve(Rs,  s.T)
            except np.linalg.LinAlgError:
                print "ERROR: QR cannot invert ZZ"
                raise

        lcov.append(ZZi)
        lz.append(np.dot(ZZi, Zx))

        """
        if 'design' in likob.gibbsmodel:
            nms = likob.npm[ii]
            lz[-1][:nms] = np.dot(psr.tmpConvi, lz[-1][:nms])

            lcov[-1][:nms, :nms] = np.dot(psr.tmpConvi, \
                    np.dot(lcov[-1][:nms, :nms], psr.tmpConvi.T))
        """

    return (lz, lcov)


def gibbs_construct_mode_covariance(likob, mode):
    """
    Given the mode number, this function calculates the full-pulsar correlation
    matrix of the red noise/GWs. Data is used from likob.Phivec, and likob.Scor,
    and likob.Svec. Remember that likob.Scor is contains only the correlations
    of one single (the last in the array) signal. So this only works for one
    correlated signal in the model (for now).
    """
    gw_pcdoubled = likob.Svec
    msk = likob.freqmask[:, mode]

    cov = likob.Scor[msk,:][:,msk] * gw_pcdoubled[mode]
    for ii, psr in likob.ptapsrs:
        ind = np.sum(msk[:ii])
        cov[ind, ind] += likob.Phivec[likob.nfs[ii]+mode]

    return cov
    


def gibbs_loglikelihood(likob, aparameters):
    """
    Within the Gibbs sampler, we would still like to have access to the
    loglikelihood value, even though it is not necessarily used for the sampling
    itself. This function evaluates the ll.

    This function does not set any of the noise/correlation auxiliaries. It
    assumes that has been done earlier in the Gibbs step. Also, it assumes the
    Gibbsresiduals have been set properly.

    NOTE:   the timing model parameters are assumed to be in the basis of Gcmat,
            not Mmat. This for numerical stability (really doesn't work
            otherwise). CONTINUE MAKING THIS!!!

    @param likob:       The full likelihood object
    @param aparameters: All the model parameters, including the quadratic pars
    @param coeffs:      List of all the Gibbs coefficients per pulsar

    @return:            The log-likelihood
    """

    ndim = likob.dimensions     # This does not include the quadratic parameters
    quadparind = ndim + 0       # Index of quadratic parameters

    # Now we also know the position of the hyper parameters
    allparameters = aparameters.copy()
    parameters = allparameters[:ndim]

    # Set the white noise
    likob.setPsrNoise(parameters)

    # Set the red noise / DM correction quantities. Use the Gibbs expansion to
    # build the per-frequency Phi-matrix
    likob.constructPhiAndTheta(parameters, make_matrix=False, \
            noise_vec=True, gibbs_expansion=True)

    if likob.haveDetSources:
        likob.updateDetSources(parameters)

    ksi = []        # Timing model parameters
    a = []          # Red noise / GWB Fourier modes
    d = []          # DM variation Fourier modes
    j = []          # Jitter/epochave residuals

    for ii, psr in enumerate(likob.ptapsrs):
        nzs = likob.npz[ii]
        nms = likob.npm[ii]
        findex = np.sum(likob.npf[:ii])
        nfs = likob.npf[ii]
        fdmindex = np.sum(likob.npfdm[:ii])
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        ntot = 0
        nqind = quadparind + 0
        if 'design' in likob.gibbsmodel:
            #allparameters[nqind:nqind+nms] = np.dot(psr.tmpConvi, allparameters[nqind:nqind+nms])
            ksi.append(allparameters[nqind:nqind+nms])
            ntot += nms
            nqind += nms
        if 'rednoise' in likob.gibbsmodel:
            a.append(allparameters[nqind:nqind+nfs])
            ntot += nfs
            nqind += nfs
        if 'dm' in likob.gibbsmodel:
            d.append(allparameters[nqind:nqind+nfdms])
            ntot += nfdms
            nqind += nfdms
        if 'jitter' in likob.gibbsmodel:
            j.append(allparameters[nqind:nqind+npus])
            ntot += npus
            nqind += npus

        # Calculate the quadratic parameter subtracted residuals
        gibbscoefficients = allparameters[quadparind:quadparind+ntot]
        psr.gibbssubresiduals = np.dot(psr.Zmat, gibbscoefficients)
        psr.gibbsresiduals = psr.detresiduals - psr.gibbssubresiduals

        quadparind += ntot

    # Now evaluate the various quadratic forms
    xi2 = 0
    ldet = 0
    for ii, psr in enumerate(likob.ptapsrs):
        # The quadratic form of the residuals
        #xi2 += np.sum(psr.gibbsresiduals ** 2 / psr.Nvec)
        #ldet += np.sum(np.log(psr.Nvec))
        if 'jitter' in likob.gibbsmodel:
            xi2 += np.sum(psr.gibbsresiduals ** 2 / psr.Nvec)
            ldet += np.sum(np.log(psr.Nvec))
            xi2 += np.sum(j[ii] ** 2 / psr.Jvec)
            ldet += np.sum(np.log(psr.Jvec))
        else:
            jldet, jxi2 = cython_block_shermor_1D(psr.gibbsresiduals, \
                    psr.Nvec, psr.Jvec, psr.Uinds)
            xi2 += jxi2
            ldet += jldet

    # Quadratic form of red noise, done for full array
    if 'rednoise' in likob.gibbsmodel:
        # Do some fancy stuff here per frequency
        if len(likob.ptapsrs) > 1:
            # Loop over all frequencies
            for ii in range(0, len(likob.Svec), 2):
                msk = likob.freqmask[:, ii]

                # The covariance between sin/cos modes is identical
                cov = gibbs_construct_mode_covariance(likob, ii)
                cf = sl.cho_factor(cov)

                # Cosine mode
                bc = likob.freqb[msk, ii]
                Lx = sl.cho_solve(cf, bc)
                xi2 += np.sum(Lx**2)
                ldet += 2*np.sum(np.log(np.diag(cf[0])))

                # Sine mode
                bs = likob.freqb[msk, ii+1]
                Lx = sl.cho_solve(cf, bs)
                xi2 += np.sum(Lx**2)
                ldet += 2*np.sum(np.log(np.diag(cf[0])))
        else:
            # Single pulsar. Just combine the correlated and noise frequencies
            pcd = likob.Phivec + likob.Svec
            xi2 += np.sum(np.hstack(a)**2 / pcd)
            ldet += np.sum(np.log(pcd))

    # Quadratic form of DM variations, for full array
    if 'dm' in likob.gibbsmodel:
        xi2 += np.sum(np.hstack(d)**2 / likob.Thetavec)
        ldet += np.sum(np.log(likob.Thetavec))

    #print "xi2:", xi2, "  ldet:", ldet, "   const:", np.sum(likob.npobs)*np.log(2*np.pi)

    return -0.5*np.sum(likob.npobs)*np.log(2*np.pi) - 0.5*xi2 - 0.5*ldet

def gibbs_logprior(likob, allparameters):
    """
    Calculate the prior on the model parameters(from signals), not from the
    prior of the quadratic (Gibbs) parameters.
    """
    ndim = likob.dimensions     # This does not include the quadratic parameters
    parameters = allparameters[:ndim]

    logpr = likob.mark4logprior(parameters)

    return logpr

def gibbs_logposterior(likob, allparameters):
    """
    Include the prior on the model parameters (from signals). Do not include a
    prior for the quadratic (Gibbs) parameters.
    """

    return  gibbs_logprior(likob, allparameters) + \
            gibbs_loglikelihood(likob, allparameters)

class gibbs_likob_class(object):
    """
    Tiny class, that allow the Gibbs likelihood to be sampled.
    """

    def __init__(self, likob):
        self.likob = likob

    def logprior(self, pars):
        return gibbs_logprior(self.likob, pars)

    def loglikelihood(self, pars):
        return gibbs_loglikelihood(self.likob, pars)

    def logposterior(self, pars):
        return self.logprior(pars) + self.loglikelihood(pars)


class pulsarNoiseLL(object):
    """
    This class represents the likelihood function in the block with
    white-noise-only parameters.
    """

    def __init__(self, residuals, psrindex, maskJvec=None):
        """
        @param residuals:   Initialise the residuals we'll work with
        @param psrindex:    Index of the pulsar this noise applies to
        @param maskJvec:    Selection mask of the jitter Jvec
        """
        self.vNvec = []                             # Mask residuals (varying)
        self.fNvec = []                             # Mask residuals (fixed)
        self.vis_efac = []                          # Is it an efac? (varying)
        self.fis_efac = []                          # Is it an equad (fixed)

        self.residuals = residuals                  # The residuals
        self.Nvec = np.zeros(len(residuals))        # Full noise vector
        self.pmin = np.zeros(0)                     # Minimum of prior domain
        self.pmax = np.zeros(0)                     # Maximum of prior domain
        self.pstart = np.zeros(0)                   # Start position
        self.pwidth = np.zeros(0)                   # Initial step-size
        self.fval = np.zeros(0)                     # Current value (non-varying)
        self.pindex = np.zeros(0, dtype=np.int)     # Index of parameters

        self.psrindex = psrindex                    # Inde xof pulsar
        self.maskJvec = maskJvec                    # Selection mask Jvec

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance
                                    # updates


    def addSignal(self, Nvec, is_efac, pmin, pmax, pstart, pwidth, index, fixed=False):
        """
        Add a efac/equad signal to this model
        @param Nvec:    To which residuals does this signal apply?
        @param is_efac: Is this signal an efac, or an equad?
        @param pmin:    Minimum of the prior domain
        @param pmax:    Maximum of the prior domain
        @param fixed:   Whether or not we vary this parameter
        """
        if not fixed:
            self.vNvec.append(Nvec)
            self.vis_efac.append(is_efac)
            self.pmin = np.append(self.pmin, [pmin])
            self.pmax = np.append(self.pmax, [pmax])
            self.pstart = np.append(self.pstart, [pstart])
            self.pwidth = np.append(self.pwidth, [pwidth])
            self.pindex = np.append(self.pindex, index)
        else:
            self.fNvec.append(Nvec)
            self.fis_efac.append(is_efac)
            self.fval = np.append(self.fval, pstart)

    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate

    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :].copy()

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin, self.pmax, self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx

    def setNewData(self, residuals):
        self.residuals = residuals

    def loglikelihood(self, parameters):
        """
        @param parameters:  the vector with model parameters
        """
        self.Nvec[:] = 0
        for ii, par in enumerate(parameters):
            if self.vis_efac[ii]:
                self.Nvec += par**2 * self.vNvec[ii]
            else:
                self.Nvec += 10**(2*par) * self.vNvec[ii]

        for ii, par in enumerate(self.fval):
            if self.fis_efac[ii]:
                self.Nvec += par**2 * self.fNvec[ii]
            else:
                self.Nvec += 10**(2*par) * self.fNvec[ii]

        return -0.5 * np.sum(self.residuals**2 / self.Nvec) - \
                0.5 * np.sum(np.log(self.Nvec))

    def logprior(self, parameters):
        bok = -np.inf
        #bok = -1e99
        if np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax):
            bok = 0

        return bok

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        if len(self.vNvec) != len(self.vis_efac):
            raise ValueError("dimensions not set correctly")

        return len(self.vNvec)


class pulsarJNoiseLL(object):
    """
    This class represents the likelihood function in the block with
    white-noise-only parameters.
    """

    def __init__(self, residuals, Uinds, psrindex):
        """
        @param residuals:   Initialise the residuals we'll work with
        @param Uinds:       Replaces Umat, now start/end indices of Jitter
        @param psrindex:    Index of the pulsar this noise applies to
        """
        # NOTE: vis_efac/fis_efac is changed to an ID, with the values
        #       0 (efac), 1 (equad), 2 (cequad/jitter)
        #       changed is_efac to which='efac'/'equad'/'cequad'or'jitter'
        self.vNvec = []                             # Mask residuals (varying)
        self.fNvec = []                             # Mask residuals (fixed)

        #self.vis_efac = []                          # Is it an efac? (varying)
        #self.fis_efac = []                          # Is it an equad (fixed)
        self.vsig_id = []                           # Type of signal
        self.fsig_id = []                           # Type of signal

        self.residuals = residuals                  # The residuals
        self.Uinds = Uinds                          # Exploder indices (of epochs)
        self.Nvec = np.zeros(len(residuals))        # Full noise vector
        self.Jvec = np.zeros(len(Uinds))            # Full jitter vector

        self.pmin = np.zeros(0)                     # Minimum of prior domain
        self.pmax = np.zeros(0)                     # Maximum of prior domain
        self.pstart = np.zeros(0)                   # Start position
        self.pwidth = np.zeros(0)                   # Initial step-size
        self.fval = np.zeros(0)                     # Current value (non-varying)
        self.pindex = np.zeros(0, dtype=np.int)     # Index of parameters

        self.psrindex = psrindex                    # Inde xof pulsar

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance
                                    # updates


    def addSignal(self, Nvec, which, pmin, pmax, pstart, pwidth, index, \
            fixed=False):
        """
        Add a efac/equad signal to this model
        @param Nvec:    To which residuals does this signal apply?
                        ex: (0, 0, toaerr**2, toaerr**2, 0, 0, 0)
                        For 'jitter', which epochs does it apply to
        @param which:   'efac'/'equad'/'jitter'or'cequad'
        @param pmin:    Minimum of the prior domain
        @param pmax:    Maximum of the prior domain
        @param pstart:  Start position of parameter
        @param pwidth:  Typical step-size
        @param index:   Index of parameter in the full parameter array
        @param fixed:   Whether or not we vary this parameter
        """
        if not fixed:
            self.vsig_id.append(which)
            self.pmin = np.append(self.pmin, [pmin])
            self.pmax = np.append(self.pmax, [pmax])
            self.pstart = np.append(self.pstart, [pstart])
            self.pwidth = np.append(self.pwidth, [pwidth])
            self.pindex = np.append(self.pindex, index)
            self.vNvec.append(Nvec)
        else:
            self.fsig_id.append(which)
            self.fval = np.append(self.fval, pstart)
            self.fNvec.append(Nvec)


    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate

    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :].copy()

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin, self.pmax, self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx

    def setNewData(self, residuals):
        self.residuals = residuals

    def loglikelihood(self, parameters):
        """
        @param parameters:  the vector with model parameters
        """
        self.Nvec[:] = 0.0
        self.Jvec[:] = 0.0
        for ii, par in enumerate(parameters):
            if self.vsig_id[ii] == 'efac':
                self.Nvec += par**2 * self.vNvec[ii]
            elif self.vsig_id[ii] == 'equad':
                self.Nvec += 10**(2*par) * self.vNvec[ii]
            elif self.vsig_id[ii] in ['jitter', 'cequad']:
                self.Jvec += 10**(2*par) * self.vNvec[ii]

        for ii, par in enumerate(self.fval):
            if self.fsig_id[ii] == 'efac':
                self.Nvec += par**2 * self.fNvec[ii]
            elif self.fsig_id[ii] == 'equad':
                self.Nvec += 10**(2*par) * self.fNvec[ii]
            elif self.fsig_id[ii] in ['jitter', 'cequad']:
                self.Jvec += 10**(2*par) * self.fNvec[ii]

        # From the jitter extension module
        Jldet, xNx = cython_block_shermor_1D(self.residuals, \
                self.Nvec, self.Jvec, self.Uinds)

        return -0.5 * xNx - 0.5 * Jldet - \
                len(self.residuals)*np.log(2*np.pi)

    def logprior(self, parameters):
        bok = -np.inf
        #bok = -1e99
        if np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax):
            bok = 0

        return bok

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        """
        How many dimensions are in this conditional likelihood
        """
        if len(self.vNvec) != len(self.vsig_id):
            raise ValueError("dimensions not set correctly")

        return len(self.vNvec)



class pulsarPSDLL(object):
    """
    Like pulsarNoiseLL, but for the power spectrum coefficients. Expects the
    matrix to be diagonal. Is always a function of amplitude and spectral index
    """

    def __init__(self, a, freqs, Tmax, pmin, pmax, pstart, pwidth, index, \
            psrindex, gindices, bvary):
        """
        @param a:           The Fourier components of the signal
        @param freqs:       The frequencies of the signal
        @param pmin:        Minimum value of the parameters prior domain
        @param pmax:        Maximum value of the parameters prior domain
        @param pwidth:      Initial step-size of this parameter
        @param index:       First index in the parameter array of this signal
        @param psrindex:    Index of the pulsar this applies to
        @param gindices:    The indices of the relevant Gibbs-parameters
        @param bvary:       Which parameters are actually varying
        """

        self.a = a
        self.freqs = freqs
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.pindex = index
        self.psrindex = psrindex
        self.gindices = gindices
        self.bvary = bvary
        self.Tmax = Tmax

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance

    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth[self.bvary]**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate


    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """
        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :].copy()

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin[self.bvary], self.pmax[self.bvary], self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx



    def setNewData(self, a):
        self.a = a

    def loglikelihood(self, parameters):
        pars = self.pstart.copy()
        pars[self.bvary] = parameters

        freqpy = self.freqs * pic_spy
        pcdoubled = ((10**(2*pars[0])) * pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-pars[1])

        return -0.5 * np.sum(self.a**2 / pcdoubled) - 0.5*np.sum(np.log(pcdoubled))

    def logprior(self, parameters):
        bok = -np.inf
        #bok = -1e99
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        return np.sum(self.bvary)



class corrPSDLL(object):
    """
    Like pulsarPSDLL, but now for correlated signals.
    """

    def __init__(self, b, freqs, Tmax, pmin, pmax, pstart, pwidth, index, \
            psrindex, gindices, bvary):
        """
        @param b:           The Fourier components of the signal
                            (list for pulsars)
        @param freqs:       The frequencies of the signal
        @param pmin:        Minimum value of the parameters prior domain
        @param pmax:        Maximum value of the parameters prior domain
        @param pwidth:      Initial step-size of this parameter
        @param index:       First index in the parameter array of this signal
        @param psrindex:    Index of the pulsar this applies to
        @param gindices:    The indices of the relevant Gibbs-parameters
        @param bvary:       Which parameters are actually varying
        """

        self.b = b
        self.freqs = freqs
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.pindex = index
        self.psrindex = psrindex
        self.gindices = gindices
        self.bvary = bvary
        self.Tmax = Tmax

        self.allPsrSame = False
        self.Scor_inv = None
        self.Scor_ldet = None

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance


    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth[self.bvary]**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate

    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :].copy()

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin[self.bvary], self.pmax[self.bvary], self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx


    def setNewData(self, b, Scor_inv, Scor_ldet):
        """
        Set new data at the beginning of each small MCMC chain

        @param b:           New Gibbs modes
        @param Scor_inv:    New pulsar correlation inverse
        @param Scor_ldet:   New pulsar correlation log-det
        """
        self.b = b
        self.Scor_inv = Scor_inv
        self.Scor_ldet = Scor_ldet

        self.numPsrFreqs = []
        self.freqmask = np.zeros((len(self.b), len(self.freqs)), dtype=np.bool)
        self.freqb = np.zeros((len(self.b), len(self.freqs)))

        # Make masks that show which pulsar has how many frequencies
        for ii in range(len(self.b)):
            self.numPsrFreqs.append(len(self.b[ii]))
            self.freqmask[ii,:len(self.b[ii])] = True

        for ii in range(len(self.b)):
            for jj in range(len(self.freqs)):
                if self.freqmask[ii, jj]:
                    self.freqb[ii, jj] = self.b[ii][jj]

        self.allPsrSame = np.all(self.freqmask)

    def loglikelihood(self, parameters):
        pars = self.pstart.copy()
        pars[self.bvary] = parameters

        freqpy = self.freqs * pic_spy
        pcdoubled = ((10**(2*pars[0])) * pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-pars[1])

        xi2 = 0
        ldet = 0

        if self.allPsrSame and False:
            nfreqs = len(pcdoubled)
            dotprod = np.dot(self.freqb[:,0], np.dot(self.Scor_inv, \
                    self.freqb[:,0]))
            xi2 += np.sum(dotprod / pcdoubled)
            ldet += nfreqs * self.Scor_ldet + np.sum(np.log(pcdoubled))
        else:
            # For every frequency, do an H&D matrix inverse inner-product
            for ii, pc in enumerate(pcdoubled):
                msk = self.freqmask[:, ii]
                xi2 += np.dot(self.freqb[msk,ii], np.dot(\
                        self.Scor_inv[msk,:][:,msk], self.freqb[msk,ii])) / pc
                ldet += self.Scor_ldet + np.sum(msk)*np.log(pc)

        return -0.5 * xi2 - 0.5 * ldet


    def logprior(self, parameters):
        bok = -np.inf
        #bok = -1e99
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        return np.sum(self.bvary)




class implicitPSDLL(object):
    """
    Like pulsarPSDLL, but now for all power-law PSD signals.

    This likelihood class assumes that we have 3 * npsr + 3 parameters, and it
    will fail if this is not True. This functionality will be expanded in the
    future
    """

    def __init__(self, b, freqs, Tmax, pmin, pmax, pstart, pwidth, pmask, \
            gindices, bvary, likob):
        """
        @param b:           The Fourier components of the signal
                            (list for pulsars)
        @param freqs:       The frequencies of the signal, single array,
                            match between pulsars. Pulsars allowed to have fewer
                            modes
        @param pmin:        Minimum value of the parameters prior domain
        @param pmax:        Maximum value of the parameters prior domain
        @param pwidth:      Initial step-size of this parameter
        @param pmask:       Boolean mask indicating the relevant parameters
        @param gindices:    The indices of the relevant Gibbs-parameters
        @param bvary:       Which parameters are actually varying
        @param likob:       The full likelihood object (need it for comm)
        """

        self.b = b
        self.freqs = freqs
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.pmask = pmask
        self.gindices = gindices
        self.bvary = bvary
        self.Tmax = Tmax
        self.likob = likob

        self.allPsrSame = False
        self.Scor = None
        self.psr_pcdoubled = None
        self.gw_pcdoubled = None

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance


    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth[self.bvary]**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate

    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :].copy()

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin[self.bvary], self.pmax[self.bvary], self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx


    def setNewData(self, b, Scor):
        """
        Set new data at the beginning of each small MCMC chain

        @param b:           New Gibbs modes
        @param Scor:        New pulsar correlation matrix of corr sig
        """
        self.b = b
        self.Scor = Scor

        self.numPsrFreqs = []
        self.freqmask = np.zeros((len(self.b), len(self.freqs)), dtype=np.bool)
        self.freqb = np.zeros((len(self.b), len(self.freqs)))

        # Make masks that show which pulsar has how many frequencies
        for ii in range(len(self.b)):
            self.numPsrFreqs.append(len(self.b[ii]))
            self.freqmask[ii,:len(self.b[ii])] = True

        for ii in range(len(self.b)):
            for jj in range(len(self.freqs)):
                if self.freqmask[ii, jj]:
                    self.freqb[ii, jj] = self.b[ii][jj]

        self.allPsrSame = np.all(self.freqmask)


    def calcCoeffs(self, parameters):
        """
        Pre-calculate the PSD coefficients
        """
        pars = self.pstart.copy()
        pars[self.bvary] = parameters

        freqpy = self.freqs * pic_spy
        self.gw_pcdoubled = ((10**(2*pars[-3])) * pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-pars[-2])
        psr_pcdoubled = []
        for ii in range(int((len(pars)-3)/3)):
            psr_pcdoubled.append(
                    ((10**(2*pars[3*ii])) * pic_spy**3 / \
                            (12*np.pi*np.pi * self.Tmax)) * \
                            freqpy ** (-pars[3*ii+1]))
        self.psr_pcdoubled = np.array(psr_pcdoubled)


    def loglikelihood(self, parameters):
        xi2 = 0
        ldet = 0
        cov = None

        self.calcCoeffs(parameters)

        self.likob.Scor_im = []
        self.likob.Scor_im_cf = []
        for ii, gw_pc in enumerate(self.gw_pcdoubled):
            if ii % 2 == 0:
                # Re-evaluate the cholesky decomposition of the covariance
                # matrix. (same for all sine & cosine terms)
                msk = self.freqmask[:, ii]
                psrcoeff = self.psr_pcdoubled[msk, ii]

                cov = self.Scor[msk,:][:,msk] * self.gw_pcdoubled[ii] + \
                        np.diag(psrcoeff)

                cov_cf = sl.cho_factor(cov)

            # Append twice, not once
            self.likob.Scor_im.append(cov)
            self.likob.Scor_im_cf.append(cov_cf)

            x = self.freqb[msk, ii]
            ldet += 2*np.sum(np.log(np.diag(cov_cf[0])))
            xi2 += np.dot(x, sl.cho_solve(cov_cf, x))

        return -0.5 * xi2 - 0.5 * ldet


    def logprior(self, parameters):
        bok = -np.inf
        #bok = -1e99
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        return np.sum(self.bvary)



class pulsarDetLL(object):
    """
    Like pulsarNoiseLL, but for all other deterministic sources, like GW BWM
    """
    def __init__(self, likob, allpars, mask, pmin, pmax, pstart, pwidth, bvary):
        """
        @param likob:   The likelihood object, containing all the pulsars etc.
        @param allpars: Array with _all_ parameters, not just the det-sources
        @param mask:    Boolean mask that selects the det-source parameters
        @param pmin:    Minimum bound of all det-source parameters
        @param pmax:    Maximum bound of all det-source parameters
        @param pstart:  Start-parameter of all ''
        @param pwidth:  Width-parameter of all ''
        @param bvary:   Whether or not we vary the parameters (also with mask?)
        """

        self.likob = likob
        self.allpars = allpars.copy()
        self.mask = mask
        self.pmin = pmin
        self.pmax = pmax
        self.pstart = pstart
        self.pwidth = pwidth
        self.bvary = bvary

        self.sampler = None                         # The PTMCMC sampler
        self.singleChain = None     # How a long a ginle run is
        self.fullChain = None       # Maximum of total chain
        self.curStep = 0            # Current iteration
        self.covUpdate = 400        # Number of iterations between AM covariance

    def initSampler(self, singleChain=20, fullChain=20000, covUpdate=400):
        """
        Initialise the PTMCMC sampler for future repeated use

        @param singleChain:     Lenth of a single small MCMC chain
        @param fullChain:       Lenth of full chain that is used for
                                cov-estimates
        @param covUpdate:       Number of iterations before cov updates
                                (should be multiple of singleChain)
        """
        ndim = self.dimensions()
        cov = np.diag(self.pwidth[self.bvary]**2)
        self.sampler = ptmcmc.PTSampler(ndim, self.loglikelihood, \
                self.logprior, cov=cov, outDir='./gibbs-chains/', \
                verbose=False, nowrite=True)

        self.singleChain = singleChain
        self.fullChain = fullChain
        self.curStep = 0
        self.covUpdate = covUpdate


    def runSampler(self, p0):
        """
        Run the MCMC sampler, starting from position p0, for self.singleChain
        steps. Note: the covariance update length is also used as a length for
        the differential evolution burn-in. There is no chain-thinning

        @param p0:  Current value in parameter space

        @return:    New/latest value in parameter space
        """

        # Run the sampler for a small minichain
        self.sampler.sample(p0, self.curStep+self.singleChain, \
                maxIter=self.fullChain, covUpdate=self.covUpdate, \
                burn=self.covUpdate, i0=self.curStep, thin=1)

        self.curStep += self.singleChain
        retPos = self.sampler._chain[self.curStep-1, :].copy()

        # Subtract the mean off of the just-created samples. And because
        # covUpdate is supposed to be a multiple of singleChain, this will not
        # mess up the Adaptive Metropolis shizzle
        self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] = \
                self.sampler._chain[self.curStep-self.singleChain:self.curStep, :] - \
                np.mean(self.sampler._chain[self.curStep-self.singleChain:self.curStep, :])

        # Check whether we're almost at the end of the chain
        if self.fullChain - self.curStep <= self.covUpdate:
            midStep = int(self.fullChain / 2)

            # Copy the end half of the chain to the beginning
            self.sampler._lnprob[:self.curStep-midStep] = \
                    self.sampler._lnprob[midStep:self.curStep]
            self.sampler._lnlike[:self.curStep-midStep] = \
                    self.sampler._lnlike[midStep:self.curStep]
            self.sampler._chain[:self.curStep-midStep, :] = \
                    self.sampler._chain[midStep:self.curStep, :]
            self.sampler._AMbuffer[:self.curStep-midStep, :] = \
                    self.sampler._AMbuffer[midStep:self.curStep, :]
            self.sampler._DEbuffer = self.sampler._AMbuffer[0:self.covUpdate]

            # We are now at half the chain-length again
            self.curStep = self.curStep - midStep

        return retPos

    def runPSO(self):
        """
        Run a particle swarm optimiser on the posterior, and return the optimum
        """
        ndim = self.dimensions()
        nparticles = int(ndim**2/2) + 5*ndim
        maxiterations = 500

        swarm = Swarm(nparticles, self.pmin[self.bvary], self.pmax[self.bvary], self.logposterior)

        for ii in range(maxiterations):
            swarm.iterateOnce()

            if np.all(swarm.Rhat() < 1.02):
                # Convergence critirion satisfied!
                # print "N converged in {0}".format(ii)
                break

        return swarm.bestx



    def loglikelihood(self, parameters):
        """
        Calculate the conditional likelihood, as a function of the deterministic
        model parameters. This is an all-pulsar likelihood
        """
        self.allpars[self.mask] = parameters
        self.likob.updateDetSources(self.allpars)

        xi2 = 0
        ldet = 0

        for psr in enumerate(self.likob.ptapsrs):
            xi2 += np.sum((psr.detresiduals-psr.gibbssubresiduals)**2/psr.Nvec)
            ldet += np.sum(np.log(psr.Nvec))

        return -0.5*xi2 - 0.5*ldet


    def logprior(self, parameters):
        """
        Only return 0 when the parameters are within the prior domain
        """
        bok = -np.inf
        #bok = -1e99
        if np.all(self.pmin[self.bvary] <= parameters) and \
                np.all(parameters <= self.pmax[self.bvary]):
            bok = 0

        return bok

    def logposterior(self, parameters):
        return self.logprior(parameters) + self.loglikelihood(parameters)

    def dimensions(self):
        return np.sum(self.bvary)


def gibbs_prepare_loglik_N(likob, curpars):
    """
    Prepares the likelihood objects for white noise of all pulsars

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """

    loglik_N = []
    for ii, psr in enumerate(likob.ptapsrs):
        tempres = np.zeros(len(psr.residuals))
        loglik_N.append(pulsarNoiseLL(tempres, ii))
        pnl = loglik_N[-1]

        # Add the signals
        for ss, signal in enumerate(likob.ptasignals):
            pstart = curpars[signal['parindex']]
            if signal['pulsarind'] == ii and signal['stype'] == 'efac':
                # We have a winner: add this signal
                pnl.addSignal(signal['Nvec'], True, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        signal['parindex'], fixed=(not signal['bvary'][0]))
            elif signal['pulsarind'] == ii and signal['stype'] == 'equad':
                # We have a winner: add this signal
                pnl.addSignal(signal['Nvec'], False, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        signal['parindex'], fixed=(not signal['bvary'][0]))


        temp = pnl.loglikelihood(pnl.pstart)
        ndim = pnl.dimensions()

        pnl.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                covUpdate=ndim*400)

    return loglik_N

def gibbs_sample_loglik_N(likob, curpars, loglik_N, ml=False):
    """
    @param likob:       the full likelihood object
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_N:    List of prepared likelihood/samplers for the noise
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pnl in loglik_N:
        # Update the sampler with the new Gibbs residuals
        psr = likob.ptapsrs[pnl.psrindex]
        pnl.setNewData(psr.gibbsresiduals)

        if ml:
            newpars[pnl.pindex] = pnl.runPSO()
        else:
            p0 = newpars[pnl.pindex]
            newpars[pnl.pindex] = pnl.runSampler(p0)

    return newpars


def gibbs_prepare_loglik_J(likob, curpars):
    """
    Prepares the likelihood objects for correlated equad of all pulsars

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """
    loglik_J = []

    for ii, psr in enumerate(likob.ptapsrs):
        nzs = likob.npz[ii]
        nms = likob.npm[ii]
        nfs = likob.npf[ii]
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        # Figure out where to start counting
        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms
        if 'rednoise' in likob.gibbsmodel:
            ntot += nfs
        if 'dm' in likob.gibbsmodel:
            ntot += nfdms


        # Add the signals
        for ss, signal in enumerate(likob.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'jitter' and \
                    signal['bvary'][0] == True:
                # We have a jitter parameter
                inds = ntot
                inde = ntot+npus

                # To which avetoas does this 'jitter' apply?
                select = np.array(signal['Jvec'], dtype=np.bool)
                selall = np.ones(np.sum(select))

                res = np.zeros(np.sum(select))
                pstart = curpars[signal['parindex']]

                # Set up the conditional log-likelihood
                loglik_J.append(pulsarNoiseLL(res, ii, maskJvec=signal['Jvec']))
                pnl = loglik_J[-1]
                pnl.addSignal(selall, False, signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        signal['parindex'], fixed=(not signal['bvary'][0]))

                # Prepare the sampler
                temp = pnl.loglikelihood(pnl.pstart)

                ndim = pnl.dimensions()

                pnl.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                        covUpdate=ndim*400)

    return loglik_J

def gibbs_sample_loglik_J(likob, a, curpars, loglik_J, ml=False):
    """
    @param likob:       the full likelihood object
    @param a:       list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_J:    List of prepared likelihood/samplers for the correlated noise
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pnl in loglik_J:
        # Update the sampler with the new Gibbs residuals
        ii = pnl.psrindex
        psr = likob.ptapsrs[ii]

        # Indices
        nms = likob.npm[ii]
        nfs = likob.npf[ii]
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        # Figure out where the indices start
        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms
        if 'rednoise' in likob.gibbsmodel:
            ntot += nfs
        if 'dm' in likob.gibbsmodel:
            ntot += nfdms


        # Gibbs parameter selectors
        inds = ntot
        inde = ntot+npus

        # Set the new 'residuals'
        select = np.array(pnl.maskJvec, dtype=np.bool)
        res = a[ii][inds:inde][select]
        pnl.setNewData(res)

        if ml:
            newpars[pnl.pindex] = pnl.runPSO()
        else:
            p0 = newpars[pnl.pindex]
            newpars[pnl.pindex] = pnl.runSampler(p0)

    return newpars

def gibbs_prepare_loglik_NJ(likob, curpars):
    """
    Prepares the likelihood objects for white noise of all pulsars, where the
    white noise includes the jitter/correlated equads

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """

    loglik_NJ = []
    for ii, psr in enumerate(likob.ptapsrs):
        tempres = np.zeros(len(psr.residuals))
        loglik_NJ.append(pulsarJNoiseLL(tempres, psr.Uinds, ii))
        pnl = loglik_NJ[-1]

        # Add the signals
        for ss, signal in enumerate(likob.ptasignals):
            pstart = curpars[signal['parindex']]
            if signal['pulsarind'] == ii and \
                    signal['stype'] in ['efac', 'equad', 'cequad', 'jitter']:
                if signal['stype'] in ['cequad', 'jitter']:
                    nv = signal['Jvec']
                else:
                    nv = signal['Nvec']


                pnl.addSignal(nv, signal['stype'], signal['pmin'][0], \
                        signal['pmax'][0], pstart, signal['pwidth'][0], \
                        signal['parindex'], fixed=(not signal['bvary'][0]))

        temp = pnl.loglikelihood(pnl.pstart)
        ndim = pnl.dimensions()

        pnl.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                covUpdate=ndim*400)

    return loglik_NJ

def gibbs_sample_loglik_NJ(likob, curpars, loglik_NJ, ml=False):
    """
    @param likob:       the full likelihood object
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_NJ:   List of prepared likelihood/samplers for the noise
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pnl in loglik_NJ:
        # Update the sampler with the new Gibbs residuals
        psr = likob.ptapsrs[pnl.psrindex]
        pnl.setNewData(psr.gibbsresiduals)

        if ml:
            newpars[pnl.pindex] = pnl.runPSO()
        else:
            p0 = newpars[pnl.pindex]
            newpars[pnl.pindex] = pnl.runSampler(p0)

    return newpars

def gibbs_update_ecor_NJ(likob, pars, a, b, joinNJ=False):
    """
    After the NJ conditional, we need to draw new jitter/ecor quadratic
    parameters. Once we have those, we can update both gibbsresiduals, and
    gibbssubresiduals

    @param likob:   The full likelihood object
    @param pars:    Array of hyper-parameters
    @param a:       List of the quadratic parameters (design true)
    @param b:       List of the quadratic parameters (design orthogonalized)
    @param joinNJ:  Whether or not we actually need to do this
    """
    if joinNJ:
        # Update Jvec
        likob.setPsrNoise(pars)

        # Have to update both gibbsresiduals, and gibbssubresiduals
        for pp, psr in enumerate(likob.ptapsrs):
            Jldet, xNx, eat = cython_shermor_draw_ecor(psr.gibbsresiduals, \
                    psr.Nvec, psr.Jvec, psr.Uinds)
            eat = python_draw_ecor(psr.gibbsresiduals, \
                    psr.Nvec, psr.Jvec, psr.Uinds)

            psr.gibbssubresiduals, psr.gibbsresiduals = \
                    cython_update_ea_residuals(psr.gibbsresiduals, \
                        psr.gibbssubresiduals, eat, psr.Uinds)

            if 'jitter' in likob.gibbsmodel:
                # We need to update the quadratic parameters
                a[pp][psr.Zmask_U] = eat
                b[pp][psr.Zmask_U] = eat
    else:
        # We don't sample the ecorr parameters here
        pass

    likob.gibbs_current_a = b
    return a, b


def gibbs_prepare_loglik_Phi(likob, curpars):
    """
    Prepares the likelihood objects for the power-spectra of all pulsars

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """

    loglik_Phi = []
    for ii, psr in enumerate(likob.ptapsrs):
        for ss, signal in enumerate(likob.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'powerlaw' and \
                    signal['corr'] == 'single':
                bvary = signal['bvary']
                pindex = signal['parindex']
                pstart = signal['pstart']
                pmin = signal['pmin']
                pmax = signal['pmax']
                pwidth = signal['pwidth']
                Tmax = signal['Tmax']
                nms = likob.npm[ii]
                nfs = likob.npf[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms

                ndim = np.sum(bvary)
                if ndim > 0:
                    pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                    loglik_Phi.append(pulsarPSDLL(np.zeros(nfs), \
                            psr.Ffreqs, Tmax, pmin, pmax, pstart, pwidth, \
                            pindex, ii, np.arange(ntot, ntot+nfs), bvary))
                    psd = loglik_Phi[-1]


                    psd.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                            covUpdate=ndim*400)

            elif signal['pulsarind'] == ii and signal['stype'] == 'dmpowerlaw':
                bvary = signal['bvary']
                pindex = signal['parindex']
                pstart = signal['pstart']
                pmin = signal['pmin']
                pmax = signal['pmax']
                pwidth = signal['pwidth']
                Tmax = signal['Tmax']
                nms = likob.npm[ii]
                nfs = likob.npf[ii]
                nfdms = likob.npfdm[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms
                if 'rednoise' in likob.gibbsmodel:
                    ntot += nfs

                ndim = np.sum(bvary)
                if ndim > 0:
                    pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                    loglik_Phi.append(pulsarPSDLL(np.zeros(nfdms), \
                        psr.Fdmfreqs, Tmax, pmin, pmax, pstart, pwidth, pindex, \
                                ii, np.arange(ntot, ntot+nfdms), bvary))
                    psd = loglik_Phi[-1]

                    psd.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                            covUpdate=ndim*400)

    return loglik_Phi

def gibbs_sample_loglik_Phi_an(likob, a, curpars, ml=False):
    """
    Sample the Phi-loglikelihood conditional for the analytic parameters. The
    numerical (MCMC) parameters are done elsewhere

    @param likob:   the full likelihood object
    @param a:       list of arrays with all the Gibbs-only parameters
    @param curpars: the current value of all non-Gibbs parameters
    @param ml:      If True, return ML values, not a random sample
    """
    newpars = curpars.copy()

    # Sample the power spectra analytically
    for ii, psr in enumerate(likob.ptapsrs):
        for ss, signal in enumerate(likob.ptasignals):
            if signal['pulsarind'] == ii and signal['stype'] == 'spectrum':
                nms = likob.npm[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms

                # Loop over the frequencies
                for jj in range(signal['ntotpars']):
                    if ml:
                        tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)
                        rhonew = 0.5 * tau
                        newpars[signal['parindex']+jj] = np.log10(rhonew)
                    else:
                        # We can sample directly from this distribution.
                        # Prior domain:
                        pmin = signal['pmin'][jj]
                        pmax = signal['pmax'][jj]
                        rhomin = 10**pmin
                        rhomax = 10**pmax
                        tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)

                        # Draw samples between rhomax and rhomin, according to
                        # an exponential distribution
                        scale = 1 - np.exp(tau*(1.0/rhomax-1.0/rhomin))
                        eta = np.random.rand(1) * scale
                        rhonew = -tau / (np.log(1-eta)-tau/rhomax)

                        newpars[signal['parindex']+jj] = np.log10(rhonew)
            elif signal['pulsarind'] == ii and signal['stype'] == 'dmspectrum':
                nms = likob.npm[ii]
                nfs = likob.npf[ii]
                nfdms = likob.npfdm[ii]

                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms
                if 'rednoise' in likob.gibbsmodel:
                    ntot += nfs

                # Loop over the frequencies
                for jj in range(signal['ntotpars']):
                    if ml:
                        tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)
                        rhonew = 0.5 * tau
                        newpars[signal['parindex']+jj] = np.log10(rhonew)
                    else:
                        # We can sample directly from this distribution.
                        # Prior domain:
                        pmin = signal['pmin'][jj]
                        pmax = signal['pmax'][jj]
                        rhomin = 10**pmin
                        rhomax = 10**pmax
                        tau = 0.5*np.sum(a[ii][ntot+2*jj:ntot+2*jj+2]**2)

                        # Draw samples between rhomax and rhomin, according to
                        # an exponential distribution
                        scale = 1 - np.exp(tau*(1.0/rhomax-1.0/rhomin))
                        eta = np.random.rand(1) * scale
                        rhonew = -tau / (np.log(1-eta)-tau/rhomax)

                        newpars[signal['parindex']+jj] = np.log10(rhonew)

    return newpars

def gibbs_sample_loglik_Phi(likob, a, curpars, loglik_PSD, ml=False):
    """
    Sample the Phi-loglikelihood conditional.

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_PSD:  List of prepared likelihood/samplers for non-analytic
                        models
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    # First sample from the analytic signals
    newpars = gibbs_sample_loglik_Phi_an(likob, a, curpars)

    # Now do the numerical ones through sampling with an MCMC
    for psd in loglik_PSD:
        # Update the sampler with the new Gibbs-coefficients
        psd.setNewData(a[psd.psrindex][psd.gindices])
        ndim = psd.dimensions()

        if ml:
            newpars[psd.pindex:psd.pindex+ndim] = pnl.runPSO()
        else:
            p0 = newpars[psd.pindex:psd.pindex+ndim]
            newpars[psd.pindex:psd.pindex+ndim] = psd.runSampler(p0)

    return newpars

def gibbs_prepare_loglik_Det(likob, curpars):
    """
    Prepares the likelihood objects for deterministic parameters

    @param likob:   the full likelihood object
    @param curpars: the current value of all parameters
    """
    sigList = ['lineartimingmodel', 'nonlineartimingmodel', 'fouriermode', \
            'dmfouriermode', 'jitterfouriermode', \
            'bwm']

    loglik_Det = []

    mask = np.array([0]*likob.dimensions, dtype=np.bool)
    for ss, signal in enumerate(likob.ptasignals):
        if signal['stype'] in sigList:
            # Deterministic source, so include it
            pindex = signal['parindex']
            npars = np.sum(signal['bvary'])
            mask[pindex:pindex+npars] = True

    ndim = np.sum(mask)
    newpars = curpars.copy()

    if ndim > 0:
        # Only sample if we really have something to do
        pmin = likob.pmin[mask]
        pmax = likob.pmax[mask]
        pstart = likob.pstart[mask]
        pwidth = likob.pwidth[mask]
        bvary = np.array([1]*ndim, dtype=np.bool)

        # Prepare the likelihood function
        loglik_Det.append(pulsarDetLL(likob, newpars, mask, pmin, pmax, pstart, pwidth, bvary))
        pdl = loglik_Det[-1]

        pdl.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                covUpdate=ndim*400)

    return loglik_Det


def gibbs_sample_loglik_Det(likob, curpars, loglik_Det, ml=False):
    """
    Sample the Phi-loglikelihood conditional. Some models can be done
    analytically

    @param likob:       the full likelihood object
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_Det:  List of prepared likelihood/samplers
    @param ml:      If True, return ML values, not a random sample
    """

    # First sample from the analytic signals
    newpars = curpars.copy()

    # Now do the numerical ones through sampling with an MCMC
    for pdl in loglik_Det:
        ndim = pdl.dimensions()
        pdl.pstart = curpars.copy()

        if ml:
            newpars[pdl.mask] = pnl.runPSO()
        else:
            p0 = pdl.pstart[pdl.mask]
            newpars[pdl.mask] = pdl.runSampler(p0)

    return newpars



def gibbs_prepare_loglik_corrPhi(likob, curpars):
    """
    Prepares the likelihood objects for the power-spectra of correlated signals

    @param likob:   the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars: the current value of all parameters
    """

    loglik_corrPSD = []
    for ss, signal in enumerate(likob.ptasignals):
        if signal['stype'] == 'powerlaw' and signal['corr'] != 'single':
            bvary = signal['bvary']
            pindex = signal['parindex']
            pstart = signal['pstart']
            pmin = signal['pmin']
            pmax = signal['pmax']
            pwidth = signal['pwidth']
            Tmax = signal['Tmax']

            b = []
            gindices = []
            for ii, psr in enumerate(likob.ptapsrs):
                nms = likob.npm[ii]
                nfs = likob.npf[ii]
                nfdms = likob.npfdm[ii]
                npus = likob.npu[ii]

                b.append(np.zeros(nfs))

                # Save the indices of the Gibbs parameters
                ntot = 0
                if 'design' in likob.gibbsmodel:
                    ntot += nms
                if 'rednoise' in likob.gibbsmodel:
                    ntot += nfs
                if 'dm' in likob.gibbsmodel:
                    ntot += nfdms
                if 'jitter' in likob.gibbsmodel:
                    ntot += npus

                gindices.append(np.arange(ntot, ntot+nfs))

            ndim = np.sum(bvary)
            if ndim > 0:
                pstart[bvary] = curpars[pindex:pindex+np.sum(bvary)]

                # Make the b's with only GW gibsscoeffs here

                loglik_corrPSD.append(corrPSDLL(b, \
                        likob.Ffreqs_gw, Tmax, pmin, pmax, pstart, pwidth, \
                        pindex, -1, gindices, bvary))
                psd = loglik_corrPSD[-1]

                psd.initSampler(singleChain=ndim*20, fullChain=ndim*8000, \
                        covUpdate=ndim*400)

    return loglik_corrPSD

def gibbs_sample_loglik_corrPhi(likob, a, curpars, loglik_corrPSD, ml=False):
    """
    Sample the correlated Phi-loglikelihood conditional. Some models can be done
    analytically (latter not yet implemented)

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_PSD:  List of prepared likelihood/samplers for non-analytic
                        models
    @param ml:          If True, return ML values (PSO), not a random sample
    """
    newpars = curpars.copy()

    for psd in loglik_corrPSD:
        b = []
        for ii, ind in enumerate(psd.gindices):
            b.append(a[ii][ind])

        psd.setNewData(b, likob.Scor_inv, likob.Scor_ldet)
        ndim = psd.dimensions()

        if ml:
            newpars[psd.pindex:psd.pindex+ndim] = psd.runPSO()
        else:
            # Use an adaptive MCMC
            p0 = newpars[psd.pindex:psd.pindex+ndim]
            newpars[psd.pindex:psd.pindex+ndim] = psd.runSampler(p0)

    return newpars


def gibbs_prepare_loglik_imPhi(likob, curpars):
    """
    Prepares the likelihood objects for the power-spectra of all PSD signals in
    the case the correlated PSD coefficients are not modelled explicitly

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all parameters
    """

    loglik_imPSD = []

    # Require one PL spectrum per pulsar, and one correlated signal per pulsar
    pmin = np.array([])
    pmax = np.array([])
    pstart = np.array([])
    pwidth = np.array([])
    bvary = np.array([], dtype=np.bool)
    gindices = []
    b = []
    sigmask = np.array([0]*len(curpars), dtype=np.bool)

    for ii, psr in enumerate(likob.ptapsrs):
        sigs = likob.getSignalNumbersFromDict(likob.ptasignals, stype='powerlaw', \
                corr='single', psrind=ii)
        if len(sigs) != 1:
            raise ValueError("Pulsar {0} has no powerlaw signal")

        signal = likob.ptasignals[sigs[0]]

        nms = likob.npm[ii]
        nfs = likob.npf[ii]
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        # Save the indices of the Gibbs parameters
        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms

        gindices.append(np.arange(ntot, ntot+nfs))
        b.append(np.zeros(nfs))

        bvary = np.append(bvary, signal['bvary'])
        pstart = np.append(pstart, signal['pstart'])
        pmin = np.append(pmin, signal['pmin'])
        pmax = np.append(pmax, signal['pmax'])
        pwidth = np.append(pwidth, signal['pwidth'])

        pindex = signal['parindex']
        sigmask[pindex:pindex+2] = True

    sigs = likob.getSignalNumbersFromDict(likob.ptasignals, stype='powerlaw', \
            corr='gr', psrind=-1)
    if len(sigs) != 1:
        print "Sigs = ", sigs
        raise ValueError("Likob has no correlated powerlaw signal")

    signal = likob.ptasignals[sigs[0]]
    bvary = np.append(bvary, signal['bvary'])
    pstart = np.append(pstart, signal['pstart'])
    pmin = np.append(pmin, signal['pmin'])
    pmax = np.append(pmax, signal['pmax'])
    pwidth = np.append(pwidth, signal['pwidth'])
    Tmax = signal['Tmax']
    pindex = signal['parindex']
    sigmask[pindex:pindex+2] = True

    # Set the start parameters of the mini-chain
    pstart[bvary] = curpars[sigmask]
    ndim = np.sum(bvary)

    loglik_imPSD.append(implicitPSDLL(b, likob.Ffreqs_gw, Tmax, pmin, pmax, \
            pstart, pwidth, sigmask, gindices, bvary, likob))
    psd = loglik_imPSD[-1]

    psd.initSampler(singleChain=ndim*10, fullChain=ndim*8000, \
                        covUpdate=ndim*200)

    return loglik_imPSD


def gibbs_sample_loglik_imPhi(likob, a, curpars, loglik_imPSD, ml=False,
        runchain=True):
    """
    Sample the correlated Phi-loglikelihood conditional. Some models can be done
    analytically (latter not yet implemented)

    @param likob:       the full likelihood object
    @param a:           list of arrays with all the Gibbs-only parameters
    @param curpars:     the current value of all non-Gibbs parameters
    @param loglik_PSD:  List of prepared likelihood/samplers for non-analytic
                        models
    @param ml:          If True, return ML values (PSO), not a random sample
    @param runchain:    If False, do not run the chain (just calculate loglik)
    """
    newpars = curpars.copy()

    for psd in loglik_imPSD:
        b = []
        for ii, ind in enumerate(psd.gindices):
            b.append(a[ii][ind])

        psd.setNewData(b, likob.Scor)

        if ml:
            newpars[pmask] = psd.runPSO()
        else:
            # Use an adaptive MCMC
            p0 = newpars[psd.pmask]

            if runchain:
                newpars[psd.pmask] = psd.runSampler(p0)
            else:
                temp = psd.logposterior(p0)

    return newpars





def gibbs_prepare_correlations(likob):
    """
    Prepare the inverse covariance matrix with correlated signals, and related
    quantities for the Gibbs sampler

    @param likob:   the full likelihood object
    """
    if not np.all(likob.Svec == 0):
        likob.have_gibbs_corr = True

        # There actually is a signal, so invert the correlation covariance
        try:
            """
            U, s, Vt = sl.svd(likob.Scor)

            if not np.all(s > 0):
                raise ValueError("ERROR: WScor singular according to SVD")

            likob.Scor_inv = np.dot(U * (1.0/s), Vt)
            #likob.Scor_Li = U * (1.0 / np.sqrt(s))      # Do we need this?
            likob.Scor_ldet = np.sum(np.log(s))
            """
            cf = sl.cho_factor(likob.Scor)
            likob.Scor_inv = sl.cho_solve(cf, np.eye(likob.Scor.shape[0]))
            likob.Scor_ldet = 2*np.sum(np.log(np.diag(cf[0])))

        except ValueError:
            print "WTF?"
            print "Look in scor.txt for the Scor matrix"
            np.savetxt("scor.txt", likob.Scor)
            raise
            
    else:
        likob.have_gibbs_corr = False

def gibbs_psr_corrs_ex(likob, psrindex, a):
    """
    Get the Gibbs coefficient quadratic offsets for the correlated signals, for
    a specific pulsar

    @param likob:       The full likelihood object
    @param psrindex:    Index of the pulsar
    @param a:           List of Gibbs coefficient of all pulsar (of previous step)

    @return:    (pSinv_vec, pPvec), the quadratic offsets
    """
    psr = likob.ptapsrs[psrindex]

    # The inverse of the GWB correlations are easy
    pSinv_vec = (1.0 / likob.Svec[:likob.npf[psrindex]]) * \
            likob.Scor_inv[psrindex,psrindex]

    # For the quadratic offsets, we'll need to do some splice magic
    # First select the slice we'll need from the correlation matrix
    temp = np.arange(len(likob.ptapsrs))
    psrslice = np.delete(temp, psrindex)

    # The quadratic offset we'll return
    pPvec = np.zeros(psr.Fmat.shape[1])

    # Pre-compute the GWB-index offsets of all the pulsars
    corrmode_offset = []
    for ii in range(len(likob.ptapsrs)):
        nms = likob.npm[ii]
        nfs = likob.npf[ii]
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        # GWs are all the way at the end. Sum all we need
        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms
        if 'rednoise' in likob.gibbsmodel:
            ntot += nfs
        if 'dm' in likob.gibbsmodel:
            ntot += nfdms
        if 'jitter' in likob.gibbsmodel:
            ntot += npus

        corrmode_offset.append(ntot)

    for jj in psrslice:
        psrj = likob.ptapsrs[jj]
        minfreqs = min(len(psrj.Ffreqs), len(psr.Ffreqs))
        inda = corrmode_offset[jj]
        indb = corrmode_offset[jj] + minfreqs
        pPvec[:minfreqs] += a[jj][inda:indb] * \
                likob.Scor_inv[psrindex,jj] / likob.Svec[:minfreqs]

    return (pSinv_vec, pPvec)


def gibbs_psr_corrs_im(likob, psrindex, a):
    """
    Get the Gibbs coefficient quadratic offsets for the correlated signals, for
    a specific pulsar, when the correlated signal is not modelled explicitly
    with it's own Fourier coefficients

    @param likob:       The full likelihood object
    @param psrindex:    Index of the pulsar
    @param a:           List of Gibbs coefficient of all pulsar (of previous step)

    @return:    (pSinv_vec, pPvec), the quadratic offsets
    """
    psr = likob.ptapsrs[psrindex]

    Sinv = []
    pSinv_vec = np.zeros(likob.npf[psrindex])
    for ii, Scf in enumerate(likob.Scor_im_cf):
        Sinv.append(sl.cho_solve(Scf, np.eye(Scf[0].shape[0])))

        if ii < likob.npf[psrindex]:
            pSinv_vec[ii] = Sinv[-1][psrindex, psrindex]

    # The inverse of the GWB correlations are easy
    #pSinv_vec = (1.0 / likob.Svec[:likob.npf[psrindex]]) * \
    #        likob.Scor_inv[psrindex,psrindex]

    # For the quadratic offsets, we'll need to do some splice magic
    # First select the slice we'll need from the correlation matrix
    temp = np.arange(len(likob.ptapsrs))
    psrslice = np.delete(temp, psrindex)

    # The quadratic offset we'll return
    pPvec = np.zeros(psr.Fmat.shape[1])

    # Pre-compute the GWB-index offsets of all the pulsars
    corrmode_offset = []
    for ii in range(len(likob.ptapsrs)):
        nms = likob.npm[ii]
        nfs = likob.npf[ii]

        ntot = 0
        if 'design' in likob.gibbsmodel:
            ntot += nms

        corrmode_offset.append(ntot)


    for jj in psrslice:
        psrj = likob.ptapsrs[jj]
        minfreqs = min(len(psrj.Ffreqs), len(psr.Ffreqs))
        inda = corrmode_offset[jj]
        indb = corrmode_offset[jj] + minfreqs
        #pPvec[:minfreqs] += a[jj][inda:indb] * \
        #        likob.Scor_inv[psrindex,jj] / likob.Svec[:minfreqs]
        for ii in range(minfreqs):
            pPvec[ii] += Sinv[ii][psrindex, jj] * a[jj][inda+ii]

    return (pSinv_vec, pPvec)




def gibbs_sample_a(likob, a, ml=False):
    """
    Assume that all the noise parameters have been set (N, Phi, Theta). Given
    that, return a sample from the coefficient/timing model parameters

    @param likob:   the full likelihood object
    @param a:       the previous list of Gibbs coefficients.
    @param ml:      If True, return ML values, not a sample

    @return: list of coefficients/timing model parameters per pulsar, re-scaled
             coefficients, and the xi2
    """

    """
    if preva is None:
        preva = []
        for ii, psr in enumerate(likob.ptapsrs):
            # We have random indices, with amplitude of about 10ns
            preva.append(np.random.randn(likob.npz[ii])*1.0e-8)
            psr.gibbsresiduals = psr.detresiduals.copy()
    """

    npsrs = len(likob.ptapsrs)
    xi2 = np.zeros(npsrs)
    b = []

    for ii, psr in enumerate(likob.ptapsrs):
        nzs = likob.npz[ii]
        nms = likob.npm[ii]
        findex = np.sum(likob.npf[:ii])
        nfs = likob.npf[ii]
        fdmindex = np.sum(likob.npfdm[:ii])
        nfdms = likob.npfdm[ii]
        npus = likob.npu[ii]

        # Make ZNZ and Sigma
        #ZNZ = np.dot(psr.Zmat.T / psr.Nvec, psr.Zmat)
        ZNZ = cython_block_shermor_2D(psr.Zmat, psr.Nvec, psr.Jvec, psr.Uinds)
        Sigma = ZNZ.copy()

        # ahat is the slice ML value for the coefficients. Need ENx
        Nx = cython_block_shermor_0D(psr.detresiduals, \
                psr.Nvec, psr.Jvec, psr.Uinds)
        ENx = np.dot(psr.Zmat.T, Nx)

        # Depending on what signals are in the Gibbs model, we'll have to add
        # prior-covariances to ZNZ
        zindex = 0
        if 'design' in likob.gibbsmodel and nms > 0:
            # Do nothing, she'll be 'right
            zindex += nms

        if 'corrim' in likob.gibbsmodel and nfs > 0:
            (pSinv_vec, pPvec) = gibbs_psr_corrs_im(likob, ii, a)

            ind = range(zindex, zindex + nfs)
            Sigma[ind, ind] += pSinv_vec
            ENx[ind] -= pPvec
            zindex += nfs
        elif 'rednoise' in likob.gibbsmodel and nfs > 0:
            # Don't do this if it is included in corrim
            ind = range(zindex, zindex+nfs)
            Sigma[ind, ind] += 1.0 / likob.Phivec[findex:findex+nfs]
            zindex += nfs

        if 'dm' in likob.gibbsmodel and nfdms > 0:
            ind = range(zindex, zindex+nfdms)
            Sigma[ind, ind] += 1.0 / likob.Thetavec[fdmindex:fdmindex+nfdms]
            zindex += nfdms

        if 'jitter' in likob.gibbsmodel and npus > 0:
            ind = range(zindex, zindex+npus)
            Sigma[ind, ind] += 1.0 / psr.Jvec
            zindex += npus

        if 'correx' in likob.gibbsmodel and likob.have_gibbs_corr and nfs > 0:
            (pSinv_vec, pPvec) = gibbs_psr_corrs_ex(likob, ii, a)

            ind = range(zindex, zindex + nfs)
            Sigma[ind, ind] += pSinv_vec
            ENx[ind] -= pPvec
            zindex += nfs

        try:
            # Use a QR decomposition for the inversions
            Qs,Rs = sl.qr(Sigma) 

            #Qsb = np.dot(Qs.T, np.eye(Sigma.shape[0])) # computing Q^T*b (project b onto the range of A)
            #Sigi = sl.solve(Rs,Qsb) # solving R*x = Q^T*b
            Sigi = sl.solve(Rs,Qs.T) # solving R*x = Q^T*b
            
            # Ok, we've got the inverse... now what? Do SVD?
            U, s, Vt = sl.svd(Sigi)
            Li = U * np.sqrt(s)

            ahat = np.dot(Sigi, ENx)
        except (np.linalg.LinAlgError, ValueError):
            try:
                print "ERROR in QR. Doing SVD"

                U, s, Vt = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise np.linalg.LinAlgError
                    #raise ValueError("ERROR: Sigma singular according to SVD")
                Sigi = np.dot(U * (1.0/s), Vt)
                Li = U * (1.0 / np.sqrt(s))

                ahat = np.dot(Sigi, ENx)
            except np.linalg.LinAlgError:
                try:
                    print "ERROR in SVD. Doing Cholesky"

                    cfL = sl.cholesky(Sigma, lower=True)
                    cf = (cfL, True)

                    # Calculate the inverse Cholesky factor (can we do this faster?)
                    cfLi = sl.cho_factor(cfL, lower=True)
                    Li = sl.cho_solve(cfLi, np.eye(Sigma.shape[0]))

                    ahat = sl.cho_solve(cf, ENx)
                except np.linalg.LinAlgError:
                    # Come up with some better exception handling
                    print "ERROR in Cholesky. Help!"
                    raise
        except ValueError:
            print "WTF?"
            print "Look in sigma.txt for the Sigma matrix"
            np.savetxt("sigma.txt", Sigma)
            np.savetxt("phivec.txt", likob.Phivec[findex:findex+nfs])
            np.savetxt('nvec.txt', psr.Nvec)

            np.savetxt("znz.txt", ZNZ)
            np.savetxt("ENx.txt", ENx)

            raise

        # Get a sample from the coefficient distribution
        aadd = np.dot(Li, np.random.randn(Li.shape[0]))
        # See what happens if we use numpy
        # aadd = np.random.multivariate_normal(np.zeros(Sigi.shape[0]), \
        #        Sigi)
        #numpy.random.multivariate_normal(mean, cov[, size])

        if ml:
            addcoefficients = ahat
        else:
            addcoefficients = ahat + aadd

        # Test the coefficients for nans and infs
        nonan = np.all(np.logical_not(np.isnan(addcoefficients)))
        noinf = np.all(np.logical_not(np.isinf(addcoefficients)))
        if not (nonan and noinf):
            np.savetxt('ahat.txt', ahat)
            np.savetxt('aadd.txt', aadd)
            raise ValueError("Have inf or nan in solution")

        psr.gibbscoefficients = addcoefficients.copy()
        psr.gibbscoefficients[:psr.Mmat.shape[1]] = np.dot(psr.tmpConv, \
                addcoefficients[:psr.Mmat.shape[1]])

        # We save the quadratic parameters separately
        a[ii] = psr.gibbscoefficients
        b.append(addcoefficients)
        psr.gibbssubresiduals = np.dot(psr.Zmat, addcoefficients)
        psr.gibbsresiduals = psr.detresiduals - psr.gibbssubresiduals

        #xi2[ii] = np.sum(psr.gibbsresiduals**2 / psr.Nvec)
        # ZNZ = python_block_shermor_2d(psr.Zmat, psr.Nvec, psr.Jvec, psr.Uinds)
        tmp, xi2[ii] = cython_block_shermor_1D(psr.gibbsresiduals, \
                psr.Nvec, psr.Jvec, psr.Uinds)

    return a, b, xi2



def gibbsQuantities(likob, parameters):
    """
    Calculate basic Gibbs quantities at least once

    @param likob:       The full likelihood object
    @param parameters:  The current non-Gibbs model parameters
    """
    likob.setPsrNoise(parameters)

    # Place only correlated signals (GWB) in the Phi matrix, and the rest in the
    # noise vectors
    likob.constructPhiAndTheta(parameters, make_matrix=False, \
            noise_vec=True, gibbs_expansion=True)


    if likob.haveDetSources:
        likob.updateDetSources(parameters)



def RunGibbsMCMC(likob, steps, chainsdir, covfile=None, burnin=10000,
        noWrite=False, thin=100, p0=None):
    """
    Run a normal PTMCMC on the full likelihood used by the Gibbs sampler. So the
    quadratic parameters are created the Gibbs way.

    We are sampling the likelihood for now, so we do not have any min/max values
    for the parameters.

    @param likob:       The likelihood object, containing everything
    @param steps:       The number of full-circle Gibbs steps to take
    @param chainsdir:   Where to save the MCMC chain
    @param noWrite:     If True, do not write results to file
    """
    if not likob.likfunc in ['gibbs']:
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    # Save the description of all the parameters
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    # Clear the file for writing
    if not noWrite:
        chainfilename = chainsdir + '/chain_1.txt'
        chainfile = open(chainfilename, 'w')
        chainfile.close()

        # Also save the residuals for all pulsars
        likob.saveResiduals(chainsdir)

    # Dump samples to file every dumpint steps (no thinning)
    dumpint = 100

    ndim = likob.dimensions         # The non-Gibbs model parameters
    ncoeffs = np.sum(likob.npz)     # The Gibbs-only parameters/coefficients
    hpars = likob.pstart.copy()      # The Gibbs hyper parameters

    hpwidth = likob.pwidth.copy()   # Hyper-parameter width

    lqz, lqcov = gibbs_prepare_get_quadratic_cov(likob)

    qcov = block_diag(*lqcov)
    #lqwidth = np.hstack(*gibbs_prepare_get_quadratic_cov(likob))
    qpars = np.hstack(lqz)

    if p0 is None:
        p0 = np.append(hpars, 0.0*qpars)

    #pwidth = np.append(hpwidth, lqwidth)
    #cov = np.diag(pwidth**2)
    cov = block_diag(*[qcov, np.diag(hpwidth**2)])

    gl = gibbs_likob_class(likob)

    sampler = ptmcmc.PTSampler(ndim + ncoeffs, gl.loglikelihood, \
            gl.logprior, cov=cov, outDir=chainsdir, verbose=True)

    sampler.sample(p0, steps, thin=thin, burn=burnin)







def RunGibbs_mark1(likob, steps, chainsdir, noWrite=False, joinNJ=True):
    """
    Run a gibbs sampler on a, for now, simplified version of the likelihood.

    Parameters are grouped into several categories:
    1) a, the Fourier coefficients and timing model parameters
    2) N, the white noise parameters
    3) Phi, the red noise PSD coefficients
    4) Jitter: pulse Jitter. Can be included in N (default)
    5) Deterministic: all deterministic sources not described elsewhere

    Note that this blocked Gibbs sampler has more parameters than the usual
    MCMC's (except for mark11). This sampler also samples from the timing model
    parameters, and the Fourier coefficients: the Gibbs-only parameters. These
    are all appended in the MCMC chain after the normal parameters. Everything
    is correctly labeled in the 'ptparameters.txt' file.

    @param likob:       The likelihood object, containing everything
    @param steps:       The number of full-circle Gibbs steps to take
    @param chainsdir:   Where to save the MCMC chain
    @param noWrite:     If True, do not write results to file
    @param joinNJ:      Join sampling from white noise and jitter/ecorr
    """
    if not likob.likfunc in ['gibbs']:
        raise ValueError("Likelihood not initialised for Gibbs sampling")

    # Save the description of all the parameters
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    # Clear the file for writing
    if not noWrite:
        chainfilename = chainsdir + '/chain_1.txt'
        chainfile = open(chainfilename, 'w')
        chainfile.close()

        # Debugging purposes
        #chainfilename_b = chainsdir + '/chain_1-b.txt'
        #chainfile_b = open(chainfilename_b, 'w')
        #chainfile_b.close()

        #xi2filename = chainsdir + '/xi2.txt'
        #xi2file = open(xi2filename, 'w')
        #xi2file.close()

        # Also save the residuals for all pulsars
        likob.saveResiduals(chainsdir)

    # Dump samples to file every dumpint steps (no thinning)
    dumpint = 100

    ndim = likob.dimensions         # The non-Gibbs model parameters
    ncoeffs = np.sum(likob.npz)     # The Gibbs-only parameters/coefficients
    pars = likob.pstart.copy()      # The Gibbs

    # Make a list of all the blocked signal samplers (except for the coefficient
    # samplers)
    #if 'jitter' in likob.gibbsmodel and not joinNJ:
    if joinNJ:
        # Jitter might be present, and combined with white noise
        loglik_N = []
        loglik_J = []
        loglik_NJ = gibbs_prepare_loglik_NJ(likob, pars)
    else:
        # Jitter might be present explicitly (non-marginalized)
        loglik_N = gibbs_prepare_loglik_N(likob, pars)
        loglik_J = gibbs_prepare_loglik_J(likob, pars)
        loglik_NJ = []

        # Not joining NJ is unwise. Give a warning
        if 'jitter' in likob.gibbsmodel:
            print("WARNING: not joining NJ increases chain acor-length.")
        else:
            print("WARNING: not joining NJ. Ecorr in model will lead to error.")

    loglik_Det = gibbs_prepare_loglik_Det(likob, pars)

    if not 'corrim' in likob.gibbsmodel:
        loglik_PSD = gibbs_prepare_loglik_Phi(likob, pars)
        loglik_corrPSD = gibbs_prepare_loglik_corrPhi(likob, pars)
        loglik_imPSD = []
    else:
        loglik_PSD = []
        loglik_corrPSD = []
        loglik_imPSD = gibbs_prepare_loglik_imPhi(likob, pars)
        likob.Scor_im = []
        likob.Scor_im_cf = []


    # The gibbs coefficients are initially set to 5ns random, each
    a = []
    b = []
    for ii, psr in enumerate(likob.ptapsrs):
        gibbsQuantities(likob, pars)
        a.append(np.random.randn(likob.npz[ii])*5.0e-9)
        b.append(a[-1].copy())
        psr.gibbsresiduals = psr.detresiduals.copy()

    if 'corrim' in likob.gibbsmodel:
        # We need to sample the correlated signals once to obtain the matrices
        # for the sampling of the coefficients
        temp = gibbs_sample_loglik_imPhi(likob, a, pars, loglik_imPSD,
                runchain=False)

    samples = np.zeros((min(dumpint, steps), ndim+ncoeffs))
    #samples2 = np.zeros((min(dumpint, steps), ncoeffs))

    loglik = np.zeros(min(dumpint, steps))
    logpost = np.zeros(min(dumpint, steps))

    stepind = 0
    for step in range(steps):
        doneIteration = False
        iter = 0

        # Start with calculating the required likelihood quantities
        gibbsQuantities(likob, pars)

        # If necessary, invert the correlation matrix Svec & Scor with Ffreqs_gw
        # and Fmat_gw
        if 'correx' in likob.gibbsmodel:
            gibbs_prepare_correlations(likob)

        while not doneIteration:
            try:
                # Generate new coefficients
                #a, b, xi2 = gibbs_sample_a(likob, a)

                # This is the function from the mark2 Gibbs sampler. However,
                # that one does not transform the quadratic parameters, and has
                # the values of b in a.
                for pp, psr in enumerate(likob.ptapsrs):
                    likob.gibbs_current_a = b
                    if joinNJ:
                        # Jitter/ECORR is mot yet subtracted, so it is present
                        # in psr.gibbsresiduals!! Make sure that NJ is the first
                        # next conditional.
                        # This conditional also sets gibbssubresiduals, which is
                        # used in pulsarDetLL
                        a[pp], b[pp], xi2 = likob.gibbs_sample_psr_quadratics(pars, b[pp], pp, \
                                which='N', joinNJ=True)
                    else:
                        a[pp], b[pp], xi2 = likob.gibbs_sample_psr_quadratics(pars, b[pp], pp, \
                                which='all', joinNJ=False)

                samples[stepind, ndim:] = np.hstack(a)
                #samples2[stepind, :] = np.hstack(b)

                doneIteration = True

            except (np.linalg.LinAlgError, ValueError):
                # Why does SVD sometimes not converge?
                # Try different values...
                iter += 1

                if iter > 100:
                    print "WARNING: numpy.linalg problems"
                    raise

            # If joinNJ=True, we still have the jitter included in
            # psr.gibbsresiduals and gibbssubresiduals. Therefore, this
            # conditional is FIRST!!
            # -----
            # Generate new white noise parameters (whith jitter/cequad)
            pars = gibbs_sample_loglik_NJ(likob, pars, loglik_NJ)
            a, b = gibbs_update_ecor_NJ(likob, pars, a, b, joinNJ=joinNJ)

            # Generate new white noise parameters (whithout jitter/cequad)
            pars = gibbs_sample_loglik_N(likob, pars, loglik_N)

            # Generate new red noise parameters (or DM variations)
            pars = gibbs_sample_loglik_Phi(likob, a, pars, loglik_PSD)

            # Generate new correlated equad/jitter parameters
            pars = gibbs_sample_loglik_J(likob, a, pars, loglik_J)

            # If we have 'm, sample from the deterministic sources
            pars = gibbs_sample_loglik_Det(likob, pars, loglik_Det)

            if 'correx' in likob.gibbsmodel and likob.have_gibbs_corr:
                # Generate new GWB parameters
                pars = gibbs_sample_loglik_corrPhi(likob, a, pars, loglik_corrPSD)

            pars = gibbs_sample_loglik_imPhi(likob, a, pars, loglik_imPSD)

        samples[stepind, :ndim] = pars

        # Calculate the full log-likelihood. Use the Zmat-basis timing model
        aparameters = samples[stepind, :].copy()
        aparameters[ndim:] = np.hstack(b)
        loglik[stepind] = gibbs_loglikelihood(likob, aparameters)
        logpost[stepind] = loglik[stepind] + gibbs_logprior(likob, aparameters)

        stepind += 1
        # Write to file if necessary
        if (stepind % dumpint == 0 or step == steps-1):
            nwrite = dumpint

            # Check how many samples we are writing
            if step == steps-1:
                nwrite = stepind

            # Open the file in append mode
            if not noWrite:
                chainfile = open(chainfilename, 'a+')
                for jj in range(nwrite):
                    chainfile.write('%.17e\t  %.17e\t  0.0\t' % (logpost[jj], \
                        loglik[jj]))
                    chainfile.write('\t'.join(["%.17e"%\
                            (samples[jj,kk]) for kk in range(ndim+ncoeffs)]))
                    chainfile.write('\n')
                chainfile.close()

                #chainfile_b = open(chainfilename_b, 'a+')
                #for jj in range(nwrite):
                #    chainfile_b.write('%.17e\t  %.17e\t  0.0\t' % (logpost[jj], \
                #        loglik[jj]))
                #    chainfile_b.write('\t'.join(["%.17e"%\
                #            (samples2[jj,kk]) for kk in range(ncoeffs)]))
                #    chainfile_b.write('\n')
                #chainfile_b.close()
            stepind = 0

        #if not noWrite:
        #    xi2file = open(xi2filename, 'a+')

        #    xi2file.write('{0}\t'.format(step))
        #    xi2file.write('\t'.join(["%.17e"%\
        #            (xi2[kk]) for kk in range(len(xi2))]))
        #    xi2file.write('\n')
        #    xi2file.close()

        percent = (step * 100.0 / steps)
        sys.stdout.write("\rGibbs: %d%%" %percent)
        sys.stdout.flush()

    sys.stdout.write("\n")
