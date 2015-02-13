from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os as os
import glob
import sys

from piccard import *


def crossPower(likobs, mlpars):
    """
    Calculate the cross-power spectrum from a series of single-pulsar likelihood
    objects, and their maximum-likelihood parameters

    @param likobs:  list of likelihood objects
    @param mlpars:  list of ML parameters

    The cross-power of pulsar a and b is defined as: num / den

    num = residuals_a * C_a^{-1} * Cgw_ab * C_b^{-1} * residuals_b
    den = Trace( C_a^{-1} * Cgw_{ab} * C_b^{-1} * Cgw_{ba} )

    err = 1.0 / sqrt(den)
    """
    npsrs = len(likobs)
    if npsrs != len(mlpars):
        raise ValueError("# of likelihood objects not equal to # of mlpars")

    npf = np.zeros(npsrs, dtype=np.int)
    npfdm = np.zeros(npsrs, dtype=np.int)
    npu = np.zeros(npsrs, dtype=np.int)

    pairs = int(0.5 * npsrs * (npsrs - 1))
    crosspower = np.zeros(pairs)
    crosspowererr = np.zeros(pairs)
    hdcoeff = np.zeros(pairs)
    angle = np.zeros(pairs)

    for oo, likob in enumerate(likobs):
        if len(likob.ptapsrs) > 1:
            raise ValueError("# of pulsars for likob {0} > 1".format(oo))

        # For all the pulsars, calculate the single-pulsar likelhood once first.
        # This calculates all the necessary auxiliaries correctly
        likob.logposterior(mlpars[oo])

        npf[oo] = likob.npf[0]
        npfdm[oo] = likob.npfdm[0]
        npu[oo] = likob.npu[0]

        # We'll work in the basis of mark6 (with DM). If we do not have DM as in
        # mark3, create the correct auxiliary references
        if likob.likfunc[:5] == 'mark3':
            likob.rGE = likob.rGF
            likob.EGGNGGE = likob.FGGNGGF
        elif likob.likfunc[:5] == 'mark6':
            pass
        elif likob.likfunc[:5] == 'mark4':
            likob.rGE = likob.rGU
            likob.EGGNGGE = likob.UGGNGGU


        try:
            likob.Sigcf = sl.cho_factor(likob.Sigma)
            likob.SEGE = sl.cho_solve(likob.Sigcf, likob.EGGNGGE)
        except np.linalg.LinAlgError:
            raise
            try:
                U, s, Vh = sl.svd(likob.Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                likob.SEGE = np.dot(Vh.T, np.dot(np.diag(1.0/s), \
                        np.dot(U.T, likob.EGGNGGE)))
            except np.linalg.LinAlgError:
                raise ValueError("SVD did not converge?")

        likob.OSr = likob.rGE - np.dot(likob.SEGE.T, likob.rGE)
        likob.OSE = likob.EGGNGGE - np.dot(likob.EGGNGGE, likob.SEGE)

        # Construct the position vector
        psr = likob.ptapsrs[0]
        likob.pos = np.array([np.cos(psr.decj)*np.cos(psr.raj),
                              np.cos(psr.decj)*np.sin(psr.raj),
                              np.sin(psr.decj)])

    # Figure out how many frequency components we have for each pulsar
    #totfreqs = np.sum(npf) + np.sum(npfdm)
    #PhiTheta = np.zeros((totfreqs, totfreqs))

    # Calculate the cross-power
    ind = 0
    for pa, likoba in enumerate(likobs):
        for pb in range(pa+1, len(likobs)):
            likobb = likobs[pb]

            PhiTheta = np.zeros((npf[pa]+npfdm[pa], npf[pb]+npfdm[pb]))
            nfreqs = np.min([npf[pa], npf[pb]])
            freqpy = likoba.ptapsrs[0].Ffreqs[:nfreqs] * pic_spy
            # Tmax = likoba.ptapsrs[0].Tmax
            Tmax = likoba.Tmax
            pcdoubled = (pic_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-4.33)

            di = np.diag_indices(nfreqs)
            PhiTheta[di] = pcdoubled
            #pcc = np.append(pcdoubled, 0.0*pcdoubled)
            #PhiTheta = np.diag(pcc)

            if likoba.likfunc[:5] == 'mark4':
                tempPT = np.dot(likoba.ptapsrs[0].UtF[:,:nfreqs], PhiTheta[:nfreqs,:])
            else:
                tempPT = PhiTheta

            if likobb.likfunc[:5] == 'mark4':
                PT = np.dot(tempPT[:,:nfreqs], likobb.ptapsrs[0].UtF[:,:nfreqs].T)
            else:
                PT = tempPT

            num = np.dot(likoba.OSr, np.dot(PT, likobb.OSr))

            den = np.trace(np.dot(likoba.OSE, np.dot(PT, \
                    np.dot(likobb.OSE, PT.T))))

            # Construct position vector and H&D coeff
            angle[ind] = np.arccos(np.sum(likoba.pos * likobb.pos))
            xp = 0.5 * (1 - np.sum(likoba.pos * likobb.pos))
            logxp = 1.5 * xp * np.log(xp)
            hdcoeff[ind] = logxp - 0.25 * xp + 0.5

            # Crosspower and uncertainty
            crosspower[ind] = num / den
            crosspowererr[ind] = 1.0 / np.sqrt(den)

            ind += 1

    return (angle, hdcoeff, crosspower, crosspowererr)

def mark12prep(self, parameters):
    """
    Modified from mark12loglikelihood. Prepare all the relevant quantities to
    evaluate the crossPower

    # For the model, the 'gibbsmodel' description is used

    return: rNF, rNT, Sigma, Sigmacf, TNF
    """
    npsrs = len(self.ptapsrs)

    # The red signals (don't form matrices (slow); use the Gibbs expansion)
    # Is this faster: ?
    #self.setPhi(parameters, gibbs_expansion=True) 
    #self.setTheta(parameters, pp=pp)
    self.constructPhiAndTheta(parameters, make_matrix=False, \
            gibbs_expansion=True)
    self.gibbs_construct_all_freqcov()

    # Band-limited red noise
    for pp, psr in enumerate(self.ptapsrs):
        self.setBeta(parameters, pp=pp)

    # The white noise
    self.setPsrNoise(parameters)

    if self.haveDetSources:
        self.updateDetSources(parameters)

    # Size of the full matrix is:
    nt = np.sum(self.npz) - np.sum(self.npu)# Size full T-matrix
    nf = np.sum(self.npf)                   # Size full F-matrix
    Sigma = np.zeros((nt, nt))              # Full Sigma matrix
    FPhi = np.zeros((nf, nf))               # RN full Phi matrix
    Finds = np.zeros(nf, dtype=np.int)      # T<->F translation indices
    ZNZ = np.zeros((nt, nt))                # Full ZNZ matrix
    Zx = np.zeros(nt)
    Fx = np.zeros(nf)
    TNF = np.zeros((nt, nf))
    FNF = np.zeros((nf, nf))
    Jldet = np.zeros(npsrs)                 # All log(det(N)) values
    ThetaLD = 0.0
    PhiLD = 0.0
    BetaLD = 0.0
    rGr = np.zeros(npsrs)                   # All rNr values
    sind = 0                                # Sigma-index
    fsind = 0                               # Sigma-index
    phind = 0                               # Z/T-index

    for ii, psr in enumerate(self.ptapsrs):
        # The T-matrix is just the Z-matrix, minus the ecorr/jitter
        tsize = psr.Zmat.shape[1] - np.sum(psr.Zmask_U)
        fsize = psr.Fmat.shape[1]
        Zmat = psr.Zmat[:,:tsize]
        Fmat = psr.Fmat[:,:fsize]

        # Use jitter extension for ZNZ
        Jldet[ii], ZNZp = cython_block_shermor_2D(Zmat, psr.Nvec, \
                psr.Jvec, psr.Uinds)
        Nx = cython_block_shermor_0D(psr.detresiduals, \
                psr.Nvec, psr.Jvec, psr.Uinds)
        Zx[sind:sind+tsize] = np.dot(Zmat.T, Nx)
        Fx[fsind:fsind+fsize] = np.dot(Fmat.T, Nx)

        Jldet[ii], TNFp = python_block_shermor_2D_asymm(\
                Zmat, Fmat, psr.Nvec, psr.Jvec, psr.Uinds)
        TNF[sind:sind+tsize,fsind:fsind+fsize] = TNFp

        Jldet[ii], FNFp = cython_block_shermor_2D(\
                Fmat, psr.Nvec, psr.Jvec, psr.Uinds)
        FNF[fsind:fsind+fsize, fsind:fsind+fsize] = FNFp

        Jldet[ii], rGr[ii] = cython_block_shermor_1D(\
                psr.detresiduals, psr.Nvec, psr.Jvec, psr.Uinds)

        ZNZ[sind:sind+tsize, sind:sind+tsize] = ZNZp

        # Create the prior (Sigma = ZNZ + Phi)
        nms = self.npm[ii]
        nfs = self.npf[ii]
        nfbs = self.npfb[ii]
        nfdms = self.npfdm[ii]
        findex = np.sum(self.npf[:ii])
        fdmindex = np.sum(self.npfdm[:ii])
        fbindex = np.sum(self.npfb[:ii])
        if 'design' in self.gibbsmodel:
            phind += nms         
        if 'rednoise' in self.gibbsmodel:
            # Red noise, excluding GWS

            if npsrs == 1:
                inds = slice(sind+phind, sind+phind+nfs)
                di = np.diag_indices(nfs)
                Sigma[inds, inds][di] += 1.0 / ( \
                        self.Phivec[findex:findex+nfs] + \
                        self.Svec[findex:findex+nfs])
                PhiLD += np.sum(np.log(self.Phivec[findex:findex+nfs] + \
                        self.Svec[findex:findex+nfs]))
                phind += nfs
            elif npsrs > 1:
                # We need to do the full array at once. Do that below
                # Here, we construct the indexing matrices
                Finds[findex:findex+nfs] = np.arange(phind, phind+nfs)
                phind += nfs
        if 'freqrednoise' in self.gibbsmodel:
            inds = slice(sind+phind, sind+phind+nfbs)
            di = np.diag_indices(nfbs)
            Sigma[inds, inds][di] += 1.0 / self.Betavec[fbindex:fbindex+nfbs]
            BetaLD += np.sum(np.log(self.Betavec[fbindex:fbindex+nfbs]))
            phind += nfbs
        if 'dm' in self.gibbsmodel:
            inds = slice(sind+phind, sind+phind+nfdms)
            di = np.diag_indices(nfdms)
            Sigma[inds, inds][di] += 1.0 / self.Thetavec[fdmindex:fdmindex+nfdms]
            ThetaLD += np.sum(np.log(self.Thetavec[fdmindex:fdmindex+nfdms]))
            phind += nfdms

        sind += tsize
        fsind += fsize

    if npsrs > 1 and 'rednoise' in self.gibbsmodel:
        msk_ind = np.zeros(self.freqmask.shape, dtype=np.int)
        msk_ind[self.freqmask] = np.arange(np.sum(self.freqmask))
        msk_zind = np.arange(np.sum(self.npf))
        for mode in range(0, self.freqmask.shape[1], 2):
            freq = int(mode/2)          # Which frequency

            # We had pre-calculated the Cholesky factor and the inverse
            # (in gibbs_construct_all_freqcov)
            rncov_inv = self.Scor_im_inv[freq]
            cf = self.Scor_im_cf[freq]
            PhiLD += 4 * np.sum(np.log(np.diag(cf[0])))

            # We have the inverse for the individual modes now. Add them to
            # the full prior covariance matrix

            # Firstly the Cosine mode
            newmsk = np.zeros(self.freqmask.shape, dtype=np.bool)
            newmsk[:, mode] = self.freqmask[:, mode]
            mode_ind = msk_ind[newmsk]
            z_ind = msk_zind[mode_ind]
            FPhi[np.array([z_ind]).T, z_ind] += rncov_inv

            # Secondly the Sine mode
            newmsk[:] = False
            newmsk[:, mode+1] = self.freqmask[:, mode+1]
            mode_ind = msk_ind[newmsk]
            z_ind = msk_zind[mode_ind]
            FPhi[np.array([z_ind]).T, z_ind] += rncov_inv

        Sigma[np.array([Finds]).T, Finds] += FPhi

    # If we have a non-trivial prior matrix, invert that stuff
    if 'rednoise' in self.gibbsmodel or \
            'dm' in self.gibbsmodel or \
            'freqrednoise' in self.gibbsmodel:
        Sigma += ZNZ

        # With Sigma constructed, we can invert it
        try:
            cf = sl.cho_factor(Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rSr = np.dot(Zx, sl.cho_solve(cf, Zx))
        except np.linalg.LinAlgError:
            print "Using SVD... return -inf"
            return -np.inf

            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rSr = np.dot(Nx, np.dot(Vh.T / s, np.dot(U.T, Nx)))
    else:
        SigmaLD = 0.0
        ThetaLD = 0.0
        PhiLD = 0.0
        BetaLD = 0.0
        rSr = 0.0
        cf = None

    # rNF, rNT, Sigma, Sigmacf, TNF
    # Fx, Zx, Sigma, cf, TNF

    return Fx, Zx, Sigma, cf, TNF, FNF

def mark12CrossPower(likobs, mlpars):
    """
    Calculate the cross-power spectrum from a series of single-pulsar likelihood
    objects, and their maximum-likelihood parameters

    @param likobs:  list of likelihood objects
    @param mlpars:  list of ML parameters

    The cross-power of pulsar a and b is defined as: num / den

    num = residuals_a * C_a^{-1} * Cgw_ab * C_b^{-1} * residuals_b
    den = Trace( C_a^{-1} * Cgw_{ab} * C_b^{-1} * Cgw_{ba} )

    err = 1.0 / sqrt(den)

    Note: use the mark12 formalism
    """
    npsrs = len(likobs)
    if npsrs != len(mlpars):
        raise ValueError("# of likelihood objects not equal to # of mlpars")

    npf = np.zeros(npsrs, dtype=np.int)
    npfdm = np.zeros(npsrs, dtype=np.int)
    npu = np.zeros(npsrs, dtype=np.int)

    pairs = int(0.5 * npsrs * (npsrs - 1))
    crosspower = np.zeros(pairs)
    crosspowererr = np.zeros(pairs)
    hdcoeff = np.zeros(pairs)
    angle = np.zeros(pairs)

    for oo, likob in enumerate(likobs):
        if len(likob.ptapsrs) > 1:
            raise ValueError("# of pulsars for likob {0} > 1".format(oo))

        # For all the pulsars, calculate the single-pulsar likelhood once first.
        # This calculates all the necessary auxiliaries correctly
        #likob.logposterior(mlpars[oo])
        Fx, Tx, Sigma, cf, TNF, FNF = mark12prep(likob, mlpars[oo])

        npf[oo] = likob.npf[0]

        psr = likob.ptapsrs[0]
        likob.pos = np.array([np.cos(psr.decj)*np.cos(psr.raj),
                              np.cos(psr.decj)*np.sin(psr.raj),
                              np.sin(psr.decj)])

        likob.OSr = Fx - np.dot(TNF.T, sl.cho_solve(cf, Tx))
        likob.OST = FNF - np.dot(TNF.T, sl.cho_solve(cf, TNF))

    # Figure out how many frequency components we have for each pulsar

    # Calculate the cross-power
    ind = 0
    for pa, likoba in enumerate(likobs):
        for pb in range(pa+1, len(likobs)):
            likobb = likobs[pb]

            Phi = np.zeros((npf[pa], npf[pb]))
            nfreqs = np.min([npf[pa], npf[pb]])
            freqpy = likoba.ptapsrs[0].Ffreqs[:nfreqs] * pic_spy
            Tmax = likoba.Tmax
            pcdoubled = (pic_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-4.33)

            di = np.diag_indices(nfreqs)
            Phi[di] = pcdoubled

            num = np.dot(likoba.OSr, np.dot(Phi, likobb.OSr))
            den = np.trace(np.dot(likoba.OST, np.dot(Phi, \
                    np.dot(likobb.OST, Phi.T))))

            # Construct position vector and H&D coeff
            angle[ind] = np.arccos(np.sum(likoba.pos * likobb.pos))
            xp = 0.5 * (1 - np.sum(likoba.pos * likobb.pos))
            logxp = 1.5 * xp * np.log(xp)
            hdcoeff[ind] = logxp - 0.25 * xp + 0.5

            # Crosspower and uncertainty
            crosspower[ind] = num / den
            crosspowererr[ind] = 1.0 / np.sqrt(den)

            ind += 1

    return (angle, hdcoeff, crosspower, crosspowererr)


def fitCrossPower(hdcoeff, crosspower, crosspowererr):
    """
    Fit the results of the optimal statistic crossPower to the Hellings and
    Downs correlation function, and return the A^2 and \delta A^2

    @param hdcoeff:         Array of H&D coefficients for all the pulsar pairs
    @param crosspower:      Array of cross-power measured for all the pulsars
    @param crosspowererr:   Array of the error of the cross-power measurements

    @return:        Value of h^2 and the uncertainty in it.
    """
    hc_sqr = np.sum(crosspower*hdcoeff / (crosspowererr*crosspowererr)) / np.sum(hdcoeff*hdcoeff / (crosspowererr*crosspowererr))
    hc_sqrerr = 1.0 / np.sqrt(np.sum(hdcoeff * hdcoeff / (crosspowererr * crosspowererr)))
    return hc_sqr, hc_sqrerr


def prepPsrForTeststat(likob, mlpar, mlNoise=False, mlWhiteNoise=False):
    """
    This function creates some Auxiliary quantities for use in the Shannonensque
    test statistic
    """
    if len(likob.ptapsrs) > 1:
        raise ValueError("# of pulsars for likob {0} > 1".format(oo))

    psr = likob.ptapsrs[0]

    # We'll work in the basis of mark6 (with DM). If we do not have DM as in
    # mark3, create the correct auxiliary references
    if likob.likfunc[:5] == 'mark3':
        psr.newEmat = psr.Fmat
    elif likob.likfunc[:5] == 'mark6':
        psr.newEmat = psr.Emat
        nfreqs = len(psr.Ffreqs)
        psr.Fmat = psr.Emat[:,:nfreqs]
    elif likob.likfunc[:5] == 'mark4':
        psr.newEmat = psr.Umat
        nfreqs = len(psr.Ffreqs)
        #psr.Fmat = psr.Emat[:,:nfreqs]
        psr.Fmat, temp = fourierdesignmatrix(psr.toas, nfreqs, likob.Tmax)
        if not np.all(psr.Ffreqs == temp):
            raise ValueError("Not all frequencies the same for {0}".format(psr.name))

    Nvec = psr.toaerrs
    MtM = np.dot(psr.Mmat.T, psr.Mmat)
    try:
        cf = sl.cho_factor(MtM)
        psr.MtMM = sl.cho_solve(cf, psr.Mmat.T)
    except np.linalg.LinAlgError:
        U, s, Vh = sl.svd(MtM)
        if not np.all(s >= 0):
            raise ValueError("MtM singular according to SVD")
        s[s!=0] = 1.0 / s[s!=0]
        psr.MtMM = np.dot(np.dot(Vh.T, np.dot(np.diag(s), U.T)), psr.Mmat.T)

    psr.MtMN = (psr.MtMM * Nvec)
    psr.MTMMF = np.dot(psr.MtMN, psr.Fmat)
    psr.NiE = ((1.0/Nvec) * psr.newEmat.T).T
    psr.MtMMNE = np.dot(psr.MtMM, psr.NiE)

    pars = likob.pstart.copy()

    for pd in likob.pardes:
        if pd['index'] >= 0:
            if pd['sigtype'] == 'efac':
                if not mlWhiteNoise:
                    pars[pd['index']] = mlpar[pd['index']]
                else:
                    pars[pd['index']] = 1.0
            elif pd['sigtype'] == 'equad':
                if not mlWhiteNoise:
                    pars[pd['index']] = mlpar[pd['index']]
                else:
                    pars[pd['index']] = likob.pmin[pd['index']]
            elif pd['sigtype'] == 'jitter':
                if not mlWhiteNoise:
                    pars[pd['index']] = mlpar[pd['index']]
                else:
                    pars[pd['index']] = likob.pmin[pd['index']]
            elif pd['sigtype'] == 'powerlaw':
                if pd['id'] == 'RN-Amplitude':
                    if mlNoise:
                        pars[pd['index']] = mlpar[pd['index']]
                    else:
                        pars[pd['index']] = likob.pmin[pd['index']]
                elif pd['id'] == 'RN-Spectral-index':
                    if mlNoise:
                        pars[pd['index']] = mlpar[pd['index']]
                    else:
                        pars[pd['index']] = 3.1
            elif pd['sigtype'] == 'dmpowerlaw':
                if pd['id'] == 'DM-Amplitude':
                    pars[pd['index']] = mlpar[pd['index']]
                elif pd['id'] == 'DM-Spectral-index':
                    pars[pd['index']] = mlpar[pd['index']]

    likob.logposterior(pars)

    # We'll work in the basis of mark6 (with DM). If we do not have DM as in
    # mark3, create the correct auxiliary references
    if likob.likfunc[:5] == 'mark3':
        likob.rGE = likob.rGF
        likob.EGGNGGE = likob.FGGNGGF
    elif likob.likfunc[:5] == 'mark6':
        pass
    elif likob.likfunc[:5] == 'mark4':
        likob.rGE = likob.rGU
        likob.EGGNGGE = likob.UGGNGGU


    try:
        likob.Sigcf = sl.cho_factor(likob.Sigma)
        likob.SEGE = sl.cho_solve(likob.Sigcf, likob.EGGNGGE)
    except np.linalg.LinAlgError:
        raise
        try:
            U, s, Vh = sl.svd(likob.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            likob.SEGE = np.dot(Vh.T, np.dot(np.diag(1.0/s), \
                    np.dot(U.T, likob.EGGNGGE)))
        except np.linalg.LinAlgError:
            raise ValueError("SVD did not converge?")

    likob.OSr = likob.rGE - np.dot(likob.SEGE.T, likob.rGE)
    likob.OSE = likob.EGGNGGE - np.dot(likob.EGGNGGE, likob.SEGE)

    # Construct the position vector
    psr = likob.ptapsrs[0]
    likob.pos = np.array([np.cos(psr.decj)*np.cos(psr.raj),
                          np.cos(psr.decj)*np.sin(psr.raj),
                          np.sin(psr.decj)])

def genFakeSet(likob, gwAmp):
    psr = likob.ptapsrs[0]

    Tmax = likob.Tmax
    freqpy = psr.Ffreqs * pic_spy
    pcdoubled = (gwAmp**2 * pic_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-4.33)
    pc = np.sqrt(pcdoubled)

    nphi = psr.Fmat.shape[1]
    n = len(psr.toas)
    xi = np.random.randn(nphi)
    chi = np.random.randn(n)

    r = np.dot(psr.Fmat, pc * xi) - \
        np.dot(psr.Mmat, np.dot(psr.MTMMF, pc * xi)) + \
        psr.toaerrs * chi - \
        np.dot(psr.Mmat, np.dot(psr.MtMN,  chi))

    return r

def rGE(psr, r):
    """
    get rGE = r G (G^{T} N G) G^{T} newEmat = \
            r Ni newEmat - r Ni Mmat ( Mmat^{T} Mmat)^{-1} Mmat^{T} Ni newEmat
    """
    #psr.NiE = ((1.0/Nvec) * psr.newEmat.T).T
    #psr.MtMMNE = np.dot(MtMM, psr.NiE)

    return np.dot(r, psr.NiE) - np.dot(np.dot(r,  psr.Mmat), psr.MtMMNE)

def autoPower(likobs, datasets):
    """
    Calculate an equivalent of the PPTA teststat
    """
    #tstat_num = []
    #tstat_den = []
    #tstat = []
    auto_power = []
    auto_powererr = []
    for ii, likob in enumerate(likobs):
        psr = likob.ptapsrs[0]

        likob.rGE = rGE(psr, datasets[ii])
        likob.OSr = likob.rGE - np.dot(likob.SEGE.T, likob.rGE)

        freqpy = psr.Ffreqs * pic_spy
        Tmax = likob.Tmax
        pcdoubled = (pic_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-4.33)

        nfreqs = psr.Fmat.shape[1]

        if likob.likfunc[:5] == 'mark4':
            PhiTheta = np.zeros((nfreqs, nfreqs))
        else:
            PhiTheta = np.zeros((len(likob.OSr), len(likob.OSr)))

        di = np.diag_indices(nfreqs)
        PhiTheta[di] = pcdoubled

        if likob.likfunc[:5] == 'mark4':
            #PT = np.dot(np.dot(psr.UtF, PhiTheta), psr.UtF.T)
            PT = np.dot(psr.UtF, np.dot(PhiTheta, psr.UtF.T))
        else:
            PT = PhiTheta
            
        num = np.dot(likob.OSr, np.dot(PT, likob.OSr))
        #den = np.trace(np.dot(likob.OSE, np.dot(PT, \
        #        np.dot(likob.OSE, PT.T))))
        auto_power.append(num / likob.consDen)
        auto_powererr.append(1.0 / np.sqrt(likob.mlDen))

    # np.sum(tstat_num) / np.sum(tstat_den), np.sum(tstat)
    return np.array(auto_power), np.array(auto_powererr)


def statDen(likob):
    """
    Calculate the denominator of the autoPower term. This term is independent of
    the data, and hence only needs to be calculated once
    """
    psr = likob.ptapsrs[0]

    freqpy = psr.Ffreqs * pic_spy
    Tmax = likob.Tmax
    pcdoubled = (pic_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-4.33)

    nfreqs = psr.Fmat.shape[1]

    if likob.likfunc[:5] == 'mark4':
        PhiTheta = np.zeros((nfreqs, nfreqs))
    else:
        PhiTheta = np.zeros((len(likob.OSr), len(likob.OSr)))

    di = np.diag_indices(nfreqs)
    PhiTheta[di] = pcdoubled

    if likob.likfunc[:5] == 'mark4':
        #PT = np.dot(np.dot(psr.UtF, PhiTheta), psr.UtF.T)
        PT = np.dot(psr.UtF, np.dot(PhiTheta, psr.UtF.T))
    else:
        PT = PhiTheta
        
    return np.trace(np.dot(likob.OSE, np.dot(PT, \
            np.dot(likob.OSE, PT.T))))


def testStat(likobs, datasets):
    """
    Perform a least-squares fit to the GW amplitude, based on the auto-power of
    the data
    """
    auto_power, auto_powererr = autoPower(likobs, datasets)

    hc_sqr = np.sum(auto_power / (auto_powererr**2)) / \
            np.sum(1.0 / (auto_powererr**2))

    hc_sqrerr = 1.0 / np.sqrt(np.sum(1.0 / (auto_powererr**2)))

    return hc_sqr, hc_sqrerr


def teststatBound(likobs, mlpars, bins=20, N=400, loggwmin=-15.3, \
                  loggwmax=-14.3, mlNoise=True):
    """
    Do the frequencies auto-term upper-bound
    """
    for ii, likob in enumerate(likobs):
        #prepPsrForTeststat(likob, mlpars[ii], mlNoise=False)
        prepPsrForTeststat(likob, mlpars[ii], mlNoise=mlNoise)
        likob.consDen = statDen(likob)
        prepPsrForTeststat(likob, mlpars[ii], mlNoise=mlNoise)
        likob.mlDen = statDen(likob)

    logamps = np.linspace(loggwmin, loggwmax, bins)
    passtest = np.zeros(bins)

    datasets = [likobs[ii].ptapsrs[0].residuals for ii in range(len(likobs))]

    teststat, teststat_err = testStat(likobs, datasets)

    for ii, loggwamp in enumerate(logamps):
        # Loop over amplitudes (bins)
        gwamp = 10 ** loggwamp

        tstat = np.zeros(N)
        tstat_err = np.zeros(N)
        for jj in range(N):
            # Loop over number of trials

            dataset_test = []
            for ll, likob in enumerate(likobs):
                dataset_test.append(genFakeSet(likob, gwamp))

            tstat[jj], tstat_err[jj] = testStat(likobs, dataset_test)

        passtest[ii] = np.sum(tstat > teststat)

    return logamps, passtest



