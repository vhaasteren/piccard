#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
anisotropy.py

Requirements:
- numpy:        pip install numpy
- matplotlib:   macports, apt-get
- libstempo:    pip install libstempo (optional, required for creating HDF5
                files, and for non-linear timing model analysis

Created by vhaasteren on 2013-08-06.
Copyright (c) 2013 Rutger van Haasteren

"""

from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os, glob
import sys
import json
import tempfile
import healpy as hp
import libstempo as lt
import triangle, acor


#from . import sphericalharmonics as ang  # Internal module
import sphericalharmonics as ang  # Internal module (actually called anisotropygammas in Piccard)
                                  # Should be replaced with this pixel/updated
                                  # Sph work.



# Some constants used in Piccard
# For DM calculations, use this constant
# See You et al. (2007) - http://arxiv.org/abs/astro-ph/0702366
# Lee et al. (in prep.) - ...
# Units here are such that delay = DMk * DM * freq^-2 with freq in MHz
pic_DMk = 4.15e3        # Units MHz^2 cm^3 pc sec

pic_spd = 86400.0       # Seconds per day
pic_spy =  31557600.0   # Seconds per year (yr = 365.25 days, so Julian years)
pic_T0 = 53000.0        # MJD to which all HDF5 toas are referenced
pic_pc = 3.08567758e16  # Parsec in meters
pic_c = 299792458     # Speed of light in m/s



def real_sph_harm(mm, ll, phi, theta):
    """
    The real-valued spherical harmonics
    """
    if mm>0:
        ans = (1./math.sqrt(2)) * \
                (ss.sph_harm(mm, ll, phi, theta) + \
                ((-1)**mm) * ss.sph_harm(-mm, ll, phi, theta))
    elif mm==0:
        ans = ss.sph_harm(0, ll, phi, theta)
    elif mm<0:
        ans = (1./(math.sqrt(2)*complex(0.,1))) * \
                (ss.sph_harm(-mm, ll, phi, theta) - \
                ((-1)**mm) * ss.sph_harm(mm, ll, phi, theta))

    return ans.real

def signalResponse(ptapsrs, gwtheta, gwphi):
    """
    Create the signal response matrix
    """
    psrpos_phi = np.array([ptapsrs[ii].raj for ii in range(len(ptapsrs))])
    psrpos_theta = np.array([np.pi/2.0 - ptapsrs[ii].decj for ii in range(len(ptapsrs))])

    return signalResponse_fast(psrpos_theta, psrpos_phi, gwtheta, gwphi)


def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a):
    """
    Create the signal response matrix FAST
    """
    npsrs = len(ptheta_a)

    # Create a meshgrid for both phi and theta directions
    gwphi, pphi = np.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = np.meshgrid(gwtheta_a, ptheta_a)

    return createSignalResponse(pphi, ptheta, gwphi, gwtheta)


def createSignalResponse(pphi, ptheta, gwphi, gwtheta):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW location
    @param gwtheta: Theta of GW location

    @return:    Signal response matrix of Earth-term

    """
    Fp = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True)
    Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=False)

    F = np.zeros((Fp.shape[0], 2*Fp.shape[1]))
    F[:, 0::2] = Fp
    F[:, 1::2] = Fc

    return F

def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True, norm=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW location
    @param gwtheta: Theta of GW location
    @param plus:    Whether or not this is the plus-polarization

    @return:    Signal response matrix of Earth-term
    """
    # Create the direction vectors. First dimension will be collapsed later
    Omega = np.array([-np.sin(gwtheta)*np.cos(gwphi), \
                      -np.sin(gwtheta)*np.sin(gwphi), \
                      -np.cos(gwtheta)])
    
    mhat = np.array([-np.sin(gwphi), np.cos(gwphi), np.zeros(gwphi.shape)])
    nhat = np.array([-np.cos(gwphi)*np.cos(gwtheta), \
                     -np.cos(gwtheta)*np.sin(gwphi), \
                     np.sin(gwtheta)])

    p = np.array([np.cos(pphi)*np.sin(ptheta), \
                  np.sin(pphi)*np.sin(ptheta), \
                  np.cos(ptheta)])
    
    # There is a factor of 3/2 difference between the Hellings & Downs
    # integral, and the one presented in Jenet et al. (2005; also used by Gair
    # et al. 2014). This factor 'normalises' the correlation matrix, but I don't
    # see why I have to pull this out of my ass here. My antennae patterns are
    # correct, so does this mean our strain amplitude is re-scaled. Check this.
    npixels = Omega.shape[2]
    if norm:
        # Add extra factor of 3/2
        c = np.sqrt(1.5) / np.sqrt(npixels)
    else:
        c = 1.0 / np.sqrt(npixels)

    # Calculate the Fplus or Fcross antenna pattern. Definitions as in Gair et
    # al. (2014), with right-handed coordinate system
    if plus:
        # The sum over axis=0 represents an inner-product
        Fsig = 0.5 * c * (np.sum(nhat * p, axis=0)**2 - np.sum(mhat * p, axis=0)**2) / \
                (1 + np.sum(Omega * p, axis=0))
    else:
        # The sum over axis=0 represents an inner-product
        Fsig = c * np.sum(mhat * p, axis=0) * np.sum(nhat * p, axis=0) / \
                (1 + np.sum(Omega * p, axis=0))

    return Fsig



def almFromClm(clm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function just
    takes the imaginary part of the abs(m) alm index.
    """
    maxl = int(np.sqrt(len(clm)))-1
    nclm = len(clm)

    # Construct alm from clm
    nalm = hp.Alm.getsize(maxl)
    alm = np.zeros((nalm), dtype=np.complex128)

    clmindex = 0
    for ll in range(0, maxl+1):
        for mm in range(-ll, ll+1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))
            
            if mm == 0:
                alm[almindex] += clm[clmindex]
            elif mm < 0:
                alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
            elif mm > 0:
                alm[almindex] += clm[clmindex] / np.sqrt(2)
            
            clmindex += 1
    
    return alm


def clmFromAlm(alm):
    """
    Given an array of clm values, return an array of complex alm valuex

    Note: There is a bug in healpy for the negative m values. This function just
    takes the imaginary part of the abs(m) alm index.
    """
    nalm = len(alm)
    maxl = int(np.sqrt(9.0 - 4.0 * (2.0-2.0*nalm))*0.5 - 1.5)
    nclm = (maxl+1)**2

    # Check the solution
    if nalm != int(0.5 * (maxl+1) * (maxl+2)):
        raise ValueError("Check numerical precision. This should not happen")

    clm = np.zeros(nclm)

    clmindex = 0
    for ll in range(0, maxl+1):
        for mm in range(-ll, ll+1):
            almindex = hp.Alm.getidx(maxl, ll, abs(mm))
            
            if mm == 0:
                #alm[almindex] += clm[clmindex]
                clm[clmindex] = alm[almindex].real
            elif mm < 0:
                #alm[almindex] -= 1j * clm[clmindex] / np.sqrt(2)
                clm[clmindex] = - alm[almindex].imag * np.sqrt(2)
            elif mm > 0:
                #alm[almindex] += clm[clmindex] / np.sqrt(2)
                clm[clmindex] = alm[almindex].real * np.sqrt(2)
            
            clmindex += 1
    
    return clm



def mapFromClm_fast(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation

    return:     Healpix pixels

    Use Healpix spherical harmonics for computational efficiency
    """
    maxl = int(np.sqrt(len(clm)))-1
    alm = almFromClm(clm)

    h = hp.alm2map(alm, nside, maxl, verbose=False)

    return h

def mapFromClm(clm, nside):
    """
    Given an array of C_{lm} values, produce a pixel-power-map (non-Nested) for
    healpix pixelation with nside

    @param clm:     Array of C_{lm} values (inc. 0,0 element)
    @param nside:   Nside of the healpix pixelation

    return:     Healpix pixels
    """
    npixels = hp.nside2npix(nside)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    
    h = np.zeros(npixels)

    ind = 0
    maxl = int(np.sqrt(len(clm)))-1
    for ll in range(maxl+1):
        for mm in range(-ll, ll+1):
            h += clm[ind] * real_sph_harm(mm, ll, pixels[1], pixels[0])
            ind += 1

    return h


def clmFromMap_fast(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    @param h:       Sky power map
    @param lmax:    Up to which order we'll be expanding

    return: clm values

    Use Healpix spherical harmonics for computational efficiency
    """
    alm = hp.sphtfunc.map2alm(h, lmax=lmax)
    alm[0] = np.sum(h) * np.sqrt(4*np.pi) / len(h)

    return clmFromAlm(alm)


def clmFromMap(h, lmax):
    """
    Given a pixel map, and a maximum l-value, return the corresponding C_{lm}
    values.

    @param h:       Sky power map
    @param lmax:    Up to which order we'll be expanding

    return: clm values
    """
    npixels = len(h)
    nside = hp.npix2nside(npixels)
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    
    clm = np.zeros( (lmax+1)**2 )
    
    ind = 0
    for ll in range(lmax+1):
        for mm in range(-ll, ll+1):
            clm[ind] += np.sum(h * real_sph_harm(mm, ll, pixels[1], pixels[0]))
            ind += 1
            
    return clm * 4 * np.pi / npixels



def getCov(clm, nside, F_e):
    """
    Given a vector of clm values, construct the covariance matrix
    """
    # Create a sky-map (power)
    # Use mapFromClm to compare to real_sph_harm. Fast uses Healpix
    sh00 = mapFromClm_fast(clm, nside)

    # Double the power (one for each polarization)
    sh = np.array([sh00, sh00]).T.flatten()

    # Create the cross-pulsar covariance
    hdcov_F = np.dot(F_e * sh, F_e.T)

    # The pulsar term is added (only diagonals: uncorrelated)
    return hdcov_F + np.diag(np.diag(hdcov_F))



def SH_CorrBasis(psr_locs, lmax, nside=32):
    """
    Calculate the correlation basis matrices using the pixel-space
    transormations

    @param psr_locs:    Location of the pulsars [phi, theta]
    @param lmax:        Maximum l to go up to
    @param nside:       What nside to use in the pixelation [32]
    """
    npsrs = len(psr_locs)
    pphi = psr_locs[:,0]
    ptheta = psr_locs[:,1]

    # Create the pixels
    npixels = hp.nside2npix(nside)    # number of pixels total
    pixels = hp.pix2ang(nside, np.arange(npixels), nest=False)
    gwtheta = pixels[0]
    gwphi = pixels[1]

    # Create the signal response matrix
    F_e = signalResponse_fast(ptheta, pphi, gwtheta, gwphi)

    # Loop over all (l,m)
    basis = []
    nclm = (lmax+1)**2
    clmindex = 0
    for ll in range(0, lmax+1):
        for mm in range(-ll, ll+1):
            clm = np.zeros(nclm)
            clm[clmindex] = 1.0

            basis.append(getCov(clm, nside, F_e))
            clmindex += 1

    return basis











def fourierdesignmatrix(t, nmodes, Ttot=None):
    """
    Calculate the matrix of Fourier modes A, given a set of timestamps

    These are sine/cosine basis vectors at evenly separated frequency bins

    Mode 0: sin(f_0)
    Mode 1: cos(f_0)
    Mode 2: sin(f_1)
    ... etc

    @param nmodes:  The number of modes that will be included (= 2*nfreq)
    @param Ttot:    Total duration experiment (in case not given by t)

    @return: (A, freqs), with A the 'fourier design matrix', and f the associa
    """
    N = t.size
    A = np.zeros([N, nmodes])
    T = t.max() - t.min()

    if(nmodes % 2 != 0):
      print "WARNING: Number of modes should be even!"

    # The frequency steps
    #deltaf = (N-1.0) / (N*T)    # This would be orthogonal for regular sampling

    if Ttot is None:
        deltaf = 1.0 / T
    else:
        deltaf = 1.0 / Ttot

    freqs1 = np.linspace(deltaf, (nmodes/2)*deltaf, nmodes/2)
    freqs = np.array([freqs1, freqs1]).T.flatten()

    # The cosine modes
    for i in range(0, nmodes, 2):
        omega = 2.0 * np.pi * freqs[i]
        A[:,i] = np.cos(omega * t)

    # The sine modes
    for i in range(1, nmodes, 2):
        omega = 2.0 * np.pi * freqs[i]
        A[:,i] = np.sin(omega * t)

    # This normalisation would make F unitary in the case of regular sampling
    # A = A * np.sqrt(2.0/N)

    return (A, freqs)


def designqsd(t):
    """
    Calculate the design matrix for quadratic spindown

    @param t: array of toas
    """
    M = np.ones([len(t), 3])
    M[:,1] = t
    M[:,2] = t ** 2
    
    return M.copy()




# Very simple pulsar class
class ptaPulsar(object):
    def __init__(self, raj, decj, name, toas, residuals, toaerrs, dist=1.0, \
            nfreqs=10):
        self.raj = raj
        self.decj = decj
        self.name = name
        self.toas = toas
        self.residuals = residuals
        self.toaerrs = toaerrs
        self.pos = np.array([np.cos(self.decj)*np.cos(self.raj),
                              np.cos(self.decj)*np.sin(self.raj),
                              np.sin(self.decj)])
        self.dist = dist

        self.T = (np.max(self.toas) - np.min(self.toas)) * pic_spd

        self.Ft, self.freqs = fourierdesignmatrix(self.toas * pic_spd, 2*nfreqs)

    
    def getRMS(self, useResiduals=False):
        """
        Calculate the weighted RMS
        
        @param useResiduals:    If true, use the residuals to calculate the RMS.
                                Otherwise, weigh the errorbars
        """
        RMS = 0
        if useResiduals:
            W = 1.0 / self.toaerrs**2
            SPS = np.sum(self.residuals**2 * W)
            SP = np.sum(self.residuals * W)
            SW = np.sum(W)
            
            chi2 = (SPS - SP*SP/SW)
            RMS = np.sqrt( (SPS - SP*SP/SW)/SW )
        else:
            RMS = np.sqrt(len(self.toas) / np.sum(1.0 / self.toaerrs**2))
            
        return RMS

def readArray(partimdir, mindist=0.5, maxdist=2.0):
    """
    Read in a list of ptaPulsar objects from a set of par/tim files. Pulsar
    distances are randomly drawn between two values

    @param partimdir:   Directory of par/tim files
    @param mindist:     Minimum distance of pulsar
    @param maxdist:     Maximum distance of pulsar

    @return: list of ptapsrs
    """
    ptapsrs = []
    curdir = os.getcwd()
    os.chdir(partimdir)
    for ii, infile in enumerate(glob.glob(os.path.join('./', 'J*.par') )):
        filename = os.path.splitext(infile)
        basename = os.path.basename(filename[0])
        parfile = './' + basename +'.par'
        timfile = './' + basename +'.tim'
        
        psr = lt.tempopulsar(parfile, timfile, dofit=False)

        dist = mindist + np.random.rand(1) * (maxdist - mindist)

        ptapsrs.append(ptaPulsar(psr['RAJ'].val, psr['DECJ'].val, psr.name, \
                                 psr.toas(), psr.residuals(), \
                                 psr.toaerrs*1.0e-6, 1000*dist))
    
    os.chdir(curdir)

    return ptapsrs


def genArray(npsrs=20, Ntoas=500, toaerr=1.0e-7, T=315576000.0, mindist=0.5, maxdist=2.0):
    """
    Generate a set of pulsars

    @param npsrs:   Number of pulsars
    @param Ntoas:   Number of observations per pulsar
    @param toaerr:  TOA uncertainty (sec.)
    @param T:       Length dataset (sec.)
    @param mindist: Minimum distance of pulsar (kpc)
    @param maxdist: Maximum distance of pulsar (kpc)

    @return: list of ptapsrs
    """
    ptapsrs = []
    for ii in range(npsrs):
        # Make a pulsar
        name = "Pulsar" + str(ii)
        loc = [np.random.rand(1)[0] * np.pi*2, np.arccos(2*np.random.rand(1)[0]-1)]
        toas = np.linspace(0, T, Ntoas)
        toaerrs = np.ones(Ntoas) * toaerr
        residuals = np.random.randn(Ntoas) * toaerr

        dist = mindist + np.random.rand(1) * (maxdist - mindist)
        
        ptapsrs.append(ptaPulsar(loc[0], np.pi/2.0 - loc[1], name, toas / pic_spd,\
                residuals, toaerrs, 1000*dist))

    return ptapsrs





"""
with n the number of pulsars, return an nxn matrix representing the H&D
correlation matrix
"""
def hdcorrmat(ptapsrs, psrTerm=True):
    """ Constructs a correlation matrix consisting of the Hellings & Downs
        correlation coefficients. See Eq. (A30) of Lee, Jenet, and
        Price ApJ 684:1304 (2008) for details.

        @param: list of ptaPulsar (or any other markXPulsar) objects
        
    """
    npsrs = len(ptapsrs)
    
    raj = [ptapsrs[i].raj for i in range(npsrs)]
    decj = [ptapsrs[i].decj for i in range(npsrs)]
    pp = np.array([np.cos(decj)*np.cos(raj), np.cos(decj)*np.sin(raj), np.sin(decj)]).T
    cosp = np.array([[np.dot(pp[i], pp[j]) for i in range(npsrs)] for j in range(npsrs)])
    cosp[cosp > 1.0] = 1.0
    xp = 0.5 * (1 - cosp)

    old_settings = np.seterr(all='ignore')
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old_settings)

    if psrTerm:
        coeff = 1.0
    else:
        coeff = 0.0
    hdmat = logxp - 0.25 * xp + 0.5 + coeff * 0.5 * np.diag(np.ones(npsrs))

    return hdmat


"""
with n the number of pulsars, return an nxn matrix representing the dipole
(ephemeris) correlation matrix
"""
def dipolecorrmat(ptapsrs):
    """ Constructs a correlation matrix consisting of simple dipole correlations
    """
    npsrs = len(ptapsrs)
    
    raj = [ptapsrs[i].raj[0] for i in range(npsrs)]
    decj = [ptapsrs[i].decj[0] for i in range(npsrs)]
    pp = np.array([np.cos(decj)*np.cos(raj), np.cos(decj)*np.sin(raj), np.sin(decj)]).T
    cosp = np.array([[np.dot(pp[i], pp[j]) for i in range(npsrs)] for j in range(npsrs)])

    cosp[cosp > 1.0] = 1.0

    return cosp


# The GWB general anisotropic correlations as defined in
# Mingarelli and Vecchio (submitted); Taylor and Gair (submitted)
class aniCorrelations(object):
    def __init__(self, psrs=None, l=1):
        self.phiarr = None           # The phi pulsar position parameters
        self.thetaarr = None         # The theta pulsar position parameters
        self.gamma_ml = None         # The gamma_ml (see anisotropygammas.py)

        self.priorNside = 8
        self.priorNpix = 8
        self.priorPix = None
        self.SpHmat = None
        #self.priorgridbins = 16
        #self.priorphi = None
        #self.priortheta = None

        self.corrhd = None
        self.corr = []

        if psrs != None:
            # If we have a pulsars object, initialise the angular quantities
            self.setmatrices(psrs, l)

    def clmlength(self):
        return (self.l+1)**2-1

    def setmatrices(self, psrs, l):
        # First set all the pulsar positions
        self.phiarr = np.zeros(len(psrs))
        self.thetaarr = np.zeros(len(psrs))
        self.l = l

        for ii in range(len(psrs)):
            self.phiarr[ii] = psrs[ii].raj
            self.thetaarr[ii] = np.pi/2 - psrs[ii].decj

        # Create the prior-grid pixels
        self.priorNside = 8
        self.priorNpix = hp.nside2npix(self.priorNside)
        self.priorPix = hp.pix2ang(self.priorNside, \
                np.arange(self.priorNpix), nest=False)

        self.SpHmat = np.zeros((self.priorNpix, self.clmlength()))
        for ii in range(self.priorNpix):
            cindex = 0
            for ll in range(1, self.l+1):
                for mm in range(-ll, ll+1):
                    self.SpHmat[ii, cindex] = \
                            real_sph_harm(mm, ll, \
                            self.priorPix[1][ii], \
                            self.priorPix[0][ii])
                    cindex += 1


        self.corrhd = hdcorrmat(psrs)

        for ll in range(1, self.l+1):
            mmodes = 2*ll+1     # Number of modes for this ll

            # Create the correlation matrices for this value of l
            for mm in range(mmodes):
                self.corr.append(np.zeros((len(psrs), len(psrs))))

            for aa in range(len(psrs)):
                for bb in range(aa, len(psrs)):
                    plus_gamma_ml = []  # gammas for this pulsar pair
                    neg_gamma_ml = []
                    gamma_ml = []
                    for mm in range(ll+1):
                        intg_gamma = ang.int_Gamma_lm(mm, ll, \
                                self.phiarr[aa], self.phiarr[bb], \
                                self.thetaarr[aa],self.thetaarr[bb])


                        neg_intg_gamma= (-1)**(mm) * intg_gamma  # (-1)^m Gamma_ml
                        plus_gamma_ml.append(intg_gamma)     # all gammas
                        neg_gamma_ml.append(neg_intg_gamma)  # neg m gammas

                    neg_gamma_ml = neg_gamma_ml[1:]          # Use 0 only once
                    rev_neg_gamma_ml = neg_gamma_ml[::-1]    # Reverse list direction
                    gamma_ml = rev_neg_gamma_ml+plus_gamma_ml

                    # Fill the corrcur matrices for all m
                    mindex = len(self.corr) - mmodes    # Index first m mode
                    for mm in range(mmodes):
                        m = mm - ll

                        self.corr[mindex+mm][aa, bb] = \
                                ang.real_rotated_Gammas(m, ll, \
                                self.phiarr[aa], self.phiarr[bb], \
                                self.thetaarr[aa], self.thetaarr[bb], gamma_ml)

                        if aa != bb:
                            self.corr[mindex+mm][bb, aa] = self.corr[mindex+mm][aa, bb]


    def priorIndicator(self, clm):
        # Check whether sum_lm c_lm * Y_lm > 0 for this combination of clm
        if self.priorPix == None or self.SpHmat == None:
            raise ValueError("ERROR: first define the anisotropic prior-check positions")

        # Number of clm is 3 + 5 + 7 + ... (2*self.l+1)
        if len(clm) != self.clmlength():
            print "len(clm) = ", len(clm), "clmlength = ", self.clmlength()
            raise ValueError("ERROR: len(clm) != clmlength")

        clmYlm = clm * self.SpHmat
        S = np.sum(clmYlm, axis=1) + 1.0

        return np.all(S > 0.0)

    # Return the full correlation matrix that depends on the clm. This
    # correlation matrix only needs to be multiplied with the signal amplitude
    # and the time-correlations
    def corrmat(self, clm):
        # Number of clm is 3 + 5 + 7 + ... (2*self.l+1)
        if len(clm) != self.clmlength():
            raise ValueError("ERROR: len(clm) != clmlength")

        corrreturn = self.corrhd.copy()
        """
        np.savetxt('corrmat_0_0.txt', corrreturn)
        """
        index = 0
        for ll in range(1, self.l+1):
            for mm in range(-ll, ll+1):
                corrreturn += clm[index] * self.corr[index]

                """
                if clm[index] != 0:
                    print "\nIndex = " + str(index) + "   l, m = " + str(ll) + ',' + str(mm)
                    print "clm[index] = " + str(clm[index])
                """

                """
                # Write the matrices to file
                filename = 'corrmat_' + str(ll) + '_' + str(mm) + '.txt'
                np.savetxt(filename, self.corr[index])
                print "Just saved '" + filename + "'"
                """

                index += 1

        return corrreturn




"""
Function that calculates the earth-term gravitational-wave burst-with-memory
signal, as described in:
Seto et al, van haasteren and Levin, phsirkov et al, Cordes and Jenet.

parameter[0] = TOA time (sec) the burst hits the earth
parameter[1] = amplitude of the burst (strain h)
parameter[2] = azimuthal angle (rad)
parameter[3] = polar angle (rad)
parameter[4] = polarisation angle (rad)

raj = Right Ascension of the pulsar (rad)
decj = Declination of the pulsar (rad)
t = timestamps where the waveform should be returned

returns the waveform as induced timing residuals (seconds)

"""
def bwmsignal(parameters, raj, decj, t):
    # The rotation matrices
    rot1 = np.eye(3)
    rot2 = np.eye(3)
    rot3 = np.eye(3)

    # Rotation along the azimuthal angle (raj source)
    rot1[0,0] = np.cos(parameters[2])   ; rot1[0,1] = np.sin(parameters[2])
    rot1[1,0] = -np.sin(parameters[2])  ; rot1[1,1] = np.cos(parameters[2])

    # Rotation along the polar angle (decj source)
    rot2[0,0] = np.sin(parameters[3])   ; rot2[0,2] = -np.cos(parameters[3])
    rot2[2,0] = np.cos(parameters[3])   ; rot2[2,2] = np.sin(parameters[3])

    # Rotate the bwm polarisation to match the x-direction
    rot3[0,0] = np.cos(parameters[4])   ; rot3[0,1] = np.sin(parameters[4])
    rot3[1,0] = -np.sin(parameters[4])  ; rot3[1,1] = np.cos(parameters[4])

    # The total rotation matrix
    rot = np.dot(rot1, np.dot(rot2, rot3))

    # The pulsar position in Euclidian coordinates
    ppos = np.zeros(3)
    ppos[0] = np.cos(raj) * np.cos(decj)
    ppos[1] = np.sin(raj) * np.cos(decj)
    ppos[2] = np.sin(decj)

    # Rotate the position of the pulsar
    ppr = np.dot(rot, ppos)

    # Antenna pattern
    ap = 0.0
    if np.abs(ppr[2]) < 1:
        # Depending on definition of source position, it could be (1 - ppr[2])
        ap = 0.5 * (1 + ppr[2]) * (2 * ppr[0] * ppr[0] / (1 - ppr[2]*ppr[2]) - 1)
        
        2 * ppr[0] * ppr[0] 

    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    # Return the time series
    return ap * (10**parameters[1]) * heaviside(t - parameters[0]) * (t - parameters[0])



def genWhite(ptapsrs):
    """
    Generate white residuals, according to the TOA uncertainties
    """

    for ii, psr in enumerate(ptapsrs):
        psr.residuals = np.random.randn(nobs)*psr.toaerrs

def addSignal(ptapsrs, sig, reset=False):
    """
    Add the signal to the residuals
    """
    for ii, psr in enumerate(ptapsrs):
        if reset:
            psr.residuals = np.random.randn(len(psr.residuals)) * psr.toaerrs

        psr.residuals += sig[:, ii]


# Generation of gravitational waves (white spectrum)
def genGWB_white(ptapsrs, amp, Si=4.33, Fmat_e=None, Fmat_p=None):
    """
    Returns a signal, in the form of a npsr x ntoas matrix, not yet added to the
    residuals

    @param ptapsrs: The list with pulsars
    @param amp:     GW amplitude
    @param Si:      Spectral index
    @param Fmat_e:  Earth-term signal response
    @param Fmat_p:  Pulsar-term signal response
    """

    # Remember, this is the decomposition: C = U * D * U.T
    npsrs = len(ptapsrs)
    nmodes = ptapsrs[0].Ft.shape[1]
    nobs = len(ptapsrs[0].toas)

    # We really use signal response matrices
    if Fmat_e is None:
        raise ValueError("No signal resonse given")
    
    if Fmat_p is not None:
        psrTerm = True
        F = Fmat_e + Fmat_p
    else:
        psrTerm = False
        F = Fmat_e
    
    # Do a thin SVD
    U, s, Vt = sl.svd(F, full_matrices=False)

    # Generate the mode data
    #xi_t = np.random.randn(nmodes)          # Time-correlations
    #xi_c = np.random.randn(npsrs)           # Spatial-correlations
    xi_full = np.random.randn(npsrs, nobs)
    sig_full = np.dot(U, (s*xi_full.T).T) * amp

    return sig_full.T


# Generation of gravitational waves (white spectrum)
def genGWB_fromcov_white(ptapsrs, amp, Si=4.33, cov=None):
    """
    Returns a signal, in the form of a npsr x ntoas matrix, not yet added to the
    residuals

    @param ptapsrs: The list with pulsars
    @param amp:     GW amplitude
    @param Si:      Spectral index
    @param cov:     The covariance matrix we will generate from
    """

    # Remember, this is the decomposition: C = U * D * U.T
    npsrs = len(ptapsrs)
    nmodes = ptapsrs[0].Ft.shape[1]
    nobs = len(ptapsrs[0].toas)

    # We really do need the covariance matrix
    if cov is None:
        raise ValueError("No covariance matrix given")

    # Cholesky factor of the correlations
    cf = sl.cholesky(cov, lower=True)

    # Generate the mode data
    #xi_t = np.random.randn(nmodes)          # Time-correlations
    #xi_c = np.random.randn(npsrs)           # Spatial-correlations
    xi_full = np.random.randn(npsrs, nobs)
    sig_full = np.dot(cf, xi_full) * amp

    return sig_full.T





# Generation of gravitational waves
def genGWB_red(ptapsrs, amp, Si=4.33, Fmat_e=None, Fmat_p=None):
    """
    Returns a signal, in the form of a npsr x ntoas matrix, not yet added to the
    residuals

    @param ptapsrs: The list with pulsars
    @param amp:     GW amplitude
    @param Si:      Spectral index
    @param Fmat_e:  Earth-term signal response
    @param Fmat_p:  Pulsar-term signal response
    """

    # Remember, this is the decomposition: C = U * D * U.T
    npsrs = len(ptapsrs)
    nmodes = ptapsrs[0].Ft.shape[1]
    nobs = len(ptapsrs[0].toas)

    # We really use signal response matrices
    if Fmat_e is None:
        raise ValueError("No signal resonse given")
    
    if Fmat_p is not None:
        psrTerm = True
        F = Fmat_e + Fmat_p
    else:
        psrTerm = False
        F = Fmat_e
    
    # Do a thin SVD
    U, s, Vt = sl.svd(F, full_matrices=False)

    # Generate the mode data
    xi_t = np.random.randn(nmodes)          # Time-correlations
    xi_c = np.random.randn(npsrs)           # Spatial-correlations

    # Generate spatial correlations
    s_c = np.dot(U, (s*xi_c.T).T)

    # Generate the residuals
    sig = np.zeros((nobs, npsrs))
    psds = np.zeros((nmodes, npsrs))

    for ii, psr in enumerate(ptapsrs):
        # Generate in frequency domain, and transform to time domain
        freqpy = psr.freqs * pic_spy
        psd = (amp**2 * pic_spy**3 / (12*np.pi*np.pi * psr.T)) * freqpy ** (-Si)

        # Generate the time-correlations
        s_t = np.sqrt(psd) * xi_t

        # Generate the signal
        psds[:, ii] = s_c[ii] * s_t
        sig[:, ii] = np.dot(psr.Ft, psds[:, ii])

    return (sig, psds, s_c)


# Generation of gravitational waves
def genGWB_fromcov_red(ptapsrs, amp, Si=4.33, cov=None):
    """
    Returns a signal, in the form of a npsr x ntoas matrix, not yet added to the
    residuals

    @param ptapsrs: The list with pulsars
    @param amp:     GW amplitude
    @param Si:      Spectral index
    @param cov:     The covariance matrix we will generate from
    """

    # Remember, this is the decomposition: C = U * D * U.T
    npsrs = len(ptapsrs)
    nmodes = ptapsrs[0].Ft.shape[1]
    nobs = len(ptapsrs[0].toas)

    # We really do need the covariance matrix
    if cov is None:
        raise ValueError("No covariance matrix given")

    # Cholesky factor of the correlations
    cf = sl.cholesky(cov, lower=True)

    # Generate the mode data
    xi_t = np.random.randn(nmodes)          # Time-correlations
    xi_c = np.random.randn(npsrs)           # Spatial-correlations

    #p = np.dot(U, (s*xi.T).T)
    s_c = np.dot(cf, xi_c)

    # Generate the residuals
    sig = np.zeros((nobs, npsrs))
    psds = np.zeros((nmodes, npsrs))

    for ii, psr in enumerate(ptapsrs):
        # Generate in frequency domain, and transform to time domain
        freqpy = psr.freqs * pic_spy
        psd = (amp**2 * pic_spy**3 / (12*np.pi*np.pi * psr.T)) * freqpy ** (-Si)

        # Generate the time-correlations
        s_t = np.sqrt(psd) * xi_t

        # Generate the signal
        psds[:, ii] = s_c[ii] * s_t

        sig[:, ii] = np.dot(psr.Ft, psds[:, ii])

    return (sig, psds, s_c)



def crossPower(ptapsrs):
    """
    Calculate the cross-power according to the Jenet/Demorest method
    (Noise spectra are now diagonal, so it's rather quick)
    """
    npsrs = len(ptapsrs)
    pairs = int(npsrs * (npsrs-1) * 0.5)
    
    angle = np.zeros(pairs)
    crosspower = np.zeros(pairs)
    crosspowererr = np.zeros(pairs)
    hdcoeff = np.zeros(pairs)
    
    ii = 0
    for aa in range(npsrs):
        psra = ptapsrs[aa]
        for bb in range(aa+1, npsrs):
            psrb = ptapsrs[bb]
            angle[ii] = np.arccos(np.sum(psra.pos * psrb.pos))
            xp = 0.5 * (1 - np.sum(psra.pos * psrb.pos))
            logxp = 1.5 * xp * np.log(xp)
            hdcoeff[ii] = logxp - 0.25 * xp + 0.5
            
            # Create 'optimal statistic' (the errs are not necessary for now)
            num = np.sum(psra.residuals * psrb.residuals / \
                         (psra.toaerrs * psrb.toaerrs)**2)
            den = np.sum(1.0 / (psra.toaerrs*psrb.toaerrs)**2)
        
            # Crosspower and uncertainty
            crosspower[ii] = num / den
            crosspowererr[ii] = 1.0 / np.sqrt(den)

            ii += 1
            
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
















# The following likelihoods are for too simplistic models, but with red signals.
# Don't use:
def mark1loglikelihood_red(pars, ptapsrs, Usig):
    """
    @par pars:      Parameters: GW amplitude, Spectral index, mode amps
    @par ptapsrs:   List of all PTA pulsars
    @par Usig:      Re-scaled range matrix
    
    @return:    log-likelihood
    """
    npsrs = len(ptapsrs)
    nobs = len(ptapsrs[0].toas)
    nmodes = len(ptapsrs[0].freqs)
    
    iia = 0
    iib = iia + 2
    iic = iib + npsrs
    iid = iic + nmodes
    
    Amp = 10**pars[0]
    Si = pars[1]
    r = pars[iib:iic]       # Range amplitudes (isotropic for now)
    a = pars[iic:iid]       # Fourier modes of GW
    
    # Find the correlation coefficients
    c = np.dot(Usig, r)
    
    xi2 = 0.0
    ld = 0.0
    for ii, psr in enumerate(ptapsrs):
        xred = psr.residuals - c[ii] * np.dot(psr.Ft, a)
        
        xi2 -= 0.5 * np.sum(xred**2 / psr.toaerrs**2)
        ld -= 0.5 * np.sum(np.log(psr.toaerrs**2))
    
    # Now the prior
    freqpy = ptapsrs[0].freqs * pic_spy
    psd = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * ptapsrs[0].T)) * freqpy ** (-Si)
    
    # Subtract the correlations, too
    ld -= 0.5 * np.sum(c**2)

    xi2 -= 0.5 * np.sum(a**2 / psd)
    ld -= 0.5 * np.sum(np.log(psd))
        
    return xi2 + ld

def mark2loglikelihood_red(pars, ptapsrs, corrs):
    """
    @par pars:      Parameters: GW amplitude, Spectral index, mode amps
    @par ptapsrs:   List of all PTA pulsars
    @par Usig:      Re-scaled range matrix
    
    @return:    log-likelihood
    
    This one uses the Clm values
    """
    npsrs = len(ptapsrs)
    nobs = len(ptapsrs[0].toas)
    nmodes = len(ptapsrs[0].freqs)
    nclm = len(pars) - 2 #- nmodes
    nfreqs = nmodes / 2
    lmax = np.sqrt(nclm+1)-1
    
    iia = 0
    iib = iia + 2
    iic = iib + nclm
    #iid = iic + nmodes
    
    Amp = 10**pars[0]
    Si = pars[1]
    clm = pars[iib:iic]     # Anisotropy parameters
    #a = pars[iic:iid]       # Fourier modes of GW
    
    # Calculate the covariance matrix
    cov = corrs.corrmat(clm)
    try:
        ccf = sl.cho_factor(cov)
    except np.linalg.LinAlgError:
        return -np.inf
    cld = 2*np.sum(np.log(np.diag(ccf[0])))
    covinv = sl.cho_solve(ccf, np.eye(npsrs))
    
    # Calculate phi-inv
    phiinv = np.zeros((npsrs*nmodes, npsrs*nmodes))
    Sigma = np.zeros((npsrs*nmodes, npsrs*nmodes))
    for ii in range(0, nmodes, 2):
        sti = ii
        eni = ii+npsrs*nmodes
        stride = nmodes
        phiinv[sti:eni:stride, sti:eni:stride] = covinv
        phiinv[1+sti:eni:stride, 1+sti:eni:stride] = covinv
    
    Sigma = phiinv.copy()
    FNx = np.zeros(npsrs*nmodes)
    xi2 = 0.0
    ld = 0.0
    for ii, psr in enumerate(ptapsrs):
        xred = psr.residuals
        
        xi2 -= 0.5 * np.sum(xred**2 / psr.toaerrs**2)
        ld -= 0.5 * np.sum(np.log(psr.toaerrs**2))
        
        Sigma[ii*nmodes:(ii+1)*nmodes, ii*nmodes:(ii+1)*nmodes] += np.dot(psr.Ft.T / (psr.toaerrs**2), psr.Ft)
        FNx[ii*nmodes:(ii+1)*nmodes] = np.dot(psr.Ft.T, xred/(psr.toaerrs**2))
    
    try:
        scf = sl.cho_factor(Sigma)
    except np.linalg.LinAlgError:
        return -np.inf
    
    sld = 2*np.sum(np.log(np.diag(scf[0])))
    SFNx = sl.cho_solve(scf, FNx)
    xi2 += 0.5 * np.dot(FNx, SFNx)
    ld -= 0.5*npsrs*nmodes*cld
    ld -= 0.5*sld
    
    return xi2 + ld


def logprior_red(amps, amin=None, amax=None):
    retval = 0.0
    if amin is not None:
        if not np.all(amps > amin):
            retval = -np.inf
        
    if amax is not None:
        if not np.all(amps < amax):
            retval = -np.inf
            
    return retval







    
if __name__ == "__main__":
    # Do some stuff
    print "Hello, world!"
