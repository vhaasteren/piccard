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

try:
    import healpy as hp
except:
    hp = None

from constants import *





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

def signalResponse(ptapsrs, gwtheta, gwphi, dirconv=True):
    """
    Create the signal response matrix
    @param dirconv: True when Omega in direction of source (not prop.)
    """
    psrpos_phi = np.array([ptapsrs[ii].raj for ii in range(len(ptapsrs))])
    psrpos_theta = np.array([np.pi/2.0 - ptapsrs[ii].decj for ii in range(len(ptapsrs))])

    return signalResponse_fast(psrpos_theta, psrpos_phi, gwtheta, gwphi, dirconv)


def signalResponse_fast(ptheta_a, pphi_a, gwtheta_a, gwphi_a, dirconv=True):
    """
    Create the signal response matrix FAST
    @param dirconv: True when Omega in direction of source (not prop.)
    """
    npsrs = len(ptheta_a)

    # Create a meshgrid for both phi and theta directions
    gwphi, pphi = np.meshgrid(gwphi_a, pphi_a)
    gwtheta, ptheta = np.meshgrid(gwtheta_a, ptheta_a)

    return createSignalResponse(pphi, ptheta, gwphi, gwtheta, dirconv=dirconv)


def createSignalResponse(pphi, ptheta, gwphi, gwtheta, dirconv=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW location
    @param gwtheta: Theta of GW location
    @param dirconv: True when Omega in direction of source (not prop.)

    @return:    Signal response matrix of Earth-term

    """
    Fp = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True, dirconv=dirconv)
    Fc = createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=False, dirconv=dirconv)

    F = np.zeros((Fp.shape[0], 2*Fp.shape[1]))
    F[:, 0::2] = Fp
    F[:, 1::2] = Fc

    return F

def createSignalResponse_pol(pphi, ptheta, gwphi, gwtheta, plus=True, norm=True,
        dirconv=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param gwphi:   Phi of GW location
    @param gwtheta: Theta of GW location
    @param plus:    Whether or not this is the plus-polarization
    @param dirconv: True when Omega in direction of source (not of propagation)

    @return:    Signal response matrix of Earth-term
    """
    if dirconv:
        dc = 1.0
    else:
        dc = -1.0


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
                (1 + dc*np.sum(Omega * p, axis=0))
    else:
        # The sum over axis=0 represents an inner-product
        Fsig = c * np.sum(mhat * p, axis=0) * np.sum(nhat * p, axis=0) / \
                (1 + dc*np.sum(Omega * p, axis=0))

    return Fsig


def dip_signal_response(ptapsrs, diptheta, dipphi):
    """
    Create the signal response matrix (dipole signal, for ephemeris)
    """
    psrpos_phi = np.array([ptapsrs[ii].raj for ii in range(len(ptapsrs))])
    psrpos_theta = np.array([np.pi/2.0 - ptapsrs[ii].decj for ii in range(len(ptapsrs))])

    return dip_signalResponse_fast(psrpos_theta, psrpos_phi, diptheta, dipphi)


def dip_signalResponse_fast(ptheta_a, pphi_a, diptheta_a, dipphi_a):
    """
    Create the signal response matrix FAST (dipole signal, for ephemeris)
    """
    npsrs = len(ptheta_a)

    # Create a meshgrid for both phi and theta directions
    dipphi, pphi = np.meshgrid(dipphi_a, pphi_a)
    diptheta, ptheta = np.meshgrid(diptheta_a, ptheta_a)

    return dip_createSignalResponse(pphi, ptheta, dipphi, diptheta)


def dip_createSignalResponse(pphi, ptheta, dipphi, diptheta):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality. (dipole signal, for ephemeris)

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param dipphi:   Phi of Dipole location
    @param diptheta: Theta of Dipole location

    @return:    Signal response matrix of Dipole

    """
    F1 = dip_createSignalResponse_pol(pphi, ptheta, dipphi, diptheta, ksi=0.0)
    F2 = dip_createSignalResponse_pol(pphi, ptheta, dipphi, diptheta, ksi=0.5*np.pi)

    F = np.zeros((F1.shape[0], 2*F1.shape[1]))
    F[:, 0::2] = F1
    F[:, 1::2] = F2

    return F


def dip_createSignalResponse_pol(pphi, ptheta, dipphi, diptheta, ksi=0.0, norm=True):
    """
    Create the signal response matrix. All parameters are assumed to be of the
    same dimensionality.

    @param pphi:    Phi of the pulsars
    @param ptheta:  Theta of the pulsars
    @param dipphi:   Phi of Dip location
    @param diptheta: Theta of Dip location
    @param ksi:    Whether this is normalized

    @return:    Signal response matrix of Dipole
    """
    # Create the direction vectors. First dimension will be collapsed later
    Omega = np.array([-np.sin(diptheta)*np.cos(dipphi), \
                      -np.sin(diptheta)*np.sin(dipphi), \
                      -np.cos(diptheta)])

    mhat = np.array([-np.sin(dipphi), np.cos(dipphi), np.zeros(dipphi.shape)])
    nhat = np.array([-np.cos(dipphi)*np.cos(diptheta), \
                     -np.cos(diptheta)*np.sin(dipphi), \
                     np.sin(diptheta)])

    p = np.array([np.cos(pphi)*np.sin(ptheta), \
                  np.sin(pphi)*np.sin(ptheta), \
                  np.cos(ptheta)])

    # Pixel normalization
    npixels = Omega.shape[2]
    c = np.sqrt(1.5) / np.sqrt(npixels)

    # Now the Dipole signal
    Fsig = np.cos(ksi) * np.sum(nhat * p, axis=0) + \
            np.sin(ksi) * np.sum(mhat * p, axis=0)

    return c*Fsig




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



def bwmsignal(parameters, raj, decj, t):
    """
    Function that calculates the earth-term gravitational-wave burst-with-memory
    signal, as described in:
    Seto et al, van haasteren and Levin, phsirkov et al, Cordes and Jenet.

    This version uses the F+/Fx polarization modes, as verified with the
    Continuous Wave and Anisotropy papers. The rotation matrices were not very
    insightful anyway.

    parameter[0] = TOA time (sec) the burst hits the earth
    parameter[1] = amplitude of the burst (strain h)
    parameter[2] = azimuthal angle (rad)    [0, 2pi]
    parameter[3] = polar angle (rad)        [0, pi]
    parameter[4] = polarisation angle (rad) [0, pi]

    raj = Right Ascension of the pulsar (rad)
    decj = Declination of the pulsar (rad)
    t = timestamps where the waveform should be returned

    returns the waveform as induced timing residuals (seconds)

    """
    psrpos_phi = np.array([raj])
    psrpos_theta = np.array([0.5*np.pi-decj])
    gwphi = np.array([parameters[2]])
    gwtheta = np.array([parameters[3]])

    # Get the signal response matrix, which contains the Fplus and Fcross
    Fr = signalResponse_fast(psrpos_theta, psrpos_phi, gwphi, gwtheta)
    Fp = Fr[0, 0]
    Fc = Fr[0, 1]

    pol = np.cos(2*parameters[4]) * Fp + np.sin(2*parameters[4]) * Fc

    # Define the heaviside function
    heaviside = lambda x: 0.5 * (np.sign(x) + 1)

    # Return the time-series for teh pulsar
    return pol * (10**parameters[1]) * heaviside(t - parameters[0]) * (t - parameters[0])


