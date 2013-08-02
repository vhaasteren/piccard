#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

import numpy as np
import scipy.linalg as sl, scipy.special as ss
import h5py as h5
#import bangwrapper as bw
import matplotlib.pyplot as plt
import os as os
import sys
import libstempo as t2
import pytwalk
import emcee
import statsmodels.api as sm


"""
Calculate the design matrix for quadratic spindown
"""
def designqsd(t):
  M = np.ones([len(t), 3])
  M[:,1] = t
  M[:,2] = t ** 2
  return M.copy()

"""
Calculate the G-matrix of van Haasteren and Levin (2012)
"""
def gdesign(M):
  U, s, Vh = sl.svd(M)

  return U[:,M.shape[1]:].copy()


"""
Calculate the matrix of Fourier modes A, given a set of timestamps

These are sine/cosine amplitudes at evenly separated frequency bins

Mode 0: constant (cos(0))
Mode 1: sin(f_0)
Mode 2: cos(f_0)
Mode 3: sin(f_1)
... etc
"""
def fourierdesignmatrix(t, nmodes, Ttot=None):
  N = t.size
  A = np.zeros([N, nmodes])
  freqs = np.zeros(nmodes)
  T = t.max() - t.min()

  if(nmodes % 2 != 0):
    print "WARNING: Number of modes should be even!"

  # The frequency steps
  #deltaf = (N-1.0) / (N*T)    # This would be orthogonal for regular sampling
  if Ttot == None:
      deltaf = 1.0 / T
  else:
      deltaf = 1.0 / Ttot


  # The zeroth mode (constant, cos(0))
  # Skip this one now!
  # A[:,0] = 0.5 * np.sqrt(2)
  # freqs[0] = 0.0

  # The cosine modes
  for i in range(0, nmodes, 2):
    # Mode number
    k = 1 + int(i / 2)
    # frequency
    omega = 2.0 * np.pi * k * deltaf
    A[:,i] = np.cos(omega * t)
    freqs[i] = k * deltaf

  # The sine modes
  for i in range(1, nmodes, 2):
    # Mode number
    k = int((i + 1) / 2)
    # frequency
    omega = 2.0 * np.pi * k * deltaf
    A[:,i] = np.sin(omega * t)
    freqs[i] = k * deltaf

  # This normalisation would make F unitary in the case of regular sampling
  # A = A * np.sqrt(2.0/N)

  return (A, freqs)


# Calculate the PTA covariance matrix (only GWB)
def Cgw_sec(toas, alpha=-2.0/3.0, fL=1.0/20, approx_ksum=False):
    day    = 86400.0
    year   = 3.15581498e7
    EulerGamma = 0.5772156649015329

    psrobs = [len(toas)]
    alphaab = np.array([[1.0]])
    times_f = toas / day
    
    npsrs = alphaab.shape[0]

    t1, t2 = np.meshgrid(times_f,times_f)

    # t1, t2 are in units of days; fL in units of 1/year (sidereal for both?)
    # so typical values here are 10^-6 to 10^-3
    x = 2 * np.pi * (day/year) * fL * np.abs(t1 - t2)

    del t1
    del t2

    # note that the gamma is singular for all half-integer alpha < 1.5
    #
    # for -1 < alpha < 0, the x exponent ranges from 4 to 2 (it's 3.33 for alpha = -2/3)
    # so for the lower alpha values it will be comparable in value to the x**2 term of ksum
    #
    # possible alpha limits for a search could be [-0.95,-0.55] in which case the sign of `power`
    # is always positive, and the x exponent ranges from ~ 3 to 4... no problem with cancellation

    # The tolerance for which to use the Gamma function expansion
    tol = 1e-5

    # the exact solutions for alpha = 0, -1 should be acceptable in a small interval around them...
    if abs(alpha) < 1e-7:
        cosx, sinx = np.cos(x), np.sin(x)

        power = cosx - x * sinx
        sinint, cosint = sl.sici(x)

        corr = (year**2 * fL**-2) / (24 * math.pi**2) * (power + x**2 * cosint)
    elif abs(alpha + 1) < 1e-7:
        cosx, sinx = np.cos(x), np.sin(x)

        power = 6 * cosx - 2 * x * sinx - x**2 * cosx + x**3 * sinx
        sinint, cosint = ss.sici(x)

        corr = (year**2 * fL**-4) / (288 * np.pi**2) * (power - x**4 * cosint)
    else:
        # leading-order expansion of Gamma[-2+2*alpha]*Cos[Pi*alpha] around -0.5 and 0.5
        if   abs(alpha - 0.5) < tol:
            cf =  np.pi/2   + (np.pi - np.pi*EulerGamma)              * (alpha - 0.5)
        elif abs(alpha + 0.5) < tol:
            cf = -np.pi/12  + (-11*np.pi/36 + EulerGamma*math.pi/6)     * (alpha + 0.5)
        elif abs(alpha + 1.5) < tol:
            cf =  np.pi/240 + (137*np.pi/7200 - EulerGamma*np.pi/120) * (alpha + 1.5)
        else:
            cf = ss.gamma(-2+2*alpha) * np.cos(np.pi*alpha)

        power = cf * x**(2-2*alpha)

        # Mathematica solves Sum[(-1)^n x^(2 n)/((2 n)! (2 n + 2 alpha - 2)), {n, 0, Infinity}]
        # as HypergeometricPFQ[{-1+alpha}, {1/2,alpha}, -(x^2/4)]/(2 alpha - 2)
        # the corresponding scipy.special function is hyp1f2 (which returns value and error)
        # TO DO, for speed: could replace with the first few terms of the sum!
        if approx_ksum:
            ksum = 1.0 / (2*alpha - 2) - x**2 / (4*alpha) + x**4 / (24 * (2 + 2*alpha))
        else:
            ksum = ss.hyp1f2(alpha-1,0.5,alpha,-0.25*x**2)[0]/(2*alpha-2)

        del x

        # this form follows from Eq. (A31) of Lee, Jenet, and Price ApJ 684:1304 (2008)
        corr = -(year**2 * fL**(-2+2*alpha)) / (12 * np.pi**2) * (power + ksum)
        
    return corr





# 5 yr dataset, 1000 observations, regular sampling, 100ns timing noise
spd = 3600 * 24         # seconds per day
spy = spd * 365.25      # seconds per year
N = 1000                # observations
T = 5.0*spy
toas = np.linspace(0, T, N)
toaerrs = np.ones(N) * 1.0e-7

# Use 40 modes (20 frequencies) in total
F, Ffreqs = fourierdesignmatrix(toas, 400)

# Construct design matrices
M = designqsd(toas)
G = gdesign(M)

# Power-law spectrum with amplitude Ag=5.0e-14
Ag = 5.0e-14
pcoefs = (Ag**2 * spy**3 / (12*np.pi*np.pi*T)) * ((Ffreqs * spy) ** (-13.0/3.0))
SigmaFD = np.dot(F, np.dot(np.diag(pcoefs), F.T))
SigmaTD = Ag**2 * Cgw_sec(toas)

# Add the noise matrices
CFD = np.diag(toaerrs**2) + SigmaFD
CTD = np.diag(toaerrs**2) + SigmaTD

# Project out the timing model
GCfG = np.dot(G.T, np.dot(CFD, G))
GCtG = np.dot(G.T, np.dot(CTD, G))

#print Ffreqs * spy

print "The fractional difference in the covariance matrix elements is max ", np.max((GCfG - GCtG) / GCfG)

# Decompose these matrices
cff = sl.cholesky(GCfG).T
cft = sl.cholesky(GCtG).T

# Generate some random numbers for signal generation
xi = np.random.randn(GCtG.shape[0])

# Construct the signals
yf = np.dot(G, np.dot(cff, xi))
yt = np.dot(G, np.dot(cft, xi))

# Plot the lot
plt.figure()
plt.subplot(3, 1, 1)
plt.errorbar(toas, yf, yerr=toaerrs, fmt='.', c='blue')
plt.annotate('frequency domain',
        xy=(0.02, 0.20), xycoords='axes fraction',
        bbox=dict(boxstyle='round', fc='1.0'))
plt.grid(True)

plt.subplot(3, 1, 2)
plt.errorbar(toas, yt, yerr=toaerrs, fmt='.', c='blue')
plt.annotate('time domain',
        xy=(0.02, 0.20), xycoords='axes fraction',
        bbox=dict(boxstyle='round', fc='1.0'))
plt.grid(True)

plt.subplot(3, 1, 3)
plt.errorbar(np.log10(Ffreqs), np.log10(pcoefs), fmt='.', c='blue')
plt.annotate('Power spectrum',
        xy=(0.02, 0.20), xycoords='axes fraction',
        bbox=dict(boxstyle='round', fc='1.0'))
plt.xlabel('log10(f)')
plt.ylabel('log10(S(f))')
plt.grid(True)

plt.show()
