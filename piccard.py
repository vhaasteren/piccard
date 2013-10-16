#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
piccard.py

Requirements:
- numpy:        pip install numpy
- h5py:         macports, apt-get, http://h5py.googlecode.com/
- matplotlib:   macports, apt-get
- emcee:        pip install emcee (fallback option included)
- pyMultiNest:  (optional)
- pytwalk:      (included)
- pydnest:      (included)

Created by vhaasteren on 2013-08-06.
Copyright (c) 2013 Rutger van Haasteren

Work that uses this code should reference van Haasteren et al. (in prep). (I'll
add the reference later).

Contributed code for anisotropic gravitrational-wave background by Chiara
Mingarelli. Work that uses the anisotropic background functionality should
reference Mingarelli and Vecchio 2013,  arXiv:1306.5394

Contributed work on anisotropic gravitational-wave background by Steve Taylor.
Work that uses the anisotropic background functionality should reference Taylor
and Gair 2013, arXiv:1306:5395

"""

from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import h5py as h5
import matplotlib.pyplot as plt
import os as os
import sys
import pytwalk                  # Internal module
import pydnest                  # Internal module
import anisotropygammas as ang  # Internal module
import rjmcmchammer as rjemcee  # Internal module
from triplot import *


try:
    import statsmodels.api as smapi
    sm = smapi
except ImportError:
    sm = None

try:    # If without libstempo, can still read hdf5 files
    import libstempo
    t2 = libstempo
except ImportError:
    t2 = None

try:    # Fall back to internal emcee implementation
    import emcee as emceehammer
    emcee = emceehammer
except ImportError:
    import mcmchammer as mcmchammer
    emcee = mcmchammer

try:    # If MultiNest is not installed, do not use it
    import pymultinest

    pymultinest = pymultinest
except ImportError:
    pymultinest = None

# For DM calculations, use this constant
# See You et al. (2007) - http://arxiv.org/abs/astro-ph/0702366
# Lee et al. (in prep.) - ...
# Units here are such that delay = DMk * DM * freq^-2 with freq in MHz
DMk = 4.15e3    #  Units MHz^2 cm^3 pc sec



"""
The DataFile class is the class that supports the HDF5 file format. It
most likely needs to be re-designed, since it does not provide a universal
interface
"""
class DataFile(object):
    filename = None
    h5file = None

    def __init__(self, filename=None):
        # Open the hdf5 file?
        self.setfile(filename)

    def __del__(self):
        # Delete the instance, and close the hdf5 file?
        pass

    """
    Add another pulsar to the HDF5 file, given a tempo2 par and tim file. The
    HDF5 file should not yet contain a description of the model. Adding data
    would invalidate the model, so the model should first be deleted (or
    otherwise adjusted, TODO).

    The main idea is that this folder in the HDF5 file only contains information
    obtained from tempo2. It is the 'input' to any further processing, and as
    such should not be modified. Adding flags and other stuff should be done as
    part of the modelling (in the /Models folder).
    """
    def addpulsar(self, parfile, timfile):
        # Check whether the two files exist
        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile, timfile)
        assert(self.filename != None), "ERROR: HDF5 file not set!"

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')

        if "Model" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "model already available in '%s'. Refusing to add data" % (self.filename)

        # Obtain the directory name of the timfile, and change to it
        timfiletup = os.path.split(timfile)
        dirname = timfiletup[0]
        reltimfile = timfiletup[-1]
        relparfile = os.path.relpath(parfile, dirname)
        savedir = os.getcwd()

        # Create the data subgroup if it does not exist
        if "Data" in self.h5file:
            datagroup = self.h5file["Data"]
        else:
            datagroup = self.h5file.create_group("Data")

        # Load pulsar data from the JPL Cython tempo2 library
        os.chdir(dirname)
        t2pulsar = t2.tempopulsar(relparfile, reltimfile)
        os.chdir(savedir)

        # Create the pulsar subgroup if it does not exist
        if "Pulsars" in datagroup:
            pulsarsgroup = datagroup["Pulsars"]
        else:
            pulsarsgroup = datagroup.create_group("Pulsars")

        # Look up the name of the pulsar, and see if it exist
        if t2pulsar.name in pulsarsgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "%s already exists in %s!" % (t2pulsar.name, self.filename)

        # TODO: pulsarsgroup is re-defined here. Name it pulsargroup or
        # something like that. This is soooo unclear
        pulsarsgroup = pulsarsgroup.create_group(t2pulsar.name)

        # Create the datasets, with reference time pepoch = 53000
        spd = 24.0*3600     # seconds per day
        pulsarsgroup.create_dataset('TOAs', data=np.double(np.array(t2pulsar.toas())-53000)*spd)       # days (MJD) * sec per day
        pulsarsgroup.create_dataset('prefitRes', data=np.double(t2pulsar.residuals()))      # seconds
        pulsarsgroup.create_dataset('postfitRes', data=np.double(t2pulsar.residuals()))  # seconds
        pulsarsgroup.create_dataset('toaErr', data=np.double(1e-6*t2pulsar.toaerrs))          # seconds
        pulsarsgroup.create_dataset('freq', data=np.double(t2pulsar.freqs))              # MHz


        # Read the data from the tempo2 structure. Use pepoch=53000 for all
        # pulsars so that the time-correlations are synchronous
        # TODO: Do not down-convert quad precision to double precision here
        #t2data = np.double(t2pulsar.data(pepoch=53000))
        #designmatrix = np.double(t2pulsar.designmatrix(pepoch=53000))

        # Write the full design matrix
        # TODO: this should be done irrespective of fitting flag
        desmat = t2pulsar.designmatrix()
        pulsarsgroup.create_dataset('designmatrix', data=desmat)

        # Write the G-matrix
        U, s, Vh = sl.svd(desmat)
        pulsarsgroup.create_dataset('Gmatrix', data=U[:, desmat.shape[1]:])

        # Write the coG-matrix (complement of the G-matrix)
        pulsarsgroup.create_dataset('coGmatrix', data=U[:, :desmat.shape[1]])

        # Now write the timing model parameters
        tmpname = ['Offset'] + list(t2pulsar.pars)
        tmpvalpre = np.zeros(len(tmpname))
        tmpvalpost = np.zeros(len(tmpname))
        tmperrpre = np.zeros(len(tmpname))
        tmperrpost = np.zeros(len(tmpname))

        for i in range(len(t2pulsar.pars)):
            tmpvalpre[i+1] = t2pulsar.prefit[tmpname[i+1]].val
            tmpvalpost[i+1] = t2pulsar.prefit[tmpname[i+1]].val
            tmperrpre[i+1] = t2pulsar.prefit[tmpname[i+1]].err
            tmperrpost[i+1] = t2pulsar.prefit[tmpname[i+1]].err

        # Write the timing model parameter (TMP) descriptions
        pulsarsgroup.create_dataset('tmp_name', data=tmpname)       # TMP name
        pulsarsgroup.create_dataset('tmp_valpre', data=tmpvalpre)   # TMP pre-fit value
        pulsarsgroup.create_dataset('tmp_valpost', data=tmpvalpost) # TMP post-fit value
        pulsarsgroup.create_dataset('tmp_errpre', data=tmperrpre)   # TMP pre-fit error
        pulsarsgroup.create_dataset('tmp_errpost', data=tmperrpost) # TMP post-fit error

        # Delete a group for flags if it exists
        # TODO: is this ok??
        if "Flags" in pulsarsgroup:
            print "WARNING: deleting the already existing flags group for (%s)" % (pulsarsgroup.name)
            del pulsarsgroup["Flags"]

        # Freshly create the flags from scratch
        flaggroup = pulsarsgroup.create_group("Flags")

        # Obtain the unique flags in this dataset
        uflags = list(set(t2pulsar.flags))

        # For every flag id, write the values for the TOAs
        # print "# For every flag id, write the values for the TOAs"
        for flagid in uflags:
            #flaggroup.create_dataset(flagid, data=t2pulsar.flagvalue(flagid))
            flaggroup.create_dataset(flagid, data=t2pulsar.flags[flagid])

        if not "efacequad" in flaggroup:
            # Check if the sys-flag is present in this set. If it is, add an
            # efacequad flag with pulsarname+content of the sys-flag. If it isn't, add an
            # efacequad flag with the pulsar name as it's elements.
            efacequad = []
            nobs = len(t2pulsar.toas())
            pulsarname = map(str, [t2pulsar.name] * nobs)
            if "sys" in flaggroup:
                efacequad = map('-'.join, zip(pulsarname, flaggroup['sys']))
            else:
                efacequad = pulsarname

            flaggroup.create_dataset("efacequad", data=efacequad)

        if not "pulsarname" in flaggroup:
            nobs = len(t2pulsar.toas())
            pulsarname = map(str, [t2pulsar.name] * nobs)
            flaggroup.create_dataset("pulsarname", data=pulsarname)

        # Close the hdf5 file
        self.h5file.close()
        self.h5file = None


    """
    Given a flag and a flag value, figure out whether this source only applies
    to a pulsar, or to several pulsars. The return value is either the index
    number of the pulsar, or -1

    flagvalue: the value of the flag for this source
    flagvalues: list of the flag values for all TOAs
    pulsarflagvalues: list of pulsar names for all TOAs
    pulsarnames: list of all pulsars
    """
    def pulsarnumberfromflagvalue(self, flagvalue, flagvalues, pulsarflags, pulsarnames):
        indices = np.flatnonzero(np.array(flagvalues == flagvalue))
        sourcepulsars = pulsarflags[indices]
        uniquepulsars = list(set(sourcepulsars))
        if len(uniquepulsars) == 1:
            retvalue = pulsarnames.index(uniquepulsars[0])
        else:
            retvalue = -1

        return retvalue

    """
    This function uses the function listed above. It figures out what the pulsar
    number is, given a flag, a flag value, the pulsar names, and the hdf5
    processed data group

    This is used to see if a particular source (which works on a flag) only
    works on a single pulsar or on several/all pulsars. If it only works on a
    single pulsar, a covariance matrix of that source for only one pulsar can be
    used.  Otherwise, it should be calculated for all pulsars, which takes more
    time and memory.
    """
    def pulsarnumberfromflag(self, flag, flagvalue, pulsarnames, processeddatagroup):
        retvalue = -1
        if flag in processeddatagroup['Flags']:
            flagvalues = np.array(map(str, processeddatagroup['Flags'][flag]))
            pulsarflags = np.array(map(str, processeddatagroup['Flags']['pulsarname']))
            retvalue = self.pulsarnumberfromflagvalue(flagvalue, flagvalues, pulsarflags, pulsarnames)

        return retvalue


    def setfile(self, filename):
        self.filename = filename


"""
Calculate the daily-averaging exploder matrix, and the daily averaged site
arrival times. In the modelling, the residuals will not be changed. It is only
for calculating correlations

Input is a vector of site arrival times. Returns the reduced-size average toas,
and the exploder matrix  Cfull = U Cred U^{T}
"""
def dailyaveragequantities(toas):
    spd = 3600.0 * 24.0     # Seconds per day

    processed = np.array([0]*len(toas), dtype=np.bool)  # No toas processed yet
    U = np.zeros((len(toas), 0))
    avetoas = np.empty(0)

    while not np.all(processed):
        npindex = np.where(processed == False)[0]
        ind = npindex[0]
        satmin = toas[ind] - spd
        satmax = toas[ind] + spd

        dailyind = np.where(np.logical_and(toas > satmin, toas < satmax))[0]

        newcol = np.zeros((len(toas)))
        newcol[dailyind] = 1.0

        U = np.append(U, np.array([newcol]).T, axis=1)
        avetoas = np.append(avetoas, np.mean(toas[dailyind]))
        processed[dailyind] = True

    return (avetoas, U)



"""
Calculate the two Fourier modes A, given a set of timestamps and a frequency

These are sine/cosine modes
"""
def singleFreqFourierModes(t, freqs):
    N = t.size
    M = len(freqs)
    A = np.zeros([N, 2*M])

    for ii in range(len(freqs)):
        A[:,2*ii] = np.cos(2.0 * np.pi * freqs[ii] * t)
        A[:,2*ii+1] = np.sin(2.0 * np.pi * freqs[ii] * t)

    return A


"""
Calculate the matrix of Fourier modes A, given a set of timestamps

These are sine/cosine basis vectors at evenly separated frequency bins

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


"""
Calculate the design matrix for quadratic spindown
"""
def designqsd(t, f=None):
  if not f==None:
    cols = 4
  else:
    cols = 3
  M = np.ones([len(t), cols])
  
  M[:,1] = t
  M[:,2] = t ** 2
    
  if not f==None:
    M[:,3] = 1.0 / (f**2)
    
  return M.copy()




"""
with n the number of pulsars, return an nxn matrix representing the H&D
correlation matrix
"""
def hdcorrmat(ptapsrs):
    """ Constructs a correlation matrix consisting of the Hellings & Downs
        correlation coefficients. See Eq. (A30) of Lee, Jenet, and
        Price ApJ 684:1304 (2008) for details.

        @param: list of ptaPulsar (or any other markXPulsar) objects
        
    """
    npsrs = len(ptapsrs)
    
    raj = [ptapsrs[i].raj[0] for i in range(npsrs)]
    decj = [ptapsrs[i].decj[0] for i in range(npsrs)]
    pp = np.array([np.cos(decj)*np.cos(raj), np.cos(decj)*np.sin(raj), np.sin(decj)]).T
    cosp = np.array([[np.dot(pp[i], pp[j]) for i in range(npsrs)] for j in range(npsrs)])
    cosp[cosp > 1.0] = 1.0
    xp = 0.5 * (1 - cosp)

    old_settings = np.seterr(all='ignore')
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old_settings)
    hdmat = logxp - 0.25 * xp + 0.5 + 0.5 * np.diag(np.ones(npsrs))

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


# Calculate the covariance matrix for a red signal
# (for a GWB with unitless amplitude h_c(1yr^{-1}) = 1)
def Cred_sec(toas, alpha=-2.0/3.0, fL=1.0/20, approx_ksum=False):
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


"""
The real-valued spherical harmonics
"""
def real_sph_harm(mm, ll, phi, theta):
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


# The GWB general anisotropic correlations as defined in
# Mingarelli and Vecchio (submitted); Taylor and Gair (submitted)
class aniCorrelations(object):
    phiarr = None           # The phi pulsar position parameters
    thetaarr = None         # The theta pulsar position parameters
    gamma_ml = None         # The gamma_ml (see anisotropygammas.py)

    # The anisotropic search requires a specific type of prior: the combination
    # c_lm * 
    priorgridbins = 16
    priorphi = None         # Phi value of the locations for a prior check
    priortheta = None       # Theta value of the locations for a prior check

    # Correlation matrices for the anisotropic components
    corrhd = None   # H&D correlations
    corr = []
    l = 1           # The order of the anisotropic correlations (dipole, quadrupole, ...)

    # Pre-compute the spherical harmonics for all sky positions
    SpHmat = None

    def __init__(self, psrs=None, l=1):
        # If we have a pulsars object, initialise the angular quantities
        if psrs != None:
            self.setmatrices(psrs, l)
        else:
            self.phiarr = None           # The phi pulsar position parameters
            self.thetaarr = None         # The theta pulsar position parameters
            self.gamma_ml = None         # The gamma_ml (see anisotropygammas.py)

            self.priorgridbins = 16
            self.priorphi = None
            self.priortheta = None

            self.corrhd = None
            self.corr = []
            self.SpHmat = None

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

        # Construct a nxn grid of phi/theta values for prior checks
        prphi = np.linspace(0.0, 2*np.pi, self.priorgridbins, endpoint=False)
        prtheta = np.linspace(0.0, np.pi, self.priorgridbins)
        pprphi, pprtheta = np.meshgrid(prphi, prtheta)
        self.priorphi = pprphi.flatten()
        self.priortheta = pprtheta.flatten()

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

                        """
                        if aa == 0 and bb == 1:
                            print "-----------------"
                            print "pulsars: ", psrs[aa].name, psrs[bb].name
                            print "phi: ", self.phiarr[aa], self.phiarr[bb]
                            print "theta: ", self.thetaarr[aa], self.thetaarr[bb]
                            print "(ll, mm) = ", ll, m
                            print "indexlm = ", mindex+mm
                            print "mindex = ", mindex
                            print "-----------------"

                            newnorm = 3./(8*np.pi)
                            oldnorm = 3./(4*np.sqrt(np.pi))

                            print "corr: ", ang.real_rotated_Gammas(m, ll, \
                                self.phiarr[aa], self.phiarr[bb], \
                                self.thetaarr[aa], self.thetaarr[bb], gamma_ml)
                        """

                        if aa != bb:
                            self.corr[mindex+mm][bb, aa] = self.corr[mindex+mm][aa, bb]

        self.SpHmat = np.zeros((self.priorgridbins*self.priorgridbins, self.clmlength()))
        gridindex = 0
        cindex = 0
        #for ii in range(self.priorgridbins):
        #    for jj in range(self.priorgridbins):
        for ii in range(self.priorgridbins**2):
                cindex = 0
                for ll in range(1, self.l+1):
                    for mm in range(-ll, ll+1):
                        self.SpHmat[gridindex, cindex] = real_sph_harm(mm, ll, self.priorphi[ii], self.priortheta[ii])

                        cindex += 1

                gridindex += 1


    def priorIndicator(self, clm):
        # Check whether sum_lm c_lm * Y_lm > 0 for this combination of clm
        if self.priorphi == None or self.priortheta == None:
            raise ValueError("ERROR: first define the anisotropic prior-check positions")

        if len(self.priorphi) != len(self.priortheta):
            raise ValueError("ERROR: len(self.priorphi) != len(self.priortheta)")

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
A general signal element of the pta model/likelihood.

For now, the Fmat fourier design matrices are assumed to be for identical
frequencies for all pulsars.
"""
class ptasignal(object):
    pulsarind = None        # pulsar nr. for EFAC/EQUAD
    stype = "none"          # EFAC, EQUAD, spectrum, powerlaw,
                            # dmspectrum, dmpowerlaw, fouriercoeff...
    corr = "single"         # single, gr, uniform, dipole, anisotropicgwb...
                            # Here dipole is not the dipole in anisotropies, but
                            # in 'ephemeris' etc.

    flagname = 'efacequad'  # Name of flag this applies to
    flagvalue = 'noname'    # Flag value this applies to

    npars = 0               # Number of parameters
    ntotpars = 0            # Total number of parameters (also non-varying)
    nindex = 0              # Index in parameters array
    npsrfreqindex = 0       # Index of frequency line for this psr (which line)
                            #   Do not double-count frequencies (so not modes, but
                            #   freqs)
    npsrdmfreqindex = 0     # Index of DM frequency line for this psr (which line)

    bvary = None            # Which parameters are varying of this signals

    pmin = None             # Minimum bound for all parameters (also n.v.)
    pmax = None             # Maximum bound for all parameters (also n.v.)
    pwidth = None           # Stepsize bound for all parameters (also n.v.)
    pstart = None           # Start position for all parameters (also n.v.)

    # Quantities for EFAC/EQUAD
    Nvec = None             # For in the mark3+ likelihood

    # If this pulsar has only one efac/equad parameter, use computational
    # shortcut, using:
    accelNoise = False      # Works only for efac-only (implement automagically?)

    # Quantities for spectral noise
    Tmax = None
    corrmat = None

    aniCorr = None       # Anisotropic correlations are described by this class



"""
Basic implementation of the model/likelihood. Most of the likelihood functions
use models as outlined in Lentati et al. (2013).

mark1loglikelihood: The red noise Fourier modes are included numerically. Larger
dimensional space, but quicker evaluation. At his point, the inversion of the
Phi matrix is only fast if phi is diagonal.

mark2loglikelihood: only efac + equad signals

mark3loglikelihood: analytically integrated over the red noise Fourier modes. DM
variations are projected on these modes, which is very suboptimal. Do not use
with DM variations. The Phi matrix inversion is not optimised per frequency. At
least one red signal must be included for each pulsar

mark6loglikelihood: analytically integrated over the red noise
Fourier modes, and the DM variation Fourier modes. The integration is done
simultaneously. Makes for a larger Phi matrix.

mark7loglikelihood: like mark3loglikelihood, but allows for RJMCMC Fourier mode
selection

mark8loglikelihood: like mark6loglikelihood, but allows for RJMCMC Fourier mode
selection

mark9loglikelihood: like mark3loglikelihood, but allows for an extra noise
source, which models a single frequency-source with a variable frequency and
amplitude
"""
class ptaPulsar(object):
    raj = 0
    decj = 0
    toas = None
    toaerrs = None
    residuals = None
    freqs = None
    Gmat = None
    Gcmat = None
    Mmat = None
    ptmpars = []
    ptmdescription = []
    flags = None
    name = "J0000+0000"

    # The auxiliary quantities
    Fmat = None
    SFmat = None            # Fmatrix for the frequency lines
    FFmat = None            # Total of Fmat and SFmat
    Fdmmat = None
    SFdmmat = None         # Fdmmatrix for the dm frequency lines
    #FFdmmat = None         # Total of SFdmmatrix and Fdmmat
    Dmat = None
    DF = None
    DSF = None
    DFF = None              # Total of DF and DSF
    Ffreqs = None       # Frequencies of the red noise
    SFfreqs = None      # Single modelled frequencies
    SFdmfreqs = None
    frequencyLinesAdded = 0      # Whether we have > 1 single frequency line
    dmfrequencyLinesAdded = 0      # Whether we have > 1 single frequency line
    Fdmfreqs = None
    Emat = None
    EEmat = None
    Gr = None
    GGr = None
    GtF = None
    #GGtF = None
    #GGtFF = None
    GGtD = None
    AGr = None      # Replaces GGr in 2-component noise model
    AGF = None      # Replaces GGtF in 2-component noise model
    AGD = None      # Replaces GGtD in 2-component noise model
    AGE = None      # Replaces GGtE in 2-component noise model

    # Auxiliaries used in the likelihood
    twoComponentNoise = False       # Whether we use the 2-component noise model
    coarseGrainedEquad = False      # Whether we coarse-grain the equad
    Nvec = None             # The total white noise (eq^2 + ef^2*err)
    Wvec = None             # The weights in 2-component noise
    Nwvec = None            # Total noise in 2-component basis (eq^2 + ef^2*Wvec)

    # To select the number of Frequency modes
    bfinc = None        # Number of modes of all internal matrices
    bfdminc = None      # Number of modes of all internal matrices (DM)
    bcurfinc = None     # Current number of modes in RJMCMC
    bcurfdminc = None   # Current number of modes in RJMCMC

    def __init__(self):
        self.raj = 0
        self.decj = 0
        self.toas = None
        self.toaerrs = None
        self.residuals = None
        self.freqs = None
        self.Gmat = None
        self.Gcmat = None
        self.Mmat = None
        self.ptmpars = []
        self.ptmdescription = []
        self.flags = None
        self.name = "J0000+0000"

        self.Fmat = None
        self.SFmat = None
        self.FFmat = None
        self.Fdmmat = None
        self.Dmat = None
        self.DF = None
        self.Ffreqs = None
        self.SFfreqs = None
        self.Fdmfreqs = None
        self.Emat = None
        self.EEmat = None
        self.Gr = None
        self.GGr = None
        self.GtF = None
        #self.GGtF = None
        #self.GGtFF = None
        self.GGtD = None

        self.bfinc = None
        self.bfdminc = None
        self.bprevfinc = None
        self.bprevfdminc = None

    def readFromH5(self, filename, psrname):
        h5file = h5.File(filename, 'r+')

        # Retrieve the models group
        if not "Data" in h5file:
            h5file.close()
            h5file = None
            raise IOError, "no Data group in hdf5 file"

        datagroup = h5file["Data"]

        # Retrieve the pulsars group
        if not "Pulsars" in datagroup:
            h5file.close()
            h5file = None
            raise IOError, "no Pulsars group in hdf5 file"

        pulsarsgroup = datagroup["Pulsars"]

        # Retrieve the pulsar
        if not psrname in pulsarsgroup:
            h5file.close()
            h5file = None
            raise IOError, "no Pulsar " + psrname + " found in hdf5 file"

        # Read the position
        rajind = np.flatnonzero(np.array(pulsarsgroup[psrname]['tmp_name']) == 'RAJ')
        decjind = np.flatnonzero(np.array(pulsarsgroup[psrname]['tmp_name']) == 'DECJ')
        self.raj = np.array(pulsarsgroup[psrname]['tmp_valpre'])[rajind]
        self.decj = np.array(pulsarsgroup[psrname]['tmp_valpre'])[decjind]

        # Obtain residuals, TOAs, etc.
        self.toas = np.array(pulsarsgroup[psrname]['TOAs'])
        self.toaerrs = np.array(pulsarsgroup[psrname]['toaErr'])
        self.residuals = np.array(pulsarsgroup[psrname]['prefitRes'])
        self.freqs = np.array(pulsarsgroup[psrname]['freq'])
        self.Mmat = np.array(pulsarsgroup[psrname]['designmatrix'])

        # See if we can find the G-matrix and coG-matrix
        if not "Gmatrix" in pulsarsgroup[psrname] or not "coGmatrix" in pulsarsgroup[psrname]:
            print "(co)Gmatrix not found for " + psrname + ". Constructing it now."
            U, s, Vh = sl.svd(self.Mmat)
            self.Gmat = U[:, self.Mmat.shape[1]:].copy()
            self.Gcmat = U[:, :self.Mmat.shape[1]].copy()
        else:
            self.Gmat = np.array(pulsarsgroup[psrname]['Gmatrix'])
            self.Gcmat = np.array(pulsarsgroup[psrname]['coGmatrix'])

        # Obtain the other stuff
        self.ptmpars = np.array(pulsarsgroup[psrname]['tmp_valpre'])
        if "efacequad" in pulsarsgroup[psrname]['Flags']:
            self.flags = map(str, pulsarsgroup[psrname]['Flags']['efacequad'])
        else:
            self.flags = [psrname] * len(self.toas)

        self.ptmdescription = map(str, pulsarsgroup[psrname]['tmp_name'])
        self.name = psrname

        h5file.close()
        h5file = None

    def readFromImagination(self, filename, psrname):
        # Read the position
        self.raj = np.array([np.pi/4.0])
        self.decj = np.array([np.pi/4.0])

        # Obtain residuals, TOAs, etc.
        self.toas = np.linspace(0, 10*365.25*3600*24, 300)
        self.toaerrs = np.array([1.0e-7]*len(self.toas))
        self.residuals = np.array([0.0e-7]*len(self.toas))
        self.freqs = np.array([
                np.array([720]*int(len(self.toas)/3)), \
                np.array([1440]*int(len(self.toas)/3)), \
                np.array([2880]*int(len(self.toas)/3))]).T.flatten()

        self.Mmat = designqsd(self.toas, self.freqs)

        U, s, Vh = sl.svd(self.Mmat)
        self.Gmat = U[:, self.Mmat.shape[1]:].copy()

        self.psrname = 'J0000+0000'
        self.flags = [self.psrname] * len(self.toas)

        self.ptmdescription = ['QSDpar'] * self.Mmat.shape[1]

    # Modify the design matrix in some general way. Either add DM derivatives,
    # or remove jumps, or ...
    def getModifiedDesignMatrix(self, addDMQSD=False, removeJumps=False):
        # Which rows of Mmat to keep:
        indkeep = np.array([1]*self.Mmat.shape[1], dtype=np.bool)
        dmaddes = ['DM', 'DM1', 'DM2']
        dmadd = np.array([addDMQSD]*len(dmaddes), dtype=np.bool)
        for ii in range(self.Mmat.shape[1]):
            # Check up on jumps.
            parlabel = self.ptmdescription[ii]
            firstfour = parlabel[:4]
            if removeJumps and firstfour.upper() == 'JUMP':
                # This is a jump. Remove it
                indkeep[ii] = False

            # Check up on DM parameters
            if parlabel in dmaddes:
                # Parameter in the list, so mark as such
                dmadd[dmaddes.index(parlabel)] = False

        # Construct a new design matrix with/without the Jump parameters
        tempM = self.Mmat[:, indkeep]
        tempptmpars = self.ptmpars[indkeep]
        tempptmdescription = []
        for ii in range(len(self.ptmdescription)):
            tempptmdescription.append(self.ptmdescription[ii])

        # Construct the design matrix elements for the DM QSD if required, and
        # add them to the new M matrix
        if np.sum(dmadd) > 0:
            # Construct the DM QSD matrix
            dmqsdM = np.zeros((self.Mmat.shape[0], np.sum(dmadd)))
            dmqsddes = []
            dmqsdpars = np.zeros(np.sum(dmadd))
            Dmatdiag = DMk / (self.freqs**2)
            index = 0
            for ii in range(len(dmaddes)):
                if dmadd[ii]:
                    dmqsdM[:, index] = Dmatdiag * (self.toas ** ii)
                    if ii > 0:
                        description = 'DM' + str(ii)
                    else:
                        description = 'DM'
                    dmqsddes.append(description)
                    dmqsdpars[index] = 0.0
                    index += 1

            newM = np.append(tempM, dmqsdM, axis=1)
            newptmpars = np.append(tempptmpars, dmqsdpars)
            newptmdescription = tempptmdescription + dmqsddes
        else:
            newM = tempM
            newptmpars = tempptmpars
            newptmdescription = tempptmdescription

        # Construct the G-matrices
        U, s, Vh = sl.svd(newM)
        newG = U[:, (newM.shape[1]):].copy()
        newGc = U[:, :(newM.shape[1])].copy()

        return newM, newG, newGc, newptmpars, newptmdescription



    # Modify the design matrix to include fitting for a quadratic in the DM
    # signal.
    # TODO: Check if the DM is fit for in the design matrix. Use ptmdescription
    #       for that. It should have a field with 'DM' in it.
    def addDMQuadratic(self):
        self.Mmat, self.Gmat, self.Gcmat, self.ptmpars, self.ptmdescription = \
                self.getModifiedDesignMatrix(addDMQSD=True, removeJumps=False)

        """
        if 'DM' in self.ptmdescription:
            # DM is included, do not include it again
            newM = np.zeros((self.Mmat.shape[0], self.Mmat.shape[1]+2))
            Dmatdiag = DMk / (self.freqs**2)
            d = np.array([Dmatdiag*self.toas, Dmatdiag*(self.toas**2)]).T

            self.ptmdescription.append('DM1')
            self.ptmdescription.append('DM2')
            self.ptmpars = np.append(self.ptmpars, [0.0, 0.0])
            
            newM[:,:-2] = self.Mmat
            newM[:,-2:] = d
        else:
            # DM is not included, include it now
            newM = np.zeros((self.Mmat.shape[0], self.Mmat.shape[1]+3))
            Dmatdiag = DMk / (self.freqs**2)
            d = np.array([Dmatdiag, Dmatdiag*self.toas, Dmatdiag*(self.toas**2)]).T

            self.ptmdescription.append('DM')
            self.ptmdescription.append('DM1')
            self.ptmdescription.append('DM2')
            self.ptmpars = np.append(self.ptmpars, [0.0, 0.0, 0.0])
            
            newM[:,:-3] = self.Mmat
            newM[:,-3:] = d

        self.Mmat = newM
        U, s, Vh = sl.svd(self.Mmat)
        self.Gmat = U[:, (self.Mmat.shape[1]):].copy()
        self.Gcmat = U[:, :(self.Mmat.shape[1])].copy()
        """



    """
    Create auxiliary quantities for the different likelihood functions, like GtF
    etc.

    TODO: make the flow of this function a bit more transparent. Just make a few
    big blocks for the different likelihood functions. This is too cluttered
    """
    def createAuxiliaries(self, Tmax, nfreqs, ndmfreqs, twoComponent=False, \
            nSingleFreqs=0, nSingleDMFreqs=0, likfunc='mark3'): 

        if likfunc == 'mark2':
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)

            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AG = np.dot(self.Amat.T, self.Gmat.T)

        if likfunc == 'mark3' or likfunc == 'mark3fa':
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)

            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)
                self.AG = np.dot(self.Amat.T, self.Gmat.T)

        if likfunc == 'mark4':
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)
            self.avetoas, self.U = dailyaveragequantities(self.toas)
            GtU = np.dot(self.Gmat.T, self.U)
            self.coarseGrainedEquad = True

            self.UtF = np.dot(self.U.T, self.Fmat)
            self.Qamp = 1.0

            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGU = np.dot(self.Amat.T, GtU)

        if likfunc == 'mark4ln':
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)
            self.avetoas, self.U = dailyaveragequantities(self.toas)
            GtU = np.dot(self.Gmat.T, self.U)
            self.coarseGrainedEquad = True

            self.UtF = np.dot(self.U.T, self.Fmat)
            self.Qamp = 1.0

            # Initialise the single frequency with a frequency of 10 / yr
            self.frequencyLinesAdded = nSingleFreqs
            spd = 24 * 3600.0
            spy = 365.25 * spd
            deltaf = 2.3 / spy      # Just some random number
            sfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleFreqs)
            self.SFmat = singleFreqFourierModes(self.toas, np.log10(sfreqs))
            self.FFmat = np.append(self.Fmat, self.SFmat, axis=1)
            self.SFfreqs = np.log10(np.array([sfreqs, sfreqs]).T.flatten())
            GtFF = np.dot(self.Gmat.T, self.FFmat)
            #self.GGtFF = np.dot(self.Gmat, GtFF)

            self.UtFF = np.dot(self.U.T, self.FFmat)

            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGU = np.dot(self.Amat.T, GtU)


        if likfunc == 'mark6' or likfunc == 'mark6fa':
            # Red noise
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)

            # DM
            (self.Fdmmat, self.Fdmfreqs) = fourierdesignmatrix(self.toas, 2*ndmfreqs, Tmax)
            self.Dmat = np.diag(DMk / (self.freqs**2))
            self.DF = np.dot(self.Dmat, self.Fdmmat)
            GtD = np.dot(self.Gmat.T, self.DF)
            self.GGtD = np.dot(self.Gmat, GtD)

            # DM + Red noise stuff (mark6 needs this)
            self.Emat = np.append(self.Fmat, self.DF, axis=1)
            GtE = np.dot(self.Gmat.T, self.Emat)
            self.GGtE = np.dot(self.Gmat, GtE)

            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)
                self.AGD = np.dot(self.Amat.T, GtD)
                self.AGE = np.dot(self.Amat.T, GtE)

        if likfunc == 'mark7':
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)

            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)


        if likfunc == 'mark8':

            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)

            # For the DM stuff
            (self.Fdmmat, self.Fdmfreqs) = fourierdesignmatrix(self.toas, 2*ndmfreqs, Tmax)
            self.Dmat = np.diag(DMk / (self.freqs**2))
            self.DF = np.dot(self.Dmat, self.Fdmmat)
            GtD = np.dot(self.Gmat.T, self.DF)
            self.GGtD = np.dot(self.Gmat, GtD)

            # DM + Red noise stuff
            self.Emat = np.append(self.Fmat, self.DF, axis=1)
            GtE = np.dot(self.Gmat.T, self.Emat)
            self.GGtE = np.dot(self.Gmat, GtE)

            # For a two-component noise model, we need some more stuff done
            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)
                self.AGD = np.dot(self.Amat.T, GtD)
                self.AGE = np.dot(self.Amat.T, GtE)

        if likfunc == 'mark9':
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)

            # Initialise the single frequency with a frequency of 10 / yr
            self.frequencyLinesAdded = nSingleFreqs
            spd = 24 * 3600.0
            spy = 365.25 * spd
            deltaf = 2.3 / spy      # Just some random number
            sfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleFreqs)
            self.SFmat = singleFreqFourierModes(self.toas, np.log10(sfreqs))
            self.FFmat = np.append(self.Fmat, self.SFmat, axis=1)
            self.SFfreqs = np.log10(np.array([sfreqs, sfreqs]).T.flatten())
            GtFF = np.dot(self.Gmat.T, self.FFmat)
            #self.GGtFF = np.dot(self.Gmat, GtFF)

            # For a two-component noise model, we need some more stuff done
            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)
                self.AGFF = np.dot(self.Amat.T, GtFF)

                self.AG = np.dot(self.Amat.T, self.Gmat.T)

        if likfunc == 'mark10':
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
            self.Gr = np.dot(self.Gmat.T, self.residuals)
            self.GGr = np.dot(self.Gmat, self.Gr)
            self.GtF = np.dot(self.Gmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Gmat, self.GtF)

            # For the DM stuff
            (self.Fdmmat, self.Fdmfreqs) = fourierdesignmatrix(self.toas, 2*ndmfreqs, Tmax)
            self.Dmat = np.diag(DMk / (self.freqs**2))
            self.DF = np.dot(self.Dmat, self.Fdmmat)
            GtD = np.dot(self.Gmat.T, self.DF)
            self.GGtD = np.dot(self.Gmat, GtD)

            # DM + Red noise stuff (mark6 needs this)
            self.Emat = np.append(self.Fmat, self.DF, axis=1)
            GtE = np.dot(self.Gmat.T, self.Emat)
            self.GGtE = np.dot(self.Gmat, GtE)

            # Initialise the single frequency with a frequency of 10 / yr
            self.frequencyLinesAdded = nSingleFreqs
            self.dmfrequencyLinesAdded = nSingleDMFreqs
            spd = 24 * 3600.0
            spy = 365.25 * spd
            deltaf = 2.3 / spy      # Just some random number
            sfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleFreqs)
            sdmfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleDMFreqs)
            self.SFmat = singleFreqFourierModes(self.toas, np.log10(sfreqs))
            self.SFdmmat = singleFreqFourierModes(self.toas, np.log10(sdmfreqs))
            self.FFmat = np.append(self.Fmat, self.SFmat, axis=1)
            #self.FFdmmat = np.append(self.Fdmmat, self.SFdmmat, axis=1)
            self.SFfreqs = np.log10(np.array([sfreqs, sfreqs]).T.flatten())
            self.SFdmfreqs = np.log10(np.array([sdmfreqs, sdmfreqs]).T.flatten())
            self.DSF = np.dot(self.Dmat, self.SFdmmat)
            self.DFF = np.append(self.DF, self.DSF, axis=1)

            GtFF = np.dot(self.Gmat.T, self.FFmat)
            #self.GGtFF = np.dot(self.Gmat, GtFF)

            self.EEmat = np.append(self.FFmat, self.DFF, axis=1)
            GtEE = np.dot(self.Gmat.T, self.EEmat)
            self.GGtEE = np.dot(self.Gmat, GtEE)

            # For a two-component noise model, we need some more stuff done
            if twoComponent:
                self.twoComponentNoise = True

                # Diagonalise GtEfG
                GtNeG = np.dot(self.Gmat.T, ((self.toaerrs**2) * self.Gmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)

                self.AGFF = np.dot(self.Amat.T, GtFF)
                self.AGD = np.dot(self.Amat.T, GtD)

                self.AGE = np.dot(self.Amat.T, GtE)
                self.AGEE = np.dot(self.Amat.T, GtEE)



    # When doing Fourier mode selection, like in mark7/mark8, we need an adjusted
    # version of the E-matrix, which only includes certain columns. Select those
    # here
    # bfinc and bfdminc are Boolean arrays indicating which Frequencies to include
    def setLimitedModeAuxiliaries(self, bfinc, bfdminc, likfunc='mark7'):
        bfincnp = np.array(bfinc, dtype=np.bool)
        bfdmincnp = np.array(bfdminc, dtype=np.bool)

        if not (np.all(bfincnp == self.bfinc) and np.all(bfdmincnp == self.bfdminc)):
            if self.bfinc == None or self.bfdminc == None:
                # First RJMCMC step, initialise all RJMCMC ones, too
                self.bcurfinc = bfincnp.copy()
                self.bcurfdminc = bfdmincnp.copy()

            self.bfinc = bfincnp.copy()
            self.bfdminc = bfdmincnp.copy()

            bf = np.array([bfincnp, bfincnp]).T.flatten()
            bfdm = np.array([bfdmincnp, bfdmincnp]).T.flatten()

            if self.twoComponentNoise:
                # For mark8
                # TODO: this selecting is fast, but not general. For general, need
                #       advanced indexing
                self.lAGF = self.AGF[:,bf]

                if not likfunc in ['mark2', 'mark3', 'mark3fa', 'mark4', 'mark7', 'mark9']:
                    self.lAGE = np.append(self.AGE[:,bf], self.AGD[:,bfdm], axis=1)

                if likfunc in ['mark9', 'mark10']:
                    bff = np.append(bf, [True]*self.FFmat.shape[1])
                    self.lAGFF = self.AGFF[:, bff]

                    if likfunc in ['mark10']:
                        bffdm = np.append(bff, bfdm)
                        self.lAGEE = self.AGEE[:, bffdm]
            else:
                # For mark7
                self.lFmat = self.Fmat[:,bf]
                #self.lGGtF = self.GGtF[:,bf]  # Not used

                if not likfunc in ['mark2', 'mark3', 'mark3fa', 'mark4', 'mark7', 'mark9']:
                    self.lEmat = np.append(self.Fmat[:,bf], self.DF[:,bfdm], axis=1)
                    #self.lGGtE = np.append(self.GGtF[:,bf], self.GGtD[:,bfdm], axis=1) # Not used

                if likfunc in ['mark9', 'mark10']:
                    bff = np.append(bf, [True]*self.FFmat.shape[1])
                    #self.lGGtFF = self.GGtFF[:, bff]

                    if likfunc in ['mark10']:
                        bffdm = np.append(bff, bfdm)
                        self.lGGtEE = self.GGtEE[:, bffdm]

    # Just like 'setLimitedModeAuxiliaries', but now with a number as an argument
    def setLimitedModeNumber(self, nbf, nbfdm, likfunc='mark7'):
        bfinc = np.array([0]*self.Fmat.shape[1], dtype=np.bool)
        bfdminc = np.array([0]*self.DF.shape[1], dtype=np.bool)

        bfinc[:nbf] = True
        bfdminc[:nbfdm] = True

        self.setLimitedModeAuxiliaries(bfinc, bfdminc, likfunc=likfunc)


class ptaLikelihood(object):
    # The ptaPulsar objects
    ptapsrs = []

    # The model/signals description
    ptasignals = []

    dimensions = 0
    pmin = None
    pmax = None
    pstart = None
    pwidth = None
    pamplitudeind = None
    initialised = False
    pardes = None

    # What likelihood function to use
    likfunc = 'mark3'

    # Whether we use the option of forcing the frequency lines to be ordered in
    # the prior
    orderFrequencyLines = False

    # Additional informative quantities (reset after RJMCMC jump)
    npf = None      # Number of frequencies per pulsar (red noise/signal)
    npff = None     # Number of frequencies per pulsar (single-freq components)
    npfdm = None    # Number of frequencies per pulsar (DM)
    npe = None      # Number of frequencies per pulsar (rn + DM)
    npobs = None    # Number of observations per pulsar
    npgs = None     # Number of non-projected observations per pulsar (columns Gmat)

    # The Phi, Theta, and Sigma matrices
    Phi = None          # mark1, mark3, mark?, mark6
    Thetavec = None     #               mark?, mark6
    Sigma = None        #        mark3, mark?, mark6
    GNGldet = None      # mark1, mark3, mark?, mark6

    # Other quantities that we do not want to re-initialise every likelihood call
    rGr = None          # mark1, mark3, mark?, mark6
    rGFa = None         # mark1
    aFGFa = None        # mark1
    avec = None         # mark1
    rGF = None          #        mark3, mark?
    rGE = None          #                      mark6
    FGGNGGF = None      #        mark3, mark?
    EGGNGGE = None      #                      mark6
    NGGF = None         #        mark3, mark?  mark6


    def __init__(self, filename=None):
        self.ptapsrs = []
        self.ptasignals = []

        self.dimensions = 0
        self.pmin = None
        self.pmax = None
        self.pstart = None
        self.pwidth = None
        self.pamplitudeind = None
        self.initialised = False
        self.likfunc = 'mark3'
        self.orderFrequencyLines = False

        if filename is not None:
            self.initFromFile(filename)

    def initFromFile(self, filename):
        h5file = h5.File(filename, 'r+')

        # Retrieve the models group
        if not "Data" in h5file:
            h5file.close()
            h5file = None
            raise IOError, "no Data group in hdf5 file"

        datagroup = h5file["Data"]

        # Retrieve the pulsars group
        if not "Pulsars" in datagroup:
            h5file.close()
            h5file = None
            raise IOError, "no Pulsars group in hdf5 file"

        pulsarsgroup = datagroup["Pulsars"]

        psrnames = list(pulsarsgroup)
        h5file.close()
        h5file = None

        for psrname in psrnames:
            newpsr = ptaPulsar()
            newpsr.readFromH5(filename, psrname)
            #newpsr.readFromImagination(filename, psrname)
            self.ptapsrs.append(newpsr)

    def addSignalEfac(self, psrind, index, separateEfacs=False, \
            varyEfac=True, pmin=0.001, pmax=1000.0, pwidth=0.1, pstart=1.0):
        if separateEfacs:
            uflagvals = list(set(self.ptapsrs[psrind].flags))   # uniques
            for flagval in uflagvals:
                newsignal = ptasignal()
                newsignal.pulsarind = psrind
                newsignal.stype = 'efac'
                newsignal.corr = 'single'
                newsignal.flagname = 'efacequad'
                newsignal.flagvalue = flagval

                ind = np.array(self.ptapsrs[psrind].flags) != flagval
                newsignal.Nvec = self.ptapsrs[psrind].toaerrs**2
                newsignal.Nvec[ind] = 0.0

                newsignal.bvary = np.array([varyEfac], dtype=np.bool)
                newsignal.npars = np.sum(newsignal.bvary)
                newsignal.ntotpars = len(newsignal.bvary)

                newsignal.pmin = np.array([pmin])
                newsignal.pmax = np.array([pmax])
                newsignal.pwidth = np.array([pwidth])
                newsignal.pstart = np.array([pstart])

                newsignal.nindex = index
                index += newsignal.npars

                self.ptasignals.append(newsignal)
        else:
            # One efac to rule them all
            newsignal = ptasignal()
            newsignal.pulsarind = psrind
            newsignal.stype = 'efac'
            newsignal.corr = 'single'
            newsignal.flagname = 'pulsarname'
            newsignal.flagvalue = self.ptapsrs[psrind].name
            newsignal.Nvec = self.ptapsrs[psrind].toaerrs**2

            newsignal.bvary = np.array([varyEfac], dtype=np.bool)
            newsignal.npars = np.sum(newsignal.bvary)
            newsignal.ntotpars = len(newsignal.bvary)

            newsignal.pmin = np.array([pmin])
            newsignal.pmax = np.array([pmax])
            newsignal.pwidth = np.array([pwidth])
            newsignal.pstart = np.array([pstart])

            newsignal.nindex = index

            self.ptasignals.append(newsignal)

    def addSignalEquad(self, psrind, index, \
            pmin=-8.8, pmax=-2.0, pwidth=0.1, pstart=-8.0):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind
        newsignal.stype = 'equad'
        newsignal.corr = 'single'
        newsignal.flagname = 'pulsarname'
        newsignal.flagvalue = self.ptapsrs[psrind].name
        newsignal.Nvec = np.ones(len(self.ptapsrs[psrind].toaerrs))
        newsignal.npars = 1
        newsignal.ntotpars = 1
        newsignal.bvary = np.array([1], dtype=np.bool)

        newsignal.pmin = np.array([pmin])
        newsignal.pmax = np.array([pmax])
        newsignal.pwidth = np.array([pwidth])
        newsignal.pstart = np.array([pstart])

        newsignal.nindex = index

        self.ptasignals.append(newsignal)

    def addSignalNoiseFrequencyLine(self, psrind, index, freqindex):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind

        newsignal.stype = 'frequencyline'
        newsignal.npars = 2
        newsignal.ntotpars = 2
        newsignal.bvary = np.array([True, True], dtype=np.bool)
        newsignal.npsrfreqindex = freqindex

        # 1 = frequency, 1 = amplitude
        newsignal.pmin = np.array([-9.0, -18])
        newsignal.pmax = np.array([-4.0, -7.0])
        newsignal.pstart = np.array([-7, -10.0])
        newsignal.pwidth = np.array([0.1, 0.1])

        newsignal.corr = 'single'
        newsignal.nindex = index
        self.ptasignals.append(newsignal)


    def addSignalDMFrequencyLine(self, psrind, index, freqindex):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind

        newsignal.stype = 'dmfrequencyline'
        newsignal.npars = 2
        newsignal.ntotpars = 2
        newsignal.bvary = np.array([True, True], dtype=np.bool)
        newsignal.npsrdmfreqindex = freqindex

        # 1 = frequency, 1 = amplitude
        newsignal.pmin = np.array([-9.0, -18])
        newsignal.pmax = np.array([-4.0, -7.0])
        newsignal.pstart = np.array([-7, -10.0])
        newsignal.pwidth = np.array([0.1, 0.1])

        newsignal.corr = 'single'
        newsignal.nindex = index
        self.ptasignals.append(newsignal)


    def addSignalRedNoise(self, psrind, index, Tmax, \
            noiseModel, fc=None):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind

        if noiseModel=='spectrum':
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * -7.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        elif noiseModel=='powerlaw':
            newsignal.stype = 'powerlaw'
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)
            newsignal.npars = np.sum(newsignal.bvary)
            newsignal.ntotpars = len(newsignal.bvary)

            newsignal.pmin = np.array([-16.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])
        elif noiseModel=='spectralModel':
            # A in sec^3, alpha unitless, fc in log10(yr^{-1})
            newsignal.stype = 'spectralModel'
            newsignal.bvary = np.array([1, 1, 1], dtype=np.bool)

            newsignal.pmin = np.array([-28., 0., -4.])
            newsignal.pmax = np.array([-14., 12., 2.])
            newsignal.pstart = np.array([-22., 2., -1.])
            newsignal.pwidth = np.array([-0.2, 0.1, 0.1])

            if fc is not None:
                newsignal.bvary[2] = False
                newsignal.pstart[2] = fc

            newsignal.npars = np.sum(newsignal.bvary)
            newsignal.ntotpars = len(newsignal.bvary)

        newsignal.corr = 'single'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        self.ptasignals.append(newsignal)

    def addSignalDMV(self, psrind, index, Tmax, \
            dmModel):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind

        if dmModel=='spectrum':
            newsignal.stype = 'dmspectrum'
            newsignal.npars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -14.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * -3.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -7.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        elif dmModel=='powerlaw':
            newsignal.stype = 'dmpowerlaw'
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)
            newsignal.npars = np.sum(newsignal.bvary)
            newsignal.ntotpars = len(newsignal.bvary)

            newsignal.pmin = np.array([-14.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-13.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'single'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        self.ptasignals.append(newsignal)

    # TODO: use the independent GWB frequencies, instead of those of the first pulsar
    def addSignalGWB(self, index, Tmax, \
            gwbModel):
        newsignal = ptasignal()
        newsignal.pulsarind = -1

        if gwbModel=='spectrum':
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        elif gwbModel=='powerlaw':
            newsignal.stype = 'powerlaw'
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)
            newsignal.npars = np.sum(newsignal.bvary)
            newsignal.ntotpars = len(newsignal.bvary)

            newsignal.pmin = np.array([-17.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'gr'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        newsignal.corrmat = hdcorrmat(self.ptapsrs)           # The H&D matrix
        self.ptasignals.append(newsignal)

    def addSignalClock(self, index, Tmax, \
            clockModel):
        newsignal = ptasignal()
        newsignal.pulsarind = -1

        if clockModel=='spectrum':
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        elif clockModel=='powerlaw':
            newsignal.stype = 'powerlaw'
            newsignal.npars = 2
            newsignal.ntotpars = 3
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)

            newsignal.pmin = np.array([-17.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'uniform'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        newsignal.corrmat = np.ones((len(self.ptapsrs), len(self.ptapsrs)))
        self.ptasignals.append(newsignal)

    def addSignalDipole(self, index, Tmax, \
            dipoleModel):
        newsignal = ptasignal()
        newsignal.pulsarind = -1

        if dipoleModel=='spectrum':
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        elif dipoleModel=='powerlaw':
            newsignal.stype = 'powerlaw'
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)
            newsignal.npars = np.sum(newsignal.bvary)
            newsignal.ntotpars = len(newsignal.bvary)

            newsignal.pmin = np.array([-17.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'uniform'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        newsignal.corrmat = dipolecorrmat(self.ptapsrs)
        self.ptasignals.append(newsignal)

    def addSignalAniGWB(self, index, Tmax, \
            anigwbModel, lAniGWB=2):
        newsignal = ptasignal()
        newsignal.pulsarind = -1
        newsignal.aniCorr = aniCorrelations(self.ptapsrs, lAniGWB)
        nclm = newsignal.aniCorr.clmlength()

        if anigwbModel=='spectrum':
            newsignal.stype = 'spectrum'
            newsignal.npars = nclm+int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = nclm+int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1

            newsignal.pmin[-nclm:] = -5.0
            newsignal.pmax[-nclm:] = 5.0
            newsignal.pstart[-nclm:] = 0.0
            newsignal.pwidth[-nclm:] = 0.2
        elif anigwbModel=='powerlaw':
            newsignal.stype = 'powerlaw'
            newsignal.bvary = np.array([1]*(nclm+3), dtype=np.bool)
            #newsignal.bvary[1] = False
            newsignal.bvary[2] = False
            #newsignal.bvary[3] = False
            #newsignal.bvary[5] = False
            newsignal.npars = np.sum(newsignal.bvary)
            newsignal.ntotpars = len(newsignal.bvary)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -5.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 5.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * 0.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.2

            newsignal.pmin[:3] = np.array([-17.0, 1.02, 1.0e-11])
            newsignal.pmax[:3] = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart[:3] = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth[:3] = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'anisotropicgwb'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        self.ptasignals.append(newsignal)

    def addSignalFourierCoeff(self, psrind, index, Tmax, isDM=False):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind
        if isDM:
            newsignal.stype = 'dmfouriercoeff'
            newsignal.npars = len(self.ptapsrs[0].Fdmfreqs)
            newsignal.ntotpars = len(self.ptapsrs[0].Fdmfreqs)
        else:
            newsignal.stype = 'fouriercoeff'
            newsignal.npars = len(self.ptapsrs[0].Ffreqs)
            newsignal.ntotpars = len(self.ptapsrs[0].Ffreqs)
        newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)
        newsignal.corr = 'single'
        newsignal.Tmax = Tmax
        newsignal.nindex = index

        # Since this parameter space is so large, calculate the
        # best first-estimate values of these quantities
        # We assume that many auxiliaries have been set already (is done
        # in initModel, so should be ok)
        # TODO: check whether this works, and make smarter
        npars = newsignal.npars
        psr = self.ptapsrs[newsignal.pulsarind]

        """
        if isDM:
            NGGF = np.array([(1.0/(psr.toaerrs**2)) * psr.GGtD[:,ii] for ii in range(psr.Fmat.shape[1])]).T
            FGGNGGF = np.dot(psr.GGtD.T, NGGF)
        else:
            NGGF = np.array([(1.0/(psr.toaerrs**2)) * psr.GGtF[:,ii] for ii in range(psr.Fmat.shape[1])]).T
            FGGNGGF = np.dot(psr.GGtF.T, NGGF)
        rGGNGGF = np.dot(psr.GGr, NGGF)

        try:
            cf = sl.cho_factor(FGGNGGF)
            fest = sl.cho_solve(cf, rGGNGGF)
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(FGGNGGF)
            if not np.all(s > 0):
                raise ValueError("ERROR: F^{T}F singular according to SVD")

            fest = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, rGGNGGF)))

        newsignal.pmin = -1.0e4*np.abs(fest)
        newsignal.pmax = 1.0e4*np.abs(fest)
        newsignal.pstart = fest
        newsignal.pwidth = 1.0e-1*np.abs(fest)

        self.ptasignals.append(newsignal)
        """




    # TODO: see if we can implement the RJMCMC for the Fourier modes
    def allocateAuxiliaries(self):
        # First figure out how large we have to make the arrays
        npsrs = len(self.ptapsrs)
        self.npf = np.zeros(npsrs, dtype=np.int)
        self.npu = np.zeros(npsrs, dtype=np.int)
        self.npff = np.zeros(npsrs, dtype=np.int)
        self.npfdm = np.zeros(npsrs, dtype=np.int)
        self.npffdm = np.zeros(npsrs, dtype=np.int)
        self.npobs = np.zeros(npsrs, dtype=np.int)
        self.npgs = np.zeros(npsrs, dtype=np.int)
        for ii in range(npsrs):
            if not self.likfunc in ['mark2']:
                self.npf[ii] = len(self.ptapsrs[ii].Ffreqs)
                self.npff[ii] = self.npf[ii]

            if self.likfunc in ['mark4ln', 'mark9', 'mark10']:
                self.npff[ii] += len(self.ptapsrs[ii].SFfreqs)

            if self.likfunc in ['mark4', 'mark4ln']:
                self.npu[ii] = len(self.ptapsrs[ii].avetoas)

            if self.likfunc in ['mark6', 'mark6fa', 'mark8', 'mark10']:
                self.npfdm[ii] = len(self.ptapsrs[ii].Fdmfreqs)
                self.npffdm[ii] = len(self.ptapsrs[ii].Fdmfreqs)

            if self.likfunc in ['mark10']:
                self.npffdm[ii] += len(self.ptapsrs[ii].SFdmfreqs)

            self.npobs[ii] = len(self.ptapsrs[ii].toas)
            self.npgs[ii] = self.ptapsrs[ii].Gmat.shape[1]
            self.ptapsrs[ii].Nvec = np.zeros(len(self.ptapsrs[ii].toas))
            self.ptapsrs[ii].Nwvec = np.zeros(self.ptapsrs[ii].Gmat.shape[1])

        if self.likfunc == 'mark1':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGFa = np.zeros(npsrs)
            self.aFGFa = np.zeros(npsrs)
            self.avec = np.zeros(np.sum(self.npf))
        elif self.likfunc == 'mark2':
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
        elif self.likfunc == 'mark3' or self.likfunc == 'mark7' \
                or self.likfunc == 'mark3fa':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.Sigma = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGF = np.zeros(np.sum(self.npf))
            self.FGGNGGF = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        elif self.likfunc == 'mark4':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.Sigma = np.zeros((np.sum(self.npu), np.sum(self.npu)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGU = np.zeros(np.sum(self.npu))
            self.UGGNGGU = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        elif self.likfunc == 'mark4ln':
            self.Phi = np.zeros((np.sum(self.npff), np.sum(self.npff)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.Sigma = np.zeros((np.sum(self.npu), np.sum(self.npu)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGU = np.zeros(np.sum(self.npu))
            self.UGGNGGU = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        elif self.likfunc == 'mark6' or self.likfunc == 'mark8' \
                or self.likfunc == 'mark6fa':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Sigma = np.zeros((np.sum(self.npf)+np.sum(self.npfdm), \
                    np.sum(self.npf)+np.sum(self.npfdm)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGE = np.zeros(np.sum(self.npf)+np.sum(self.npfdm))
            self.EGGNGGE = np.zeros((np.sum(self.npf)+np.sum(self.npfdm), np.sum(self.npf)+np.sum(self.npfdm)))
        elif self.likfunc == 'mark9':
            self.Phi = np.zeros((np.sum(self.npff), np.sum(self.npff)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.Sigma = np.zeros((np.sum(self.npff), np.sum(self.npff)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGF = np.zeros(np.sum(self.npff))
            self.FGGNGGF = np.zeros((np.sum(self.npff), np.sum(self.npff)))
        elif self.likfunc == 'mark10':
            self.Phi = np.zeros((np.sum(self.npff), np.sum(self.npff)))
            self.Sigma = np.zeros((np.sum(self.npff)+np.sum(self.npffdm), \
                    np.sum(self.npff)+np.sum(self.npffdm)))
            self.Thetavec = np.zeros(np.sum(self.npffdm))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGE = np.zeros(np.sum(self.npff)+np.sum(self.npffdm))
            self.EGGNGGE = np.zeros((np.sum(self.npff)+np.sum(self.npffdm), \
                    np.sum(self.npff)+np.sum(self.npffdm)))


    # Initialise the model
    def initModel(self, nfreqmodes=20, ndmfreqmodes=None, \
            incRedNoise=False, noiseModel='powerlaw', fc=None, \
            incDM=False, dmModel='powerlaw', \
            incClock=False, clockModel='powerlaw', \
            incGWB=False, gwbModel='powerlaw', \
            incDipole=False, dipoleModel='powerlaw', \
            incAniGWB=False, anigwbModel='powerlaw', lAniGWB=1, \
            varyEfac=False, incEquad=False, separateEfacs=False, \
            incSingleFreqNoise=False, \
                                        # True
            singlePulsarMultipleFreqNoise=None, \
                                        # [True, ..., False]
            multiplePulsarMultipleFreqNoise=None, \
                                        # [0, 3, 2, ..., 4]
            dmFrequencyLines=None, \
                                        # [0, 3, 2, ..., 4]
            orderFrequencyLines=False, \
            likfunc='mark3'):
        # For every pulsar, construct the auxiliary quantities like the Fourier
        # design matrix etc
        if len(self.ptapsrs) < 1:
            raise IOError, "no pulsars found in hdf5 file"

        Tstart = np.min(self.ptapsrs[0].toas)
        Tfinish = np.max(self.ptapsrs[0].toas)
        self.likfunc = likfunc
        self.orderFrequencyLines = orderFrequencyLines

        for m2psr in self.ptapsrs:
            Tstart = np.min([np.min(self.ptapsrs[0].toas), Tstart])
            Tfinish = np.max([np.max(self.ptapsrs[0].toas), Tfinish])

        # After processing the parameters, store the number of single frequency lines
        psrSingleFreqNoiseModes = np.zeros(len(self.ptapsrs), dtype=np.int)
        psrSingleDMFreqNoiseModes = np.zeros(len(self.ptapsrs), dtype=np.int)

        # Total duration of the experiment
        Tmax = Tfinish - Tstart
        pindex = 0
        for m2psr in self.ptapsrs:
            if incDM:
                m2psr.addDMQuadratic()

            if ndmfreqmodes is None:
                ndmfreqmodes = nfreqmodes

            nSingleFreqs = 0
            if incSingleFreqNoise:
                nSingleFreqs = 1
                psrSingleFreqNoiseModes[pindex] = 1
            elif singlePulsarMultipleFreqNoise is not None:
                if singlePulsarMultipleFreqNoise[pindex]:
                    nSingleFreqs = 1
                    psrSingleFreqNoiseModes[pindex] = 1
            elif multiplePulsarMultipleFreqNoise is not None:
                nSingleFreqs = multiplePulsarMultipleFreqNoise[pindex]
                psrSingleFreqNoiseModes[pindex] = multiplePulsarMultipleFreqNoise[pindex]

            nSingleDMFreqs = 0
            if dmFrequencyLines is not None:
                nSingleDMFreqs = dmFrequencyLines[pindex]
                psrSingleDMFreqNoiseModes[pindex] = dmFrequencyLines[pindex]

            m2psr.createAuxiliaries(Tmax, nfreqmodes, ndmfreqmodes, not separateEfacs, \
                            nSingleFreqs=nSingleFreqs, nSingleDMFreqs=nSingleDMFreqs, \
                                    likfunc=likfunc)

            # When selecting Fourier modes, like in mark7/mark8, the binclude vector
            # indicates whether or not a frequency is included in the likelihood. By
            # default they are all 'on'
            if self.likfunc == 'mark7' or self.likfunc == 'mark8':
                m2psr.setLimitedModeAuxiliaries([1]*nfreqmodes, [1]*ndmfreqmodes, \
                        likfunc=self.likfunc)

            pindex += 1

        # Initialise the ptasignal objects
        # Currently: one efac per pulsar, and red noise
        self.ptasignals = []
        index = 0
        for ii in range(len(self.ptapsrs)):
            # When adding efac signals, there may be many
            noldsignals = len(self.ptasignals)
            self.addSignalEfac(ii, index, separateEfacs, varyEfac)
            nnewsignals = len(self.ptasignals)
            for jj in range(noldsignals, nnewsignals):
                index += self.ptasignals[jj].npars

            if incEquad:
                self.addSignalEquad(ii, index)
                index += self.ptasignals[-1].npars

            if incRedNoise:
                self.addSignalRedNoise(ii, index, Tmax, noiseModel, fc)
                index += self.ptasignals[-1].npars

            if incDM:
                self.addSignalDMV(ii, index, Tmax, dmModel)
                index += self.ptasignals[-1].npars

            for jj in range(psrSingleFreqNoiseModes[ii]):
                self.addSignalNoiseFrequencyLine(ii, index, jj)
                index += self.ptasignals[-1].npars

            for jj in range(psrSingleDMFreqNoiseModes[ii]):
                self.addSignalDMFrequencyLine(ii, index, jj)
                index += self.ptasignals[-1].npars

        if incGWB:
            self.addSignalGWB(index, Tmax, gwbModel)
            index += self.ptasignals[-1].npars

        if incClock:
            self.addSignalGWB(index, Tmax, clockModel)
            index += self.ptasignals[-1].npars

        if incDipole:
            self.addSignalDipole(index, Tmax, dipoleModel)
            index += self.ptasignals[-1].npars

        if incAniGWB:
            self.addSignalAniGWB(index, Tmax, anigwbModel, lAniGWB)
            index += self.ptasignals[-1].npars


        # If the frequency coefficients are included explicitly (mark1
        # likelihood), we need a couple of extra signals
        if likfunc=='mark1':
            for ii in range(len(self.ptapsrs)):
                self.addSignalFourierCoeff(ii, index, Tmax)
                index += self.ptasignals[-1].npars

                if incDM:
                    self.addSignalFourierCoeff(ii, index, Tmax, isDM=True)
                    index += self.ptasignals[-1].npars

        self.allocateAuxiliaries()
        self.initPrior()
        self.pardes = self.getModelParameterList()

    def getModelParameterList(self):
        pardes = []

        for ii in range(len(self.ptasignals)):
            sig = self.ptasignals[ii]

            pindex = 0
            for jj in range(sig.ntotpars):
                if sig.bvary[jj]:
                    # This parameter is in the mcmc
                    # TODO: the parameter index for varying/nonvarying
                    # parameters is inconsistent throughout the code
                    index = sig.nindex + pindex
                    pindex += 1
                else:
                    index = -1

                psrindex = sig.pulsarind
                if sig.stype == 'efac':
                    flagname = sig.flagname
                    flagvalue = 'efac'+sig.flagvalue
                elif sig.stype == 'equad':
                    flagname = sig.flagname
                    flagvalue = 'equad'+sig.flagvalue
                elif sig.stype == 'spectrum':
                    flagname = 'frequency'

                    if jj >= len(self.ptapsrs[psrindex].Ffreqs)/2:
                        # clmind is index of clm's, plus one, since we do not
                        # model the c_00 term explicitly like that (it is the
                        # amplitude)
                        #     0            l=0     m=0
                        #   1 2 3          l=1     m=-1, 0, -1
                        # 4 5 6 7 8 etc.   l=2     m=-2, -1, 0, 1, 2
                        clmind = jj - len(self.ptapsrs[psrindex].Ffreqs)/2 + 1
                        lani = int(np.sqrt(clmind))
                        mani = clmind - lani*(lani+1)
                        flagvalue = 'C_(' + str(lani) + ',' + str(mani) + ')'
                    else:
                        flagvalue = str(self.ptapsrs[psrindex].Ffreqs[2*jj])
                elif sig.stype == 'dmspectrum':
                    flagname = 'dmfrequency'
                    flagvalue = str(self.ptapsrs[psrindex].Fdmfreqs[2*jj])
                elif sig.stype == 'powerlaw':
                    flagname = 'powerlaw'

                    if jj < 3:
                        flagvalue = ['RN-Amplitude', 'RN-spectral-index', 'low-frequency-cutoff'][jj]
                    else:
                        # Index counting same as above
                        clmind = jj - 3 + 1
                        lani = int(np.sqrt(clmind))
                        mani = clmind - lani*(lani+1)
                        flagvalue = 'C_(' + str(lani) + ',' + str(mani) + ')'
                elif sig.stype == 'dmpowerlaw':
                    flagname = 'dmpowerlaw'
                    flagvalue = ['DM-Amplitude', 'DM-spectral-index', 'low-frequency-cutoff'][jj]
                elif sig.stype == 'spectralModel':
                    flagname = 'spectralModel'
                    flagvalue = ['SM-Amplitude', 'SM-spectral-index', 'SM-corner-frequency'][jj]
                elif sig.stype == 'frequencyline':
                    flagname = 'frequencyline'
                    flagvalue = ['Line-Freq', 'Line-Ampl'][jj]
                else:
                    flagname = 'none'
                    flagvalue = 'none'

                pardes.append(\
                        {'index': index, 'pulsar': psrindex, 'sigindex': ii, \
                            'sigtype': sig.stype, 'correlation': sig.corr, \
                            'name': flagname, 'id': flagvalue})

        return pardes


    """
    Once a model is defined, it can be useful to have all the parameter names
    that enter in the MCMC stored in a file. This function does that, in the
    format: index   psrindex    stype   corr    flagname    flagvalue
    
    Example
    0   0       efac        single      pulsarname  J0030+0451
    1   1       efac        single      pulsarname  J1600-3053
    2   0       spectrum    single      frequency   1.0e-7
    3   0       spectrum    single      frequency   2.0e-7
    4   1       spectrum    single      frequency   1.0e-7
    5   1 -     spectrum    single      frequency   2.0e-7
    6   -1      powerlaw    gr          powerlaw   amplitude
    7   -1      powerlaw    gr          powerlaw   spectral-index

    As you see, 'flagname' and 'flagvalue' carry information about flags for
    efac parameters, for other signals they describe what parameter of the
    signal we are indicating - or the frequency for a spectrum

    This function should in principle always be automatically called by a
    sampler, so that parameter names/model definitions are always saved
    """
    def saveModelParameters(self, filename):
        fil = open(filename, "w")

        for ii in range(len(self.pardes)):
            fil.write("{0:d} \t{1:d} \t{2:s} \t{3:s} \t{4:s} \t{5:s}\n".format(\
                    self.pardes[ii]['index'],
                    self.pardes[ii]['pulsar'],
                    self.pardes[ii]['sigtype'],
                    self.pardes[ii]['correlation'],
                    self.pardes[ii]['name'],
                    self.pardes[ii]['id']))

        fil.close

    """
    Re-calculate the number of varying parameters per signal, and the number of
    dimensions in total.
    """
    def setDimensions(self):
        self.dimensions = 0
        for m2signal in self.ptasignals:
            m2signal.npars = np.sum(m2signal.bvary)
            self.dimensions += m2signal.npars


    """
    Before being able to run the likelihood, we need to initialise the prior

    """
    def initPrior(self):
        self.setDimensions()

        self.pmin = np.zeros(self.dimensions)
        self.pmax = np.zeros(self.dimensions)
        self.pstart = np.zeros(self.dimensions)
        self.pwidth = np.zeros(self.dimensions)

        index = 0
        for m2signal in self.ptasignals:
            for ii in range(m2signal.ntotpars):
                if m2signal.bvary[ii]:
                    self.pmin[index] = m2signal.pmin[ii]
                    self.pmax[index] = m2signal.pmax[ii]
                    self.pwidth[index] = m2signal.pwidth[ii]
                    self.pstart[index] = m2signal.pstart[ii]
                    index += 1

    """
    Return a list of all efac parameter numbers, their names, and the pulsar
    they belong to
    """
    def getEfacNumbers(self):
        parind = []
        psrind = []
        names = []

        for ii in range(len(self.ptasignals)):
            if self.ptasignals[ii].stype == 'efac' and self.ptasignals[ii].bvary[0]:
                parind.append(self.ptasignals[ii].nindex)
                psrind.append(self.ptasignals[ii].pulsarind)
                names.append(self.ptasignals[ii].flagvalue)

        return (parind, psrind, names)

    """
    Return a list of all spectrum signals: signal name, start-par, stop-par, and
    the actual frequencies

    TODO: parameters can be non-varying. Take that into accoutn as well
    """
    def getSpectraNumbers(self):
        signame = []
        signameshort = []
        parmin = []
        parmax = []
        freqs = []
        for ii in range(len(self.ptasignals)):
            if self.ptasignals[ii].stype == 'spectrum' or self.ptasignals[ii].stype == 'dmspectrum':
                if self.ptasignals[ii].stype == 'spectrum' and self.ptasignals[ii].corr == 'single':
                    signame.append('Red noise ' + self.ptapsrs[self.ptasignals[ii].pulsarind].name)
                    signameshort.append('rnspectrum-' + self.ptapsrs[self.ptasignals[ii].pulsarind].name)
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif self.ptasignals[ii].stype == 'spectrum' and self.ptasignals[ii].corr == 'gr':
                    signame.append('GWB spectrum')
                    signameshort.append('gwbspectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif self.ptasignals[ii].stype == 'spectrum' and self.ptasignals[ii].corr == 'uniform':
                    signame.append('Clock spectrum')
                    signameshort.append('clockspectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif self.ptasignals[ii].stype == 'spectrum' and self.ptasignals[ii].corr == 'dipole':
                    signame.append('Dipole spectrum')
                    signameshort.append('dipolespectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif self.ptasignals[ii].stype == 'spectrum' and self.ptasignals[ii].corr == 'anisotropicgwb':
                    signame.append('Anisotropy spectrum')
                    signameshort.append('anisotropyspectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif self.ptasignals[ii].stype == 'dmspectrum':
                    signame.append('DM variation ' + self.ptapsrs[self.ptasignals[ii].pulsarind].name)
                    signameshort.append('dmspectrum-' + self.ptapsrs[self.ptasignals[ii].pulsarind].name)
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[self.ptasignals[ii].pulsarind].Fdmfreqs)))))
                else:
                    signame.append('Spectrum')
                    signameshort.append('spectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))

                parmin.append(self.ptasignals[ii].nindex)
                parmax.append(self.ptasignals[ii].nindex+self.ptasignals[ii].npars)

        return (signame, signameshort, parmin, parmax, freqs)


    """
    Loop over all signals, and fill the diagonal pulsar noise covariance matrix
    (based on efac/equad)
    For two-component noise model, fill the total weights vector
    """
    def setPsrNoise(self, parameters):
        # For every pulsar, set the noise vector to zero
        for m2psr in self.ptapsrs:
            if m2psr.twoComponentNoise:
                m2psr.Nwvec[:] = 0
            #else:
            m2psr.Nvec[:] = 0

        # Loop over all white noise signals, and fill the pulsar Nvec
        for m2signal in self.ptasignals:
            if m2signal.stype == 'efac':
                if m2signal.npars == 1:
                    pefac = parameters[m2signal.nindex]
                else:
                    pefac = m2signal.pstart[0]

                if self.ptapsrs[m2signal.pulsarind].twoComponentNoise:
                    self.ptapsrs[m2signal.pulsarind].Nwvec += \
                            self.ptapsrs[m2signal.pulsarind].Wvec * pefac**2
                #else:
                self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pefac**2

                #if m2signal.bvary[0]:
                #    pefac = parameters[m2signal.nindex]
                #else:
                #    pefac = parameters[m2signal.ntotindex]
                #self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pefac**2
            elif m2signal.stype == 'equad' and \
                    not self.ptapsrs[m2signal.pulsarind].coarseGrainedEquad:
                if m2signal.npars == 1:
                    pequadsqr = 10**(2*parameters[m2signal.nindex])
                else:
                    pequadsqr = 10**(2*m2signal.pstart[0])

                if self.ptapsrs[m2signal.pulsarind].twoComponentNoise:
                    self.ptapsrs[m2signal.pulsarind].Nwvec += pequadsqr
                #else:
                self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pequadsqr
            elif m2signal.stype == 'equad' and \
                    self.ptapsrs[m2signal.pulsarind].coarseGrainedEquad:
                if m2signal.npars == 1:
                    pequadsqr = 10**(2*parameters[m2signal.nindex])
                else:
                    pequadsqr = 10**(2*m2signal.pstart[0])

                self.ptapsrs[m2signal.pulsarind].Qamp = pequadsqr

                #if m2signal.bvary[0]:
                #    pequadsqr = 10**(2*parameters[m2signal.nindex])
                #else:
                #    pequadsqr = 10**(2*parameters[m2signal.ntotindex])
                #self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pequadsqr


    """
    Loop over all signals, and fill the phi matrix. This function assumes that
    the self.Phi matrix has already been allocated

    In this version, the DM variations are included in self.Thetavec

    selection allows the user to specify which signals to include. By
    default=all

    TODO: if the number of possible modes gets really large, but the number of
          actually selected modes (modes, not signals) is not, this function
          becomes the computational bottleneck. Make a version of this that only
          constructs the required elements
    """
    def constructPhiAndTheta(self, parameters, selection=None):
        self.Phi[:] = 0         # Start with a fresh matrix
        self.Thetavec[:] = 0    # ''
        npsrs = len(self.ptapsrs)

        if selection is None:
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # Loop over all signals, and fill the phi matrix
        #for m2signal in self.ptasignals:
        for ss in range(len(self.ptasignals)):
            m2signal = self.ptasignals[ss]
            if selection[ss]:
                # Create a parameters array for this particular signal
                sparameters = m2signal.pstart.copy()
                sparameters[m2signal.bvary] = \
                        parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]
                if m2signal.stype == 'spectrum':
                    if m2signal.corr == 'single':
                        findex = np.sum(self.npff[:m2signal.pulsarind])
                        # nfreq = int(self.npf[m2signal.pulsarind]/2)
                        nfreq = m2signal.npars

                        # Pcdoubled is an array where every element of the parameters
                        # of this m2signal is repeated once (e.g. [1, 1, 3, 3, 2, 2, 5, 5, ...]

                        pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                        # Fill the phi matrix
                        di = np.diag_indices(2*nfreq)
                        self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += 10**pcdoubled
                    elif m2signal.corr in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                        nfreq = m2signal.npars

                        if m2signal.corr in ['gr', 'uniform', 'dipole']:
                            pcdoubled = np.array([sparameters, sparameters]).T.flatten()
                            corrmat = m2signal.corrmat
                        elif m2signal.corr == 'anisotropicgwb':
                            nclm = m2signal.aniCorr.clmlength()
                            pcdoubled = np.array([\
                                    sparameters[:-nclm],\
                                    sparameters[:-nclm]]).T.flatten()
                            clm = sparameters[-nclm:]
                            corrmat = m2signal.aniCorr.corrmat(clm)

                        indexa = 0
                        indexb = 0
                        for aa in range(npsrs):
                            for bb in range(npsrs):
                                # Some pulsars may have fewer frequencies than
                                # others (right?). So only use overlapping ones
                                nof = np.min([self.npf[aa], self.npf[bb], 2*nfreq])
                                di = np.diag_indices(nof)
                                self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += 10**pcdoubled[:nof] * corrmat[aa, bb]
                                indexb += self.npff[bb]
                            indexb = 0
                            indexa += self.npff[aa]
                elif m2signal.stype == 'dmspectrum':
                    if m2signal.corr == 'single':
                        findex = np.sum(self.npffdm[:m2signal.pulsarind])
                        nfreq = int(self.npfdm[m2signal.pulsarind]/2)

                        pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                        # Fill the Theta matrix
                        self.Thetavec[findex:findex+2*nfreq] += 10**pcdoubled
                elif m2signal.stype == 'powerlaw':
                    spd = 24 * 3600.0
                    spy = 365.25 * spd
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    if m2signal.corr == 'single':
                        findex = np.sum(self.npff[:m2signal.pulsarind])
                        nfreq = int(self.npf[m2signal.pulsarind]/2)
                        freqpy = self.ptapsrs[m2signal.pulsarind].Ffreqs * spy
                        pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)

                        # Fill the phi matrix
                        di = np.diag_indices(2*nfreq)
                        self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                    elif m2signal.corr in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                        freqpy = self.ptapsrs[0].Ffreqs * spy
                        pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)
                        nfreq = len(freqpy)

                        if m2signal.corr in ['gr', 'uniform', 'dipole']:
                            corrmat = m2signal.corrmat
                        elif m2signal.corr == 'anisotropicgwb':
                            nclm = m2signal.aniCorr.clmlength()
                            clm = sparameters[-nclm:]
                            corrmat = m2signal.aniCorr.corrmat(clm)

                        indexa = 0
                        indexb = 0
                        for aa in range(npsrs):
                            for bb in range(npsrs):
                                # Some pulsars may have fewer frequencies than
                                # others (right?). So only use overlapping ones
                                nof = np.min([self.npf[aa], self.npf[bb]])
                                if nof > nfreq:
                                    raise IOError, "ERROR: nof > nfreq. Adjust GWB freqs"

                                di = np.diag_indices(nof)
                                self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * corrmat[aa, bb]
                                indexb += self.npff[bb]
                            indexb = 0
                            indexa += self.npff[aa]
                elif m2signal.stype == 'spectralModel':
                    spd = 24 * 3600.0
                    spy = 365.25 * spd
                    Amp = 10**sparameters[0]
                    alpha = sparameters[1]
                    fc = 10**sparameters[2] / spy

                    if m2signal.corr == 'single':
                        findex = np.sum(self.npff[:m2signal.pulsarind])
                        nfreq = int(self.npf[m2signal.pulsarind]/2)
                        freqpy = self.ptapsrs[m2signal.pulsarind].Ffreqs
                        pcdoubled = (Amp * spy**3 / m2signal.Tmax) * ((1 + (freqpy/fc)**2)**(-0.5*alpha))

                        #pcdoubled = (Amp * spy**3 / (m2signal.Tmax)) * freqpy ** (-Si)

                        # Fill the phi matrix
                        di = np.diag_indices(2*nfreq)
                        self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                    elif m2signal.corr in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                        freqpy = self.ptapsrs[0].Ffreqs * spy
                        pcdoubled = (Amp * spy**3 / m2signal.Tmax) / \
                                ((1 + (freqpy/fc)**2)**(-0.5*alpha))
                        nfreq = len(freqpy)

                        if m2signal.corr in ['gr', 'uniform', 'dipole']:
                            corrmat = m2signal.corrmat
                        elif m2signal.corr == 'anisotropicgwb':
                            nclm = m2signal.aniCorr.clmlength()
                            clm = sparameters[-nclm:m2signal.nindex+m2signal.npars]
                            corrmat = m2signal.aniCorr.corrmat(clm)

                        indexa = 0
                        indexb = 0
                        for aa in range(npsrs):
                            for bb in range(npsrs):
                                # Some pulsars may have fewer frequencies than
                                # others (right?). So only use overlapping ones
                                nof = np.min([self.npf[aa], self.npf[bb]])
                                if nof > nfreq:
                                    raise IOError, "ERROR: nof > nfreq. Adjust GWB freqs"

                                di = np.diag_indices(nof)
                                self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * corrmat[aa, bb]
                                indexb += self.npff[bb]
                            indexb = 0
                            indexa += self.npff[aa]
                elif m2signal.stype == 'dmpowerlaw':
                    spd = 24 * 3600.0
                    spy = 365.25 * spd
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    if m2signal.corr == 'single':
                        findex = np.sum(self.npffdm[:m2signal.pulsarind])
                        nfreq = int(self.npfdm[m2signal.pulsarind]/2)
                        freqpy = self.ptapsrs[m2signal.pulsarind].Fdmfreqs * spy
                        # TODO: change the units of the DM signal
                        pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)

                        # Fill the Theta matrix
                        self.Thetavec[findex:findex+2*nfreq] += pcdoubled
                elif m2signal.stype == 'frequencyline':
                    # For a frequency line, the FFmatrix is assumed to be set elsewhere
                    findex = np.sum(self.npff[:m2signal.pulsarind]) + \
                            self.npf[m2signal.pulsarind] + 2*m2signal.npsrfreqindex

                    pcdoubled = np.array([sparameters[1], sparameters[1]])
                    di = np.diag_indices(2)

                    self.Phi[findex:findex+2, findex:findex+2][di] += 10**pcdoubled
                elif m2signal.stype == 'dmfrequencyline':
                    # For a DM frequency line, the DFF is assumed to be set elsewhere
                    findex = np.sum(self.npffdm[:m2signal.pulsarind]) + \
                            self.npfdm[m2signal.pulsarind] + 2*m2signal.npsrdmfreqindex

                    pcdoubled = np.array([sparameters[1], sparameters[1]])
                    self.Thetavec[findex:findex+2] += 10**pcdoubled




    """
    Set the Auxiliary quantities for mark7loglikelihood in all the pulsars,
    based on the psrbfinc boolean arrays. It returns a boolean array for both
    phi and theta, indicating which elements of the covariance matrix to use in
    the likelihood, and similar boolean arrays for the previously ('the current')
    accepted t-d position for use in the prior

    NOTE: Does not work yet with single-frequency lines
    """
    def prepareLimFreqIndicators(self, psrbfinc=None, psrbfdminc=None):
        # Because it's mark7/mark8, also set the number of frequency modes
        # Also set the 'global' index selector that we'll use for the Phi and
        # Theta matrices
        npsrs = len(self.ptapsrs)
        self.lnpf = np.zeros(npsrs, dtype=np.int)
        self.lnpfdm = np.zeros(npsrs, dtype=np.int)

        find = 0
        dmfind = 0
        for ii in range(npsrs):
            flen = int(self.ptapsrs[ii].Fmat.shape[1]/2)
            fdmlen = int(self.ptapsrs[ii].DF.shape[1]/2)
            if psrbfinc != None and psrbfdminc != None:
                self.ptapsrs[ii].setLimitedModeAuxiliaries( \
                        psrbfinc[find:find+flen], \
                        psrbfdminc[dmfind:dmfind+fdmlen], \
                        likfunc=self.likfunc)

            # Register how many modes we are including (2*number of freqs)
            self.lnpf[ii] = 2*np.sum(self.ptapsrs[ii].bfinc)
            self.lnpfdm[ii] = 2*np.sum(self.ptapsrs[ii].bfdminc)

            find += flen
            dmfind += fdmlen

        # find and dmfind now hold the total number of frequencies
        bfind = np.array([1]*find, dtype=np.bool)
        bfdmind = np.array([1]*dmfind, dtype=np.bool)
        bcurfind = np.array([1]*find, dtype=np.bool)
        bcurfdmind = np.array([1]*dmfind, dtype=np.bool)
        find = 0
        dmfind = 0
        for ii in range(npsrs):
            flen = int(self.ptapsrs[ii].Fmat.shape[1]/2)
            fdmlen = int(self.ptapsrs[ii].DF.shape[1]/2)
            bfind[find:find+flen] = self.ptapsrs[ii].bfinc
            bfdmind[dmfind:dmfind+fdmlen] = self.ptapsrs[ii].bfdminc
            bcurfind[find:find+flen] = self.ptapsrs[ii].bcurfinc
            bcurfdmind[dmfind:dmfind+fdmlen] = self.ptapsrs[ii].bcurfdminc

            find += flen
            dmfind += fdmlen

        return bfind, bfdmind, bcurfind, bcurfdmind

    """
    Convert a number of frequencies for RN and DMV to a boolean array that
    indicates which frequencies to use

    NOTE: Does not work yet with single-frequency lines
    """
    def getPsrLimFreqFromNumbers(self, psrnfinc, psrnfdminc):
        npsrs = len(self.ptapsrs)
        flentot = int(np.sum(self.npf) / 2)
        fdmlentot = int(np.sum(self.npfdm) / 2)

        psrbfinc = np.array([0]*flentot, dtype=np.bool)
        psrbfdminc = np.array([0]*fdmlentot, dtype=np.bool)
        find = 0
        dmfind = 0
        for ii in range(npsrs):
            flen = int(self.npf[ii]/2)
            fdmlen = int(self.npfdm[ii]/2)

            psrbfinc[find:find+psrnfinc[ii]] = True
            psrbfdminc[dmfind:dmfind+psrnfdminc[ii]] = True
            find += flen
            dmfind += fdmlen

        return psrbfinc, psrbfdminc

    """
    When doing an RJMCMC, this function proposed the next possible jump in both
    RN and DM spaces. This only works for a model with a single pulsar. Zero
    modes for RN or DM are not allowed.

    NOTE: Does not work yet with single-frequency lines
    """
    def proposeNextDimJump(self, stepsizemod1=1, stepsizemod2=1):
        if len(self.ptapsrs) > 1:
            raise ValueError("ERROR: modelNrToArray can only work on single psr")

        # The maximum model numbers
        maxmod1 = int(self.ptapsrs[0].Fmat.shape[1]/2)
        maxmod2 = int(self.ptapsrs[0].DF.shape[1]/2)

        if maxmod1 == 0 or maxmod2 == 0:
            raise ValueError("ERROR: RN or DMV dimension == 0")

        # The current model numbers
        curmod1 = np.sum(self.ptapsrs[0].bfinc)
        curmod2 = np.sum(self.ptapsrs[0].bfdminc)

        # The proposed model numbers
        propmod1 = curmod1
        propmod2 = curmod2

        # Draw a number from [-stepsizemodx, .., -1, 1, .., stepsizemodx]
        def drawssm(stepsize):
            step = np.random.randint(1, stepsize+1)
            sign = -1 + 2 * np.random.randint(0, 2)
            return sign * step

        # Produce the next step, in between modmin, modmax
        #def drawstep(stepsize, modmin, modmax):
        #    pass

        # Either jump in one dimension, or the other. Not both
        if np.random.rand() < 0.5:
            # Produce a valid step
            propmod1 = curmod1 + drawssm(stepsizemod1)
            if propmod1 < 1 or propmod1 > maxmod1:
                propmod1 = curmod1
        else:
            propmod2 = curmod2 + drawssm(stepsizemod2)
            if propmod2 < 1 or propmod2 > maxmod2:
                propmod2 = curmod2

        return propmod1, propmod2

    """
    If we accept a transdimensional RJMCMC jump, adjust the 'current mode'
    indicators, so that we know that we need to update the priors.

    Also, this function returns the *real* logposterior (given the temporary
    one); When we propose a trans-dimensional jump, we sample the extra dimensional
    parameters from the prior. Because of this, the prior for these two
    parameters is not included when comparing the RJMCMC acceptance. However,
    after we have accepted such a jump, we do need to record the _full_
    posterior, which includes the prior of these extra parameters. This function
    calculates that additional prior, and adds it to the temporary logposterior
    """
    def transDimJumpAccepted(self, lnprob, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        lp = 0.0    # No added prior value just yet
        
        # See if we call model per number or per parameter
        if psrnfinc != None and psrnfdminc != None:
            psrbfinc, psrbfdminc = self.getPsrLimFreqFromNumbers(psrnfinc, psrnfdminc)

        # Obtain the frequency selectors; setting the psr frequencies is done
        # already so that will be skipped in this function
        bfind, bfdmind, bcurfind, bcurfdmind = self.prepareLimFreqIndicators(psrbfinc, psrbfdminc)

        # Loop over all signals, and find the (DM)spectrum parameters
        for m2signal in self.ptasignals:
            if m2signal.stype == 'spectrum' and m2signal.corr == 'single':
                # Red noise, see if we need to include it
                findex = int(np.sum(self.npf[:m2signal.pulsarind])/2)
                nfreq = int(self.npf[m2signal.pulsarind]/2)

                # Select the common frequency modes
                inc = np.logical_and(bfind[findex:findex+nfreq], bcurfind[findex:findex+nfreq])
                # Select the _new_ frequency modes
                newind = np.logical_and(bfind[findex:findex+nfreq], inc == False)

                if np.sum(newind) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[newind] - m2signal.pmin[newind]))
            elif m2signal.stype == 'dmspectrum' and m2signal.corr == 'single':
                fdmindex = int(np.sum(self.npfdm[:m2signal.pulsarind])/2)
                nfreqdm = int(self.npfdm[m2signal.pulsarind]/2)

                # Select the common frequency modes
                inc = np.logical_and(bfdmind[findex:findex+nfreq], bcurfdmind[findex:findex+nfreq])
                # Select the _new_ frequency modes
                newind = np.logical_and(bfind[findex:findex+nfreq], inc == False)

                if np.sum(newind) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[newind] - m2signal.pmin[newind]))

        # Update the step position in trans-dimensional parameter space (model space)
        for psr in self.ptapsrs:
            psr.bcurfinc = psr.bfinc.copy()
            psr.bcurfdminc = psr.bfdminc.copy()

        # Return the adjusted logposterior value
        return lnprob + lp

    """
    When doing an RJMCMC, sometimes the sampler will jump to a space with more
    dimensions than we are currently in. In that case, the new parameters must
    be drawn from the prior. This function will draw valid new parameters if
    required
    """
    def afterJumpPars(self, parameters, npropmod1, npropmod2):
        if len(self.ptapsrs) > 1:
            raise ValueError("ERROR: modelNrToArray can only work on single psr")

        # The maximum model numbers
        maxmod1 = int(self.ptapsrs[0].Fmat.shape[1]/2)
        maxmod2 = int(self.ptapsrs[0].DF.shape[1]/2)

        if maxmod1 == 0 or maxmod2 == 0:
            raise ValueError("ERROR: RN or DMV dimension == 0")

        # The current model numbers
        curmod1 = np.sum(self.ptapsrs[0].bfinc)
        curmod2 = np.sum(self.ptapsrs[0].bfdminc)

        newparameters = parameters.copy()

        # Check if we need to draw new red noise parameters
        if npropmod1 > curmod1:
            for m2signal in self.ptasignals:
                if m2signal.stype == 'spectrum' and m2signal.corr == 'single':
                    indexfull = m2signal.nindex+npropmod1-1
                    index = npropmod1-1
                    newparameters[indexfull] = m2signal.pmin[index] + \
                            np.random.rand() * (m2signal.pmax[index] - m2signal.pmin[index])

        if npropmod2 > curmod2:
            for m2signal in self.ptasignals:
                if m2signal.stype == 'dmspectrum':
                    indexfull = m2signal.nindex+npropmod2-1
                    index = npropmod2-1
                    newparameters[indexfull] = m2signal.pmin[index] + \
                            np.random.rand() * (m2signal.pmax[index] - m2signal.pmin[index])

        return newparameters

    """
    When we are doing likelihood evaluations which contain single frequency
    lines with variable frequencies, then we need to re-form the Fourier design
    matrices at every likelihood step. That is done in this function.

    NOTE: does not yet work for DM spectral lines
    """
    def updateSpectralLines(self, parameters):
        # Loop over all signals, and obtain the new frequencies of the lines
        for ss in range(len(self.ptasignals)):
            m2signal = self.ptasignals[ss]
            if m2signal.stype == 'frequencyline':
                self.ptapsrs[m2signal.pulsarind].SFfreqs[2*m2signal.npsrfreqindex:2*m2signal.npsrfreqindex+2] = parameters[m2signal.nindex]

        for pindex in range(len(self.ptapsrs)):
            m2psr = self.ptapsrs[pindex]
            if m2psr.frequencyLinesAdded > 0:
                if self.likfunc == 'mark4ln':
                    m2psr.SFmat = singleFreqFourierModes(m2psr.toas, 10**m2psr.SFfreqs[::2])
                    m2psr.FFmat = np.append(m2psr.Fmat, m2psr.SFmat, axis=1)

                    m2psr.UtFF = np.dot(m2psr.U.T, m2psr.FFmat)
                else:
                    m2psr.SFmat = singleFreqFourierModes(m2psr.toas, 10**m2psr.SFfreqs[::2])
                    m2psr.FFmat = np.append(m2psr.Fmat, m2psr.SFmat, axis=1)
                    #GtSF = np.dot(m2psr.Gmat.T, m2psr.SFmat)
                    #GGtSF = np.dot(m2psr.Gmat, GtSF)
                    #m2psr.GGtFF = np.append(m2psr.GGtF, GGtSF, axis=1)
                    #m2psr.GGtFF = np.dot(m2psr.Gmat, GtFF)

                    if m2psr.twoComponentNoise:
                        #GtSF = np.dot(m2psr.Gmat.T, m2psr.SFmat)
                        AGSF = np.dot(m2psr.AG, m2psr.SFmat)
                        m2psr.AGFF = np.append(m2psr.AGF, AGSF, axis=1)

                        #GtFF = np.dot(m2psr.Gmat.T, m2psr.FFmat)
                        #GtFF = np.append(m2psr.GtF, GtSF, axis=1)

                        #m2psr.AGFF = np.dot(m2psr.Amat.T, GtFF)

            if m2psr.dmfrequencyLinesAdded > 0:
                m2psr.SFdmmat = singleFreqFourierModes(m2psr.toas, 10**m2psr.SFdmfreqs[::2])
                m2psr.DSF = np.dot(m2psr.Dmat, m2psr.SFdmmat)
                m2psr.DFF = np.append(m2psr.DF, m2psr.DSF, axis=1)

                m2psr.EEmat = np.append(m2psr.FFmat, m2psr.DF, axis=1)
                GtEE = np.dot(m2psr.Gmat.T, m2psr.EEmat)
                m2psr.GGtEE = np.dot(m2psr.Gmat, GtEE)

                if m2psr.twoComponentNoise:
                    m2psr.AGFF = np.dot(m2psr.Amat.T, GtFF)

                    if self.likfunc in ['mark6', 'mark6fa', 'mark8', 'mark10']:
                        m2psr.AGEE = np.dot(m2psr.Amat.T, GtEE)


    """
    mark1 loglikelihood of the pta model/likelihood implementation

    This likelihood is similar to mark3, but it uses the frequency coefficients
    explicitly in the likelihood. Mark3 marginalises over them analytically.
    Therefore, this mark1 version requires some extra parameters in the model,
    all part of an extra auxiliary 'signal'.
    
    TODO: (Including DM variations not yet implemented)

    DEPRECATED!!
    """
    def mark1loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        self.setPsrNoise(parameters)

        self.constructPhiAndTheta(parameters)

        # Loop over all white noise signals, and fill the pulsar Nvec, and the
        # Fourier coefficients
        for m2signal in self.ptasignals:
            if m2signal.stype == 'fouriercoeff':
                findex = np.sum(self.npf[:m2signal.pulsarind])
                nfour = self.npf[m2signal.pulsarind]
                if nfour != m2signal.npars:
                    raise ValueError('ERROR: len(nfour) not correct')
                self.avec[findex:findex+nfour] = parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]


        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)

            # Fill the auxiliaries
            nobs = len(self.ptapsrs[ii].toas)
            ng = self.ptapsrs[ii].Gmat.shape[1]
            self.rGr[ii] = np.sum(self.ptapsrs[ii].GGr ** 2 / self.ptapsrs[ii].Nvec)

            GGtFa = np.dot(self.ptapsrs[ii].GGtF, self.avec[findex:findex+2*nfreq])
            self.rGFa[ii] = np.sum(self.ptapsrs[ii].GGr * GGtFa / self.ptapsrs[ii].Nvec)
            self.aFGFa[ii] = np.sum(GGtFa**2 / self.ptapsrs[ii].Nvec)
            self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) * ng / nobs
        

        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(self.Phi)))
            aPhia = np.sum(self.avec * self.avec / np.diag(self.Phi))
        else:
            cf = sl.cho_factor(self.Phi)
            PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
            aPhia = np.dot(self.avec, sl.cho_solve(cf, self.avec))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.rGr) - 0.5*np.sum(self.aFGFa) + np.sum(self.rGFa) \
               -0.5*np.sum(self.GNGldet) - 0.5*aPhia - 0.5*PhiLD



    """
    mark2 loglikelihood of the pta model/likelihood implementation

    This likelihood can only deal with efac/equad signals
    """
    def mark2loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # MARK A

        self.setPsrNoise(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet)



    """
    mark3 loglikelihood of the pta model/likelihood implementation

    This likelihood uses an approximation for the noise matrices: in general GNG
    is not diagonal for the noise. Use that the projection of the inverse is
    roughly equal to the inverse of the projection (not true for red noise).
    Rationale is that white noise is not covariant with the timing model
    
    DM variation spectrum is not included anymore; the basis functions required
    to span the DMV space are too far removed from the Fourier modes

    Profiling execution time. Put MDC1 open challenge 1 in the file
    mdc1-open1.h5, and load with:
    ===============================================================================
    setup_mark3 = "import numpy as np, piccard as pic, matplotlib.pyplot as plt ; m3lik = pic.ptaLikelihood() ; m3lik.initFromFile('mdc1-open1.h5') ; m3lik.initModel(15, modelIndependentGWB=False, modelIndependentNoise=False, modelIndependentDM=False, modelIndependentAniGWB=False, varyEfac=False, incRedNoise=True, incEquad=False, separateEfacs=False, incGWB=True, incDM=False, incAniGWB=False, lAniGWB=2, likfunc='mark3') ; m3lik.initPrior()"
    timeit.timeit('m3lik.logposterior(m3lik.pstart)', setup=setup_mark3, number=1000)
    ===============================================================================

    Setup:      1           2
    ---------------------------------------
    Mark A:  10.4 sec     0.04 sec
    Mark B:  10.4 sec     0.28 sec
    Mark C:  60.4 sec    48.26 sec
    Mark D:  81.1 sec    52.90 sec
    Mark E: 490   sec   420    sec
    Mark F: 605   sec   540    sec
    """
    def mark3loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # MARK A


        self.setPsrNoise(parameters)

        # MARK B

        self.constructPhiAndTheta(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                NGGF = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGF.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].AGr, NGGF)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].AGF.T, NGGF)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Fmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].Fmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].residuals, NiF) \
                        - np.dot(GcNir, GcNiGcF)
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = \
                        np.dot(NiF.T, self.ptapsrs[ii].Fmat) - np.dot(GcNiF.T, GcNiGcF)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(self.Phi)))
            Phiinv = np.diag(1.0 / np.diag(self.Phi))
        else:
            try:
                cf = sl.cho_factor(self.Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(self.Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"

        # MARK E

        # Construct and decompose Sigma
        self.Sigma = self.FGGNGGF + Phiinv
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGF, sl.cho_solve(cf, self.rGF))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGF, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF))))
        # Mark F

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD


    """
    mark3fa loglikelihood of the pta model/likelihood implementation

    First-order approximation of mark3loglikelihood

    """
    def mark3faloglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # MARK A


        self.setPsrNoise(parameters)

        # MARK B

        self.constructPhiAndTheta(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                NGGF = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGF.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].AGr, NGGF)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].AGF.T, NGGF)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Fmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].Fmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].residuals, NiF) \
                        - np.dot(GcNir, GcNiGcF)
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = \
                        np.dot(NiF.T, self.ptapsrs[ii].Fmat) - np.dot(GcNiF.T, GcNiGcF)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert the auto-correlation part of phi.
        PhiaLD = np.sum(np.log(np.diag(self.Phi)))
        Phiainv = np.diag(1.0 / np.diag(self.Phi))
        SigmaLD = 0
        rGSigmaGr = 0
        rGFSigma = self.rGF.copy()

        # Construct the auto-part of sigma
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)

            self.Sigma[findex:findex+2*nfreq, findex:findex+2*nfreq] = \
                    self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] + \
                    np.diag(np.diag(Phiainv)[findex:findex+2*nfreq])

            # Invert this on a per-pulsar level, too
            try:
                cf = sl.cho_factor(self.Sigma[findex:findex+2*nfreq, findex:findex+2*nfreq])
                SigmaLD += 2*np.sum(np.log(np.diag(cf[0])))
                rGFSigma[findex:findex+2*nfreq] = sl.cho_solve(cf, self.rGF[findex:findex+2*nfreq])

                rGSigmaGr += np.dot(self.rGF[findex:findex+2*nfreq], \
                        rGFSigma[findex:findex+2*nfreq])
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Sigma[findex:findex+2*nfreq, findex:findex+2*nfreq])
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                SigmaLD += np.sum(np.log(s))

                rGFSigma[findex:findex+2*nfreq] = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF[findex:findex+2*nfreq])))

                rGSigmaGr += np.dot(self.rGF[findex:findex+2*nfreq], \
                        rGFSigma[findex:findex+2*nfreq])

        # Calculate the cross-correlation approximation part
        rScr = 0
        for ii in range(npsrs):
            findexi = np.sum(self.npf[:ii])
            nfreqi = int(self.npf[ii]/2)
            for jj in range(ii+1, npsrs):
                findexj = np.sum(self.npf[:jj])
                nfreqj = int(self.npf[jj]/2)
                rScr += np.dot(rGFSigma[findexi:findexi+2*nfreqi], \
                        np.dot(self.Phi[findexi:findexi+2*nfreqi, findexj:findexj+2*nfreqj], \
                        rGFSigma[findexj:findexj+2*nfreqj]))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                +0.5*rGSigmaGr - 0.5*PhiaLD - 0.5*SigmaLD + rScr




    """
    mark4 loglikelihood of the pta model/likelihood implementation

    implements coarse-graining, without added frequency lines

    """
    def mark4loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # MARK A

        self.setPsrNoise(parameters)

        # MARK B

        self.constructPhiAndTheta(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)
            uindex = np.sum(self.npu[:ii])
            nus = self.npu[ii]

            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGU))
                NGGU = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGU.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGU[uindex:uindex+nus] = np.dot(self.ptapsrs[ii].AGr, NGGU)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.UGGNGGU[uindex:uindex+nus, uindex:uindex+nus] = np.dot(self.ptapsrs[ii].AGU.T, NGGU)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiU = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].U.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiU = np.dot(NiGc.T, self.ptapsrs[ii].Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGU[uindex:uindex+nus] = np.dot(self.ptapsrs[ii].residuals, NiU) \
                        - np.dot(GcNir, GcNiGcU)
                self.UGGNGGU[uindex:uindex+nus, uindex:uindex+nus] = \
                        np.dot(NiU.T, self.ptapsrs[ii].U) - np.dot(GcNiU.T, GcNiGcU)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            # Quick and dirty:
            #PhiU = ( * self.ptapsrs[ii].AGU.T).T

            UPhiU = np.dot(self.ptapsrs[0].UtF, np.dot(self.Phi, self.ptapsrs[0].UtF.T))
            Phi = UPhiU + self.ptapsrs[0].Qamp * np.eye(len(self.ptapsrs[0].avetoas))

            #PhiLD = np.sum(np.log(np.diag(Phi)))
            #Phiinv = np.diag(1.0 / np.diag(Phi))
            try:
                cf = sl.cho_factor(Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                #print "Fallback to SVD for Phi"

        """
        else:
            try:
                cf = sl.cho_factor(Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"
        """

        # MARK E

        # Construct and decompose Sigma
        self.Sigma = self.UGGNGGU + Phiinv
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGU, sl.cho_solve(cf, self.rGU))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGU, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGU))))
        # Mark F

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD


    """
    mark4 loglikelihood of the pta model/likelihood implementation

    implements coarse-graining, including frequency lines

    """
    def mark4lnloglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # First re-construct the frequency matrices here...
        self.updateSpectralLines(parameters)

        # MARK A

        self.setPsrNoise(parameters)

        # MARK B

        self.constructPhiAndTheta(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            uindex = np.sum(self.npu[:ii])
            nus = self.npu[ii]

            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGU))
                NGGU = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGU.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGU[uindex:uindex+nus] = np.dot(self.ptapsrs[ii].AGr, NGGU)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.UGGNGGU[uindex:uindex+nus, uindex:uindex+nus] = np.dot(self.ptapsrs[ii].AGU.T, NGGU)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiU = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].U.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiU = np.dot(NiGc.T, self.ptapsrs[ii].Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGU[uindex:uindex+nus] = np.dot(self.ptapsrs[ii].residuals, NiU) \
                        - np.dot(GcNir, GcNiGcU)
                self.UGGNGGU[uindex:uindex+nus, uindex:uindex+nus] = \
                        np.dot(NiU.T, self.ptapsrs[ii].U) - np.dot(GcNiU.T, GcNiGcU)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            # Quick and dirty:
            #PhiU = ( * self.ptapsrs[ii].AGU.T).T

            UPhiU = np.dot(self.ptapsrs[0].UtFF, np.dot(self.Phi, self.ptapsrs[0].UtFF.T))
            Phi = UPhiU + self.ptapsrs[0].Qamp * np.eye(len(self.ptapsrs[0].avetoas))

            #PhiLD = np.sum(np.log(np.diag(Phi)))
            #Phiinv = np.diag(1.0 / np.diag(Phi))
            try:
                cf = sl.cho_factor(Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                #print "Fallback to SVD for Phi"
        """
        else:
            try:
                cf = sl.cho_factor(Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"
        """

        # MARK E

        # Construct and decompose Sigma
        self.Sigma = self.UGGNGGU + Phiinv
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGU, sl.cho_solve(cf, self.rGU))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGU, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGU))))
        # Mark F

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD



    """
    mark6 loglikelihood of the pta model/likelihood implementation

    This likelihood uses an approximation for the noise matrices: in general GNG
    is not diagonal for the noise. Use that the projection of the inverse is
    roughly equal to the inverse of the projection (not true for red noise).
    Rationale is that white noise is not covariant with the timing model
    
    DM variation spectrum is included 
    """
    def mark6loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # The red signals
        self.constructPhiAndTheta(parameters)

        # The white noise
        self.setPsrNoise(parameters)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            fdmindex = np.sum(self.npfdm[:ii])
            nfreq = int(self.npf[ii]/2)
            nfreqdm = int(self.npfdm[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                NGGE = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGE.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].AGr, NGGE)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].AGE.T, NGGE)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Emat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].Emat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].residuals, NiE) \
                        - np.dot(GcNir, GcNiGcE)
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = \
                        np.dot(NiE.T, self.ptapsrs[ii].Emat) - np.dot(GcNiE.T, GcNiGcE)

        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(self.Phi)))
            Phiinv = np.diag(1.0 / np.diag(self.Phi))
        else:
            try:
                cf = sl.cho_factor(self.Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(self.Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"

        ThetaLD = np.sum(np.log(self.Thetavec))

        # Construct Sigma (needs to be written out because of the ordering of
        # the phi/theta matrices)
        self.Sigma = self.EGGNGGE
        for ii in range(npsrs):
            inda = np.sum(self.npf[:ii]) + np.sum(self.npfdm[:ii])
            indaph = np.sum(self.npf[:ii])
            indb = np.sum(self.npf[:ii+1]) + np.sum(self.npfdm[:ii])
            indbph = np.sum(self.npf[:ii+1])
            indc = np.sum(self.npf[:ii+1]) + np.sum(self.npfdm[:ii+1])
            didm = np.diag_indices(self.npfdm[ii])

            # Red noise / correlated signals
            self.Sigma[inda:indb, inda:indb] += \
                    Phiinv[indaph:indbph, indaph:indbph]

            # DM variations
            self.Sigma[indb:indc, indb:indc] += \
                    np.diag(1.0 / self.Thetavec[np.sum(self.npfdm[:ii]):np.sum(self.npfdm[:ii+1])])

            # Include the cross terms of Phi in Sigma.
            for jj in range(ii, npsrs):
                inda2 = np.sum(self.npf[:jj])+np.sum(self.npfdm[:jj])
                indaph2 = np.sum(self.npf[:jj])
                indb2 = np.sum(self.npf[:jj+1])+np.sum(self.npfdm[:jj])
                indbph2 = np.sum(self.npf[:jj+1])

                # Correlated signals (no DM variations
                self.Sigma[inda:indb, inda2:indb2] += \
                        Phiinv[indaph:indbph, indaph2:indbph2]
                self.Sigma[inda2:indb2, inda:indb] += \
                        Phiinv[indaph2:indbph2, indaph:indbph]

        
        """
        di = np.diag_indices(np.sum(self.npf))
        didm = np.diag_indices(np.sum(self.npfdm))
        self.Sigma = self.EGGNGGE
        self.Sigma[0:np.sum(self.npf), 0:np.sum(self.npf)] += Phiinv
        self.Sigma[np.sum(self.npf):, np.sum(self.npf):][didm] += 1.0 / self.Thetavec
        """
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGE, sl.cho_solve(cf, self.rGE))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGE, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGE))))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD - 0.5*ThetaLD


    """
    mark6fa loglikelihood of the pta model/likelihood implementation

    Like mark6loglikelihood, but now with the first-order approximation for the
    correlations

    """
    def mark6faloglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # The red signals
        self.constructPhiAndTheta(parameters)

        # The white noise
        self.setPsrNoise(parameters)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            fdmindex = np.sum(self.npfdm[:ii])
            nfreq = int(self.npf[ii]/2)
            nfreqdm = int(self.npfdm[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                NGGE = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGE.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].AGr, NGGE)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].AGE.T, NGGE)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Emat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].Emat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].residuals, NiE) \
                        - np.dot(GcNir, GcNiGcE)
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = \
                        np.dot(NiE.T, self.ptapsrs[ii].Emat) - np.dot(GcNiE.T, GcNiGcE)


        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert the auto-correlation part of phi.
        PhiaLD = np.sum(np.log(np.diag(self.Phi)))
        SigmaLD = 0
        rGSigmaGr = 0
        rGESigma = self.rGE.copy()

        ThetaLD = np.sum(np.log(self.Thetavec))

        # Construct the auto-part of sigma
        for ii in range(npsrs):
            inda = np.sum(self.npf[:ii]) + np.sum(self.npfdm[:ii])
            indaph = np.sum(self.npf[:ii])
            indb = np.sum(self.npf[:ii+1]) + np.sum(self.npfdm[:ii])
            indbph = np.sum(self.npf[:ii+1])
            indc = np.sum(self.npf[:ii+1]) + np.sum(self.npfdm[:ii+1])
            didm = np.diag_indices(self.npfdm[ii])

            self.Sigma[inda:indb, inda:indb] = self.EGGNGGE[inda:indb, inda:indb] + \
                    np.diag(1.0 / np.diag(self.Phi)[indaph:indbph])

            # DM variations
            self.Sigma[indb:indc, indb:indc] += \
                    np.diag(1.0 / self.Thetavec[np.sum(self.npfdm[:ii]):np.sum(self.npfdm[:ii+1])])

            # Invert this on a per-pulsar level, too
            try:
                cf = sl.cho_factor(self.Sigma[inda:indc, inda:indc])
                SigmaLD += 2*np.sum(np.log(np.diag(cf[0])))
                rGESigma[inda:indc] = sl.cho_solve(cf, self.rGE[inda:indc])

                rGSigmaGr += np.dot(self.rGE[inda:indc], \
                        rGESigma[inda:indc])
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Sigma[inda:indc, inda:indc])
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                SigmaLD += np.sum(np.log(s))

                rGESigma[inda:indc] = np.dot(Vh.T, np.dot(np.diag(1.0/s), \
                        np.dot(U.T, self.rGE[inda:indc])))

                rGSigmaGr += np.dot(self.rGE[inda:indc], \
                        rGESigma[inda:indc])

        # Calculate the cross-correlation approximation part
        rScr = 0
        for ii in range(npsrs):
            inda = np.sum(self.npf[:ii]) + np.sum(self.npfdm[:ii])
            indaph = np.sum(self.npf[:ii])
            indb = np.sum(self.npf[:ii+1]) + np.sum(self.npfdm[:ii])
            indbph = np.sum(self.npf[:ii+1])
            indc = np.sum(self.npf[:ii+1]) + np.sum(self.npfdm[:ii+1])

            for jj in range(ii+1, npsrs):
                inda2 = np.sum(self.npf[:jj])+np.sum(self.npfdm[:jj])
                indaph2 = np.sum(self.npf[:jj])
                indb2 = np.sum(self.npf[:jj+1])+np.sum(self.npfdm[:jj])
                indbph2 = np.sum(self.npf[:jj+1])

                rScr += np.dot(rGESigma[inda:indb], \
                        np.dot(self.Phi[indaph:indbph, indaph2:indbph2], \
                        rGESigma[inda2:indb2]))

        print "Results: ", np.sum(self.rGr), np.sum(self.GNGldet), \
                rGSigmaGr, PhiaLD, SigmaLD, ThetaLD, rScr

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                +0.5*rGSigmaGr - 0.5*PhiaLD - 0.5*SigmaLD - 0.5*ThetaLD \
                +rScr



    """
    mark7 loglikelihood of the pta model/likelihood implementation

    This likelihood is the same as mark3loglikelihood, except that it allows for
    a variable number of Fourier modes to be included for the red noise. The
    parameters are there for DM as well, to allow the RJMCMC methods to call it
    in exactly the same way. However, these parameters are ignored

    psrbfinc, psrbfdminc: a boolean array, indicating which frequencies to
                          include.
    psrnfinc, psrnfdminc: integer array, indicating how many frequencies per
                          pulsar to include. Overrides psrbfinc and psrbfdminc
    """
    def mark7loglikelihood(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        npsrs = len(self.ptapsrs)

        # The red signals
        self.constructPhiAndTheta(parameters)

        # The white noise
        self.setPsrNoise(parameters)

        # If the included frequencies are passed by numbers -- not indicator
        # functions --, then obtain the indicators from the numbers
        if psrnfinc != None and psrnfdminc != None:
            psrbfinc, psrbfdminc = self.getPsrLimFreqFromNumbers(psrnfinc, psrnfdminc)

        # Obtain the frequency selectors, and set the psr frequencies
        bfind, bfdmind, bcurfind, bcurfdmind = self.prepareLimFreqIndicators(psrbfinc, psrbfdminc)

        # Double up the frequency indicators to get the mode indicators
        bfmind = np.array([bfind, bfind]).T.flatten()
        bfmdmind = np.array([bfdmind, bfdmind]).T.flatten()

        # Select the limited range Phi and Theta
        #lPhi = self.Phi[bfmind, bfmind]
        lPhi = self.Phi[:, bfmind][bfmind, :]
        #lThetavec = self.Thetavec[bfmdmind]
        lenphi = np.sum(bfmind)
        #lentheta = np.sum(bfmdmind)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.lnpf[:ii])
            nfreq = int(self.lnpf[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                NGGF = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].lAGF.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].AGr, NGGF)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].lAGF.T, NGGF)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].lFmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].lFmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].residuals, NiF) \
                        - np.dot(GcNir, GcNiGcF)
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = \
                        np.dot(NiF.T, self.ptapsrs[ii].lFmat) - np.dot(GcNiF.T, GcNiGcF)


        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(lPhi)))
            Phiinv = np.diag(1.0 / np.diag(lPhi))
        else:
            try:
                cf = sl.cho_factor(lPhi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(lPhi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(lPhi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"

        #ThetaLD = np.sum(np.log(lThetavec))

        # Construct and decompose Sigma
        #didm = np.diag_indices(np.sum(self.lnpfdm))
        self.Sigma = self.FGGNGGF[:lenphi, :lenphi]
        self.Sigma[0:np.sum(self.lnpf), 0:np.sum(self.lnpf)] += Phiinv
        #Sigma[np.sum(self.lnpf):, np.sum(self.lnpf):][didm] += 1.0 / lThetavec
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGF[:lenphi], sl.cho_solve(cf, self.rGF[:lenphi]))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGF[:lenphi], np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF[:lenphi]))))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                +0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD



    """
    mark8 loglikelihood of the pta model/likelihood implementation

    This likelihood is the same as mark6loglikelihood, except that it allows for
    a variable number of Fourier modes to be included, both for DM and for red
    noise

    psrbfinc, psrbfdminc: a boolean array, indicating which frequencies to
                          include.
    psrnfinc, psrnfdminc: integer array, indicating how many frequencies per
                          pulsar to include. Overrides psrbfinc and psrbfdminc

    Profiling execution time. Put J0437 of the ipta-2013 set in the file
    J0437.h5, and load with:
    =============================================================================
    setup_mark8 = "import numpy as np, piccard as pic, matplotlib.pyplot as plt ; m3lik = pic.ptaLikelihood() ; m3lik.initFromFile('J0437.h5') ; m3lik.initModel(30, modelIndependentGWB=False, modelIndependentNoise=True, modelIndependentDM=True, modelIndependentAniGWB=False, varyEfac=True, incRedNoise=True, incEquad=True, separateEfacs=True, incGWB=False, incDM=True, incAniGWB=False, lAniGWB=2, likfunc='mark6') ; m3lik.initPrior()"
    =============================================================================

    Call with:
    setup1: timeit.timeit('m3lik.mark8logposterior(m3lik.pstart, psrnfinc=[4], psrnfdminc=[4])', setup=setup_mark8, number=100)
    setup2: timeit.timeit('m3lik.mark8logposterior(m3lik.pstart, psrnfinc=[np.random.randint(1,6)], psrnfdminc=[np.random.randint(1,6)])', setup=setup_mark8, number=100)

    Setup:      1           2
    ---------------------------------------
    Mark A:   0.04  sec    0.50 sec
    Mark B:   0.074 sec    0.50 sec
    Mark C:   0.08 sec     0.51 sec
    Mark D:   0.09 sec     0.54 sec
    Mark E:   0.10 sec     0.56 sec
    Mark F:   0.41 sec     0.82 sec
    Mark G:   0.83 sec     0.91 sec
    Mark H:   0.76 sec     0.84 sec
    """
    def mark8loglikelihood(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        npsrs = len(self.ptapsrs)

        # MARK A

        # The white noise
        self.setPsrNoise(parameters)

        # MARK B

        # The red signals
        self.constructPhiAndTheta(parameters)

        # MARK C

        # If the included frequencies are passed by numbers -- not indicator
        # functions --, then obtain the indicators from the numbers
        if psrnfinc != None and psrnfdminc != None:
            psrbfinc, psrbfdminc = self.getPsrLimFreqFromNumbers(psrnfinc, psrnfdminc)

        # Obtain the frequency selectors, and set the psr frequencies
        bfind, bfdmind, bcurfind, bcurfdmind = self.prepareLimFreqIndicators(psrbfinc, psrbfdminc)

        # MARK D

        # Double up the frequency indicators to get the mode indicators
        bfmind = np.array([bfind, bfind]).T.flatten()
        bfmdmind = np.array([bfdmind, bfdmind]).T.flatten()

        # Select the limited range Phi and Theta
        #lPhi = self.Phi[bfmind, bfmind]

        lPhi = self.Phi[numpy.ix_(bfmind, bfmind)]
        # lPhi = self.Phi[:, bfmind][bfmind, :]
        lThetavec = self.Thetavec[bfmdmind]
        lenphi = np.sum(bfmind)
        lentheta = np.sum(bfmdmind)

        # MARK E

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.lnpf[:ii])
            fdmindex = np.sum(self.lnpfdm[:ii])
            nfreq = int(self.lnpf[ii]/2)
            nfreqdm = int(self.lnpfdm[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                NGGE = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].lAGE.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].AGr, NGGE)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].lAGE.T, NGGE)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].lEmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].lEmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].residuals, NiE) \
                        - np.dot(GcNir, GcNiGcE)
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = \
                        np.dot(NiE.T, self.ptapsrs[ii].lEmat) - np.dot(GcNiE.T, GcNiGcE)

        
        # MARK F

        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(lPhi)))
            Phiinv = np.diag(1.0 / np.diag(lPhi))
        else:
            try:
                cf = sl.cho_factor(lPhi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(lPhi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(lPhi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"

        # MARK G

        ThetaLD = np.sum(np.log(lThetavec))

        # Construct and decompose Sigma
        didm = np.diag_indices(np.sum(self.lnpfdm))
        self.Sigma = self.EGGNGGE[:lenphi+lentheta, :lenphi+lentheta]
        self.Sigma[0:np.sum(self.lnpf), 0:np.sum(self.lnpf)] += Phiinv
        self.Sigma[np.sum(self.lnpf):, np.sum(self.lnpf):][didm] += 1.0 / lThetavec
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGE[:lenphi+lentheta], sl.cho_solve(cf, self.rGE[:lenphi+lentheta]))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGE[:lenphi+lentheta], np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGE[:lenphi+lentheta]))))

        # MARK H

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                +0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD - 0.5*ThetaLD




    """
    mark9 loglikelihood of the pta model/likelihood implementation

    like mark3loglikelihood, but with single frequency lines
    """
    def mark9loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # First re-construct the frequency matrices here...
        self.updateSpectralLines(parameters)

        # MARK A

        self.setPsrNoise(parameters)

        # MARK B

        self.constructPhiAndTheta(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npff[:ii])
            nfreq = int(self.npff[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                NGGF = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGFF.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].AGr, NGGF)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].AGFF.T, NGGF)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].FFmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].FFmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].residuals, NiF) \
                        - np.dot(GcNir, GcNiGcF)
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = \
                        np.dot(NiF.T, self.ptapsrs[ii].FFmat) - np.dot(GcNiF.T, GcNiGcF)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(self.Phi)))
            Phiinv = np.diag(1.0 / np.diag(self.Phi))
        else:
            try:
                cf = sl.cho_factor(self.Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(self.Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"

        # MARK E

        # Construct and decompose Sigma
        self.Sigma = self.FGGNGGF + Phiinv
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGF, sl.cho_solve(cf, self.rGF))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGF, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF))))
        # Mark F

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                +0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD


    """
    mark10 loglikelihood of the pta model/likelihood implementation

    Just like mark6loglikelihood, but now with single DM frequencies included in
    the model
    """
    def mark10loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # The red signals
        self.constructPhiAndTheta(parameters)

        # The white noise
        self.setPsrNoise(parameters)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npff[:ii])
            fdmindex = np.sum(self.npffdm[:ii])
            nfreq = int(self.npff[ii]/2)
            nfreqdm = int(self.npffdm[ii]/2)

            if self.ptapsrs[ii].twoComponentNoise:
                NGGE = ((1.0/self.ptapsrs[ii].Nwvec) * self.ptapsrs[ii].AGE.T).T

                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].AGr, NGGE)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].AGE.T, NGGE)
            else:
                Nir = self.ptapsrs[ii].residuals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Gcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Gcmat.T, NiGc)
                NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Emat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].residuals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].Emat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].residuals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].residuals, NiE) \
                        - np.dot(GcNir, GcNiGcE)
                self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = \
                        np.dot(NiE.T, self.ptapsrs[ii].Emat) - np.dot(GcNiE.T, GcNiGcE)

        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(self.Phi)))
            Phiinv = np.diag(1.0 / np.diag(self.Phi))
        else:
            try:
                cf = sl.cho_factor(self.Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(self.Phi.shape[0]))
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(self.Phi)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Phi singular according to SVD")
                PhiLD = np.sum(np.log(s))
                Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))

                print "Fallback to SVD for Phi"

        ThetaLD = np.sum(np.log(self.Thetavec))

        # Construct and decompose Sigma
        di = np.diag_indices(np.sum(self.npff))
        didm = np.diag_indices(np.sum(self.npffdm))
        self.Sigma = self.EGGNGGE
        self.Sigma[0:np.sum(self.npff), 0:np.sum(self.npff)] += Phiinv
        self.Sigma[np.sum(self.npff):, np.sum(self.npff):][didm] += 1.0 / self.Thetavec
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGE, sl.cho_solve(cf, self.rGE))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGE, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGE))))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                +0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD - 0.5*ThetaLD








    def loglikelihood(self, parameters):
        ll = 0.0

        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            if self.likfunc == 'mark1':
                ll = self.mark1loglikelihood(parameters)
            elif self.likfunc == 'mark2':
                ll = self.mark2loglikelihood(parameters)
            elif self.likfunc == 'mark3':
                ll = self.mark3loglikelihood(parameters)
            elif self.likfunc == 'mark3fa':
                ll = self.mark3faloglikelihood(parameters)
            elif self.likfunc == 'mark4':
                ll = self.mark4loglikelihood(parameters)
            elif self.likfunc == 'mark4ln':
                ll = self.mark4lnloglikelihood(parameters)
            elif self.likfunc == 'mark6':
                ll = self.mark6loglikelihood(parameters)
            elif self.likfunc == 'mark6fa':
                ll = self.mark6faloglikelihood(parameters)
            elif self.likfunc == 'mark7':
                ll = self.mark7loglikelihood(parameters)
            elif self.likfunc == 'mark8':
                ll = self.mark8loglikelihood(parameters)
            elif self.likfunc == 'mark9':
                ll = self.mark9loglikelihood(parameters)
        else:
            ll = -1e99

        return ll

    # TODO: the prior for the amplitude parameters is not yet normalised
    def mark4logprior(self, parameters):
        lp = 0.0

        # Loop over all signals
        for m2signal in self.ptasignals:
            if m2signal.stype == 'powerlaw' and m2signal.corr == 'anisotropicgwb':
                nclm = m2signal.aniCorr.clmlength()
                # lp += parameters[m2signal.nindex]

                sparameters = m2signal.pstart.copy()
                nvaryclm = np.sum(m2signal.bvary[3:])
                nskip = np.sum(m2signal.bvary[:3])
                sparameters[3:][m2signal.bvary[3:]] = \
                        parameters[m2signal.nindex+nskip:m2signal.nindex+nskip+nvaryclm]

                clm = sparameters[m2signal.ntotpars-nclm:m2signal.ntotpars]
                if m2signal.aniCorr.priorIndicator(clm) == False:
                    lp -= 1e99
            elif m2signal.stype == 'powerlaw' and m2signal.corr != 'single':
                lp += parameters[m2signal.nindex]
            elif m2signal.stype == 'spectrum' and m2signal.corr == 'anisotropicgwb':
                nclm = m2signal.aniCorr.clmlength()
                sparameters = m2signal.pstart.copy()
                nfreqs = m2signal.ntotpars - nclm
                nvaryclm = np.sum(m2signal.bvary[nfreqs:])
                nskip = np.sum(m2signal.bvary[:nfreqs])
                sparameters[nfreqs:][m2signal.bvary[nfreqs:]] = \
                        parameters[m2signal.nindex+nskip:m2signal.nindex+nskip+nvaryclm]

                clm = sparameters[m2signal.nindex+m2signal.ntotpars-nclm:m2signal.nindex+m2signal.ntotpars]

                if m2signal.aniCorr.priorIndicator(clm) == False:
                    lp -= 1e99
            elif m2signal.stype == 'spectrum' and m2signal.corr != 'single':
                lp += np.sum(parameters[m2signal.nindex:m2signal.nindex+m2signal.npars])

            # Divide by the prior range
            if np.sum(m2signal.bvary) > 0:
                lp -= np.sum(np.log(m2signal.pmax[m2signal.bvary]-m2signal.pmin[m2signal.bvary]))
        return lp

    # Note: the inclusion of a uniform-amplitude part can have a big influence
    def mark7logprior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        lp = 0.0

        # MARK A1

        if psrnfinc != None and psrnfdminc != None:
            psrbfinc, psrbfdminc = self.getPsrLimFreqFromNumbers(psrnfinc, psrnfdminc)

        # MARK A2

        # Obtain the frequency selectors, and set the psr frequencies
        bfind, bfdmind, bcurfind, bcurfdmind = self.prepareLimFreqIndicators(psrbfinc, psrbfdminc)

        # MARK A3

        # Loop over all signals
        for m2signal in self.ptasignals:
            if m2signal.stype == 'spectrum' and m2signal.corr == 'single':
                # Red noise, see if we need to include it
                findex = int(np.sum(self.npf[:m2signal.pulsarind])/2)
                nfreq = int(self.npf[m2signal.pulsarind]/2)
                inc = np.logical_and(bfind[findex:findex+nfreq], bcurfind[findex:findex+nfreq])

                if np.sum(inc) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[inc] - m2signal.pmin[inc]))
                    #lp -= np.sum(inc) * 1.0
            elif m2signal.stype == 'dmspectrum' and m2signal.corr == 'single':
                fdmindex = int(np.sum(self.npfdm[:m2signal.pulsarind])/2)
                nfreqdm = int(self.npfdm[m2signal.pulsarind]/2)
                inc = np.logical_and(bfdmind[findex:findex+nfreq], bcurfdmind[findex:findex+nfreq])

                if np.sum(inc) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[inc] - m2signal.pmin[inc]))
                    #lp -= np.sum(inc) * 1.0
            else:
                if np.sum(m2signal.bvary) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[m2signal.bvary]-m2signal.pmin[m2signal.bvary]))

        return lp

    def mark8logprior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        return self.mark7logprior(parameters, psrbfinc, psrbfdminc, \
                psrnfinc, psrnfdminc)

    def mark9logprior(self, parameters):
        lp = self.mark4logprior(parameters)

        # Check if we have frequency ordering
        if lp > -1e98:
            if self.orderFrequencyLines:
                # Loop over all signals, and obtain the new frequencies of the lines
                for ss in range(len(self.ptasignals)):
                    m2signal = self.ptasignals[ss]
                    if m2signal.stype == 'frequencyline':
                        self.ptapsrs[m2signal.pulsarind].SFfreqs[2*m2signal.npsrfreqindex:2*m2signal.npsrfreqindex+2] = parameters[m2signal.nindex]

                for m2psr in self.ptapsrs:
                    if m2psr.frequencyLinesAdded > 0:
                        if all(m2psr.SFfreqs[::2][i] <= m2psr.SFfreqs[::2][i+1] for i in xrange(len(m2psr.SFfreqs[::2])-1)):
                            lp += np.log(math.factorial(m2psr.frequencyLinesAdded))
                        else:
                            lp = -1.0e99

        return lp

    def mark7logposterior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        lp = -1.0e99
        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            lp = self.mark7logprior(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        if lp > -1e98:
            lp += self.mark7loglikelihood(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        return lp

    def mark8logposterior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        lp = -1.0e99
        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            lp = self.mark8logprior(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        if lp > -1e98:
            lp += self.mark8loglikelihood(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        return lp

    def logprior(self, parameters):
        lp = 0.0

        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            if self.likfunc == 'mark1':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark2':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark3':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark3fa':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark4':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark4ln':
                lp = self.mark9logprior(parameters)
            elif self.likfunc == 'mark6':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark6fa':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark7':  # Mark7 should be called differently of course
                lp = self.mark7logprior(parameters)
            elif self.likfunc == 'mark8':  # Mark8 ''
                lp = self.mark8logprior(parameters)
            elif self.likfunc == 'mark9':  # Mark9 ''
                lp = self.mark9logprior(parameters)
        else:
            lp = -1e99

        return lp

    def logposterior(self, parameters):
        lp = self.logprior(parameters)
        if lp > -1e98:
            lp += self.loglikelihood(parameters)
        return lp

    def nlogposterior(self, parameters):
        return -self.logposterior(parameters)

    def logposteriorhc(self, cube, ndim, nparams):
        acube = np.zeros(ndim)

        for ii in range(ndim):
            acube[ii] = cube[ii]

        return self.logposterior(acube)

    def samplefromprior(self, cube, ndim, nparams):
        for ii in range(ndim):
            cube[ii] = self.pmin[ii] + cube[ii] * (self.pmax[ii] - self.pmin[ii])


    """
    Simple signal generation, use frequency domain for power-law signals by
    default
    """
    def gensig(self, parameters=None, filename=None, timedomain=False):
        if parameters == None:
            parameters = self.pstart.copy()

        npsrs = len(self.ptapsrs)

        self.setPsrNoise(parameters)

        self.constructPhiAndTheta(parameters)

        # The time-domain matrices for red noise and DM variations
        Cr = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals
        Cdm = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals

        # Allocate some auxiliary matrices
        Cov = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
        totFmat = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totDFmat = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totDmat = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
        totG = np.zeros((np.sum(self.npobs), np.sum(self.npgs)))
        tottoas = np.zeros(np.sum(self.npobs))
        tottoaerrs = np.zeros(np.sum(self.npobs))

        # Fill the auxiliary matrices
        for ii in range(npsrs):
            nindex = np.sum(self.npobs[:ii])
            findex = np.sum(self.npf[:ii])
            gindex = np.sum(self.npgs[:ii])
            npobs = self.npobs[ii]
            nppf = self.npf[ii]
            npgs = self.npgs[ii]
            #if self.ptapsrs[ii].twoComponentNoise:
            #    pass
            #else:
            #    pass
            Cov[nindex:nindex+npobs, nindex:nindex+npobs] = np.diag(self.ptapsrs[ii].Nvec)
            totFmat[nindex:nindex+npobs, findex:findex+nppf] = self.ptapsrs[ii].Fmat
            totDFmat[nindex:nindex+npobs, findex:findex+nppf] = self.ptapsrs[ii].DF
            totDmat[nindex:nindex+npobs, nindex:nindex+npobs] = self.ptapsrs[ii].Dmat

            totG[nindex:nindex+npobs, gindex:gindex+npgs] = self.ptapsrs[ii].Gmat
            tottoas[nindex:nindex+npobs] = self.ptapsrs[ii].toas
            tottoaerrs[nindex:nindex+npobs] = self.ptapsrs[ii].toaerrs


        if timedomain:
            # Do time-domain stuff explicitly here, for now
            for m2signal in self.ptasignals:
                sparameters = m2signal.pstart.copy()
                sparameters[m2signal.bvary] = \
                        parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]

                if m2signal.stype == 'powerlaw' and m2signal.corr == 'single':
                    spd = 24 * 3600.0
                    spy = 365.25 * spd
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    nindex = np.sum(self.npobs[:m2signal.pulsarind])
                    ncurobs = self.npobs[m2signal.pulsarind]

                    Cr[nindex:nindex+ncurobs,nindex:nindex+ncurobs] +=\
                            Cred_sec(self.ptapsrs[m2signal.pulsarind].toas,\
                            alpha=0.5*(3-Si),\
                            fL=1.0/100) * (Amp**2)
                elif m2signal.stype == 'dmpowerlaw' and m2signal.corr == 'single':
                    spd = 24 * 3600.0
                    spy = 365.25 * spd
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    nindex = np.sum(self.npobs[:m2signal.pulsarind])
                    ncurobs = self.npobs[m2signal.pulsarind]

                    Cdm[nindex:nindex+ncurobs,nindex:nindex+ncurobs] +=\
                            Cred_sec(self.ptapsrs[m2signal.pulsarind].toas,\
                            alpha=0.5*(3-Si),\
                            fL=1.0/100) * (Amp**2)


            Cov += Cr
            Cov += np.dot(totDmat, np.dot(Cdm, totDmat))
        else:
            # Construct them from Phi/Theta
            Cov += np.dot(totFmat, np.dot(self.Phi, totFmat.T))
            Cov += np.dot(totDFmat, np.dot(np.diag(self.Thetavec), totDFmat.T))

        # Create the projected covariance matrix, and decompose it
        GCG = np.dot(totG.T, np.dot(Cov, totG))
        cf = sl.cholesky(GCG).T

        # Generate the data in the Cholesky-basis
        xi = np.random.randn(GCG.shape[0])
        ygen = np.dot(totG, np.dot(cf, xi))

        # Display the data
        plt.errorbar(tottoas, ygen, yerr=tottoaerrs, fmt='.', c='blue')
        plt.grid(True)
        plt.show()

        # Save the data
        tindex = 0
        for ii in range(len(self.ptapsrs)):
            nobs = len(self.ptapsrs[ii].residuals)
            self.ptapsrs[ii].residuals = ygen[tindex:tindex+nobs]
            tindex += nobs

        if filename != None:
            h5file = h5.File(filename, 'a')

            if not "Data" in h5file:
                h5file.close()
                h5file = None
                raise IOError, "no Data group in hdf5 file"

            datagroup = h5file["Data"]

            # Retrieve the pulsars group
            if not "Pulsars" in datagroup:
                h5file.close()
                h5file = None
                raise IOError, "no Pulsars group in hdf5 file"

            pulsarsgroup = datagroup["Pulsars"]

            for ii in range(len(self.ptapsrs)):
                psrname = self.ptapsrs[ii].name

                #print pulsarsgroup[psrname]['prefitRes'][:]
                #print pulsarsgroup[psrname]['postfitRes'][:]

                pulsarsgroup[psrname]['prefitRes'][:] = self.ptapsrs[ii].residuals
                pulsarsgroup[psrname]['postfitRes'][:] = self.ptapsrs[ii].residuals

                #pulsarsgroup[psrname].create_dataset('prefitRes', data=np.double(self.ptapsrs[ii].residuals))
                #pulsarsgroup[psrname].create_dataset('postfitRes', data=np.double(self.ptapsrs[ii].residuals))

            h5file.close()
            h5file = None



    """
    Based on the signal number and the maximum likelihood parameters, this
    function reconstructs the signal of signum. Very useful to reconstruct only
    the white noise, or red noise, timing residuals.

    This function returns two arrays: the reconstructed signal, and the error

    If the user wants the actual DM signal, he/she can do that him/herself from
    the returned residuals
    """
    def mlPredictionFilter(self, mlparameters, signum=None, selection=None):
        npsrs = len(self.ptapsrs)

        if signum is not None:
            selection = np.array([0]*len(self.ptasignals), dtype=np.bool)
            selection[signum] = True
        elif selection is None:
            # Make a prediction for _all_ signals (i.e. true residuals with
            # timing model paramers correctly removed)
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # The full covariance matrix components
        self.constructPhiAndTheta(mlparameters)
        allPhi = self.Phi.copy()
        allThetavec = self.Thetavec.copy()

        # The covariance matrix components of the prediction signal
        self.constructPhiAndTheta(mlparameters, selection)
        predPhi = self.Phi.copy()
        predThetavec = self.Thetavec.copy()

        # The white noise
        self.setPsrNoise(mlparameters)

        GCGfull = np.zeros((np.sum(self.npgs), np.sum(self.npgs)))
        Cpred = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
        
        totGF = np.zeros((np.sum(self.npgs), np.sum(self.npf)))
        totF = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totG = np.zeros((np.sum(self.npobs), np.sum(self.npgs)))
        totGr = np.zeros(np.sum(self.npgs))
        totDvec = np.zeros(np.sum(self.npobs))

        # Construct the full covariance matrices
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            fdmindex = np.sum(self.npfdm[:ii])
            nfreq = int(self.npf[ii]/2)
            nfreqdm = int(self.npfdm[ii]/2)
            gindex = np.sum(self.npgs[:ii])
            ngs = self.npgs[ii]
            nindex = np.sum(self.npobs[:ii])
            nobs = self.npobs[ii]

            # Start with the white noise
            if self.ptapsrs[ii].twoComponentNoise:
                GCGfull[gindex:gindex+ngs, gindex:gindex+ngs] = \
                        np.dot(self.ptapsrs[ii].Amat.T, \
                        (self.ptapsrs[ii].Nwvec * self.ptapsrs[ii].Amat.T).T)
            else:
                GCGfull[gindex:gindex+ngs, gindex:gindex+ngs] = \
                        np.dot(self.ptapsrs[ii].Gmat.T, \
                        (self.ptapsrs[ii].Nvec * self.ptapsrs[ii].Gmat.T).T)

            # The Phi we cannot add yet. There can be cross-pulsar correlations.
            # Construct a total F-matrix
            totGF[gindex:gindex+ngs, findex:findex+2*nfreq] = \
                    np.dot(self.ptapsrs[ii].Gmat.T, self.ptapsrs[ii].Fmat)
            totF[nindex:nindex+nobs, findex:findex+2*nfreq] = \
                    self.ptapsrs[ii].Fmat
            totG[nindex:nindex+nobs, gindex:gindex+ngs] = self.ptapsrs[ii].Gmat
            totGr[gindex:gindex+ngs] = self.ptapsrs[ii].Gr
            totDvec[nindex:nindex+nobs] = np.diag(self.ptapsrs[ii].Dmat)

            DF = self.ptapsrs[ii].DF
            GDF = np.dot(self.ptapsrs[ii].Gmat.T, self.ptapsrs[ii].DF)

            # Add the dispersion measure variations
            GCGfull[gindex:gindex+ngs, gindex:gindex+ngs] += \
                    np.dot(GDF, (allThetavec[fdmindex:fdmindex+2*nfreqdm] * GDF).T)
            Cpred[nindex:nindex+nobs, nindex:nindex+nobs] += \
                    np.dot(DF, (predThetavec[fdmindex:fdmindex+2*nfreqdm] * DF).T)

        # Now add the red signals, too
        GCGfull += np.dot(totGF, np.dot(allPhi, totGF.T))
        Cpred += np.dot(totF, np.dot(predPhi, totF.T))
        GtCpred = np.dot(totG.T, Cpred)

        # Re-construct the DM variations, and the signal
        try:
            cf = sl.cho_factor(GCGfull)
            GCGr = sl.cho_solve(cf, totGr)
            GCGCp = sl.cho_solve(cf, GtCpred)
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(GCGfull)
            if not np.all(s > 0):
                raise ValueError("ERROR: GCGr singular according to SVD")
            GCGr = np.dot(Vh.T, np.dot(((1.0/s)*U).T, totGr))
            GCGCp = np.dot(Vh.T, np.dot(((1.0/s)*U).T, GtCpred))

        Cti = np.dot(totG, GCGr)
        recsig = np.dot(Cpred, Cti)

        CtGCp = np.dot(Cpred, np.dot(totG, GCGCp))
        recsigCov = Cpred - CtGCp

        return recsig, np.sqrt(np.diag(recsigCov))


    """
    Same as mlPredictionFilter, but this one requires one very specific extra
    object: a likelihood object initialised with the same dataset (par+tim) as
    the previous one, except that it has some virtual toas added to it. These
    virtual toas are for prediction/interpolation.
    """
    def mlPredictionFilter2(self, predlikob, mlparameters, signum=None, selection=None):
        npsrs = len(self.ptapsrs)

        if signum is not None:
            selection = np.array([0]*len(self.ptasignals), dtype=np.bool)
            selection[signum] = True
        elif selection is None:
            # Make a prediction for _all_ signals (i.e. true residuals with
            # timing model paramers correctly removed)
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # The full covariance matrix components
        self.constructPhiAndTheta(mlparameters)
        allPhi = self.Phi.copy()
        allThetavec = self.Thetavec.copy()

        # The covariance matrix components of the prediction signal
        predlikob.constructPhiAndTheta(mlparameters, selection)
        predPhi = self.Phi.copy()
        predThetavec = self.Thetavec.copy()

        # The white noise
        self.setPsrNoise(mlparameters)
        predlikob.setPsrNoise(mlparameters)

        GCGfull = np.zeros((np.sum(self.npgs), np.sum(self.npgs)))
        Cpred = np.zeros((np.sum(predlikob.npobs), np.sum(predlikob.npobs)))
        
        totGF = np.zeros((np.sum(self.npgs), np.sum(self.npf)))
        totF = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totG = np.zeros((np.sum(self.npobs), np.sum(self.npgs)))
        totGr = np.zeros(np.sum(self.npgs))
        totDvec = np.zeros(np.sum(self.npobs))
        totGFp = np.zeros((np.sum(predlikob.npgs), np.sum(predlikob.npf)))
        totFp = np.zeros((np.sum(predlikob.npobs), np.sum(predlikob.npf)))
        totGp = np.zeros((np.sum(predlikob.npobs), np.sum(predlikob.npgs)))
        totGrp = np.zeros(np.sum(predlikob.npgs))
        totDvecp = np.zeros(np.sum(predlikob.npobs))

        # Construct the full covariance matrices
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            fdmindex = np.sum(self.npfdm[:ii])
            nfreq = int(self.npf[ii]/2)
            nfreqdm = int(self.npfdm[ii]/2)
            gindex = np.sum(self.npgs[:ii])
            ngs = self.npgs[ii]
            nindex = np.sum(self.npobs[:ii])
            nobs = self.npobs[ii]
            findexp = np.sum(predlikob.npf[:ii])
            fdmindexp = np.sum(predlikob.npfdm[:ii])
            nfreqp = int(predlikob.npf[ii]/2)
            nfreqdmp = int(predlikob.npfdm[ii]/2)
            gindexp = np.sum(predlikob.npgs[:ii])
            ngsp = predlikob.npgs[ii]
            nindexp = np.sum(predlikob.npobs[:ii])
            nobsp = predlikob.npobs[ii]

            # Start with the white noise
            if self.ptapsrs[ii].twoComponentNoise:
                GCGfull[gindex:gindex+ngs, gindex:gindex+ngs] = \
                        np.dot(self.ptapsrs[ii].Amat.T, \
                        (self.ptapsrs[ii].Nwvec * self.ptapsrs[ii].Amat.T).T)
            else:
                GCGfull[gindex:gindex+ngs, gindex:gindex+ngs] = \
                        np.dot(self.ptapsrs[ii].Gmat.T, \
                        (self.ptapsrs[ii].Nvec * self.ptapsrs[ii].Gmat.T).T)

            # The Phi we cannot add yet. There can be cross-pulsar correlations.
            # Construct a total F-matrix
            totGF[gindex:gindex+ngs, findex:findex+2*nfreq] = \
                    np.dot(self.ptapsrs[ii].Gmat.T, self.ptapsrs[ii].Fmat)
            totF[nindex:nindex+nobs, findex:findex+2*nfreq] = \
                    self.ptapsrs[ii].Fmat
            totG[nindex:nindex+nobs, gindex:gindex+ngs] = self.ptapsrs[ii].Gmat
            totGr[gindex:gindex+ngs] = self.ptapsrs[ii].Gr
            totDvec[nindex:nindex+nobs] = np.diag(self.ptapsrs[ii].Dmat)

            totGFp[gindexp:gindexp+ngsp, findexp:findexp+2*nfreqp] = \
                    np.dot(predlikob.ptapsrs[ii].Gmat.T, predlikob.ptapsrs[ii].Fmat)
            totFp[nindexp:nindexp+nobsp, findexp:findexp+2*nfreqp] = \
                    predlikob.ptapsrs[ii].Fmat
            totGp[nindexp:nindexp+nobsp, gindexp:gindexp+ngsp] = predlikob.ptapsrs[ii].Gmat
            totGrp[gindexp:gindexp+ngsp] = predlikob.ptapsrs[ii].Gr
            totDvecp[nindexp:nindexp+nobsp] = np.diag(predlikob.ptapsrs[ii].Dmat)

            DF = self.ptapsrs[ii].DF
            GDF = np.dot(self.ptapsrs[ii].Gmat.T, self.ptapsrs[ii].DF)
            DFp = predlikob.ptapsrs[ii].DF
            GDFp = np.dot(predlikob.ptapsrs[ii].Gmat.T, predlikob.ptapsrs[ii].DF)

            # Add the dispersion measure variations
            GCGfull[gindex:gindex+ngs, gindex:gindex+ngs] += \
                    np.dot(GDF, (allThetavec[fdmindex:fdmindex+2*nfreqdm] * GDF).T)
            Cpred[nindexp:nindexp+nobsp, nindexp:nindexp+nobsp] += \
                    np.dot(DFp, (predThetavec[fdmindexp:fdmindexp+2*nfreqdmp] * DFp).T)

        # Now add the red signals, too
        GCGfull += np.dot(totGF, np.dot(allPhi, totGF.T))
        Cpred += np.dot(totFp, np.dot(predPhi, totFp.T))

        origlen = totG.shape[0]
        predfulllen = Cpred.shape[0]
        predlen = Cpred.shape[0] - totG.shape[0]

        Bt = Cpred[origlen:, :origlen]
        D = Cpred[origlen:, origlen:]
        BtG = np.dot(Bt, totG)

        # Re-construct the DM variations, and the signal
        try:
            cf = sl.cho_factor(GCGfull)
            GCGr = sl.cho_solve(cf, totGr)
            GCGCp = sl.cho_solve(cf, BtG.T)
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(GCGfull)
            if not np.all(s > 0):
                raise ValueError("ERROR: GCGr singular according to SVD")
            GCGr = np.dot(Vh.T, np.dot(((1.0/s)*U).T, totGr))
            GCGCp = np.dot(Vh.T, np.dot(((1.0/s)*U).T, GtCpred))

        recsig = np.dot(BtG, GCGr)

        CtGCp = np.dot(BtG, GCGCp)
        recsigCov = D - CtGCp

        return recsig, np.diag(recsigCov)







    """
    Test anisotropy code
    """
    def testanicode(self):
        Amp = 5.0e-14
        Si = 4.333
        aniCorr = aniCorrelations(self.ptapsrs, 2)

        npsrs = len(self.ptapsrs)
        Cnoise = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals
        Ctime = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals
        Ciso = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals
        Cani = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals

        totFmat = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totG = np.zeros((np.sum(self.npobs), np.sum(self.npgs)))
        tottoas = np.zeros(np.sum(self.npobs))
        tottoaerrs = np.zeros(np.sum(self.npobs))

        print "Filling auxiliary matrices"

        # Fill the auxiliary matrices and the noise
        for ii in range(npsrs):
            nindex = np.sum(self.npobs[:ii])
            findex = np.sum(self.npf[:ii])
            gindex = np.sum(self.npgs[:ii])
            npobs = self.npobs[ii]
            nppf = self.npf[ii]
            npgs = self.npgs[ii]

            Cnoise[nindex:nindex+npobs, nindex:nindex+npobs] = np.diag(self.ptapsrs[ii].toaerrs**2)
            #totFmat[nindex:nindex+npobs, findex:findex+nppf] = self.ptapsrs[ii].Fmat
            #totDFmat[nindex:nindex+npobs, findex:findex+nppf] = self.ptapsrs[ii].DF
            #totDmat[nindex:nindex+npobs, nindex:nindex+npobs] = self.ptapsrs[ii].Dmat

            totG[nindex:nindex+npobs, gindex:gindex+npgs] = self.ptapsrs[ii].Gmat
            tottoas[nindex:nindex+npobs] = self.ptapsrs[ii].toas
            tottoaerrs[nindex:nindex+npobs] = self.ptapsrs[ii].toaerrs

        print "Constructing the time-domain part"

        Ctime += Cred_sec(tottoas, alpha=0.5*(3-Si), fL=1.0/100) #* (Amp**2)

        print "Calculating the correlations"

        # Fill the GWB matrices
        for ii in range(npsrs):
            nindexi = np.sum(self.npobs[:ii])
            npobsi = self.npobs[ii]

            Ciso[nindexi:nindexi+npobsi, nindexi:nindexi+npobsi] += \
                    Ctime[nindexi:nindexi+npobsi, nindexi:nindexi+npobsi] * \
                    aniCorr.corrhd[ii, ii]

            Cani[nindexi:nindexi+npobsi, nindexi:nindexi+npobsi] += \
                    Ctime[nindexi:nindexi+npobsi, nindexi:nindexi+npobsi] * \
                    (aniCorr.corr[5])[ii, ii]
            
            for jj in range(ii+1, npsrs):
                nindexj = np.sum(self.npobs[:jj])
                npobsj = self.npobs[jj]

                Ciso[nindexi:nindexi+npobsi, nindexj:nindexj+npobsj] += \
                        Ctime[nindexi:nindexi+npobsi, nindexj:nindexj+npobsj] * \
                        aniCorr.corrhd[ii, jj]
                Ciso[nindexj:nindexj+npobsj, nindexi:nindexi+npobsi] += \
                        Ctime[nindexj:nindexj+npobsj, nindexi:nindexi+npobsi] * \
                        aniCorr.corrhd[jj, ii]

                Cani[nindexi:nindexi+npobsi, nindexj:nindexj+npobsj] += \
                        Ctime[nindexi:nindexi+npobsi, nindexj:nindexj+npobsj] * \
                        (aniCorr.corr[5])[ii, jj]
                Cani[nindexj:nindexj+npobsj, nindexi:nindexi+npobsi] += \
                        Ctime[nindexj:nindexj+npobsj, nindexi:nindexi+npobsi] * \
                        (aniCorr.corr[5])[jj, ii]

        # Make all the GCG combinations
        print "Multiplying matrices"
        Gr = np.dot(totG.T, tottoas)
        #GCiG = np.dot(totG.T, np.dot(Ciso + Cnoise, totG))
        #GCaG = np.dot(totG.T, np.dot(Cani, totG))

        GCiG = np.dot(totG.T, np.dot(Cnoise, totG))
        GCaG = np.dot(totG.T, np.dot(Ciso, totG))

        print "Calculating the likelihoods"
        if True:
            # Do not be clever. Just brute-force it for now
            c20 = np.linspace(-5.0, 5.0, 40)
            amp = np.linspace(3.9e-14, 6.1e-14, 40)
            ll = c20.copy()
            #for xi in np.linspace(-5.0, 5.0, 100):
            for ii in range(len(c20)):
                Ctot = GCiG + (amp[ii])**2 * GCaG

                try:
                    cf = sl.cho_factor(Ctot)
                    LD = 2*np.sum(np.log(np.diag(cf[0])))
                    rCr = np.dot(Gr, sl.cho_solve(cf, Gr))
                    ll[ii] = -0.5 * np.sum(rCr) - 0.5*LD
                except:
                    ll[ii] = -1.0e99

                percent = (ii) * 100.0 / len(c20)
                sys.stdout.write("\rScan: %d%%" %percent)
                sys.stdout.flush()
            sys.stdout.write("\n")

        else:
            # Do a two-component model thing
            pass

        np.savetxt('anitest.txt', np.array([amp, ll]).T)





"""
Given a collection of samples, return the 2-sigma confidence intervals
samples: an array of samples
sigmalevel: either 1, 2, or 3. Which sigma limit must be given
onesided: Give one-sided limits (useful for setting upper or lower limits)

"""
def confinterval(samples, sigmalevel=2, onesided=False, weights=None):
  # The probabilities for different sigmas
  sigma = [0.68268949, 0.95449974, 0.99730024]

  bins = 200
  xmin = min(samples)
  xmax = max(samples)

  # If we don't have any weighting (MCMC chain), use the statsmodels package
  if weights is None and sm != None:
    # Create the ecdf function
    ecdf = sm.distributions.ECDF(samples)

    # Create the binning
    x = np.linspace(xmin, xmax, bins)
    y = ecdf(x)
  else:
    # MultiNest chain with weights or no statsmodel.api package
    hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights, density=True)
    x = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])     # This was originally 1.5*, but turns out this is a bug plotting of 'stepstyle' in matplotlib
    y = np.cumsum(hist) / np.sum(hist)

  # Find the intervals
  x2min = y[0]
  if(onesided):
    bound = 1 - sigma[sigmalevel-1]
  else:
    bound = 0.5*(1-sigma[sigmalevel-1])

  for i in range(len(y)):
    if y[i] >= bound:
      x2min = x[i]
      break

  if(onesided):
    bound = sigma[sigmalevel-1]
  else:
    bound = 1 - 0.5 * (1 - sigma[sigmalevel-1])

  for i in reversed(range(len(y))):
    if y[i] <= bound:
      x2max = x[i]
      break

  return x2min, x2max





"""
Given a 2D matrix of (marginalised) likelihood levels, this function returns
the 1, 2, 3- sigma levels. The 2D matrix is usually either a 2D histogram or a
likelihood scan

"""
def getsigmalevels(hist2d):
  # We will draw contours with these levels
  sigma1 = 0.68268949
  level1 = 0
  sigma2 = 0.95449974
  level2 = 0
  sigma3 = 0.99730024
  level3 = 0

  #
  lik = hist2d.reshape(hist2d.size)
  sortlik = np.sort(lik)

  # Figure out the 1sigma level
  dTotal = np.sum(sortlik)
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma1):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level1 = sortlik[nIndex]

  # 2 sigma level
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma2):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level2 = sortlik[nIndex]

  # 3 sigma level
  nIndex = sortlik.size
  dSum = 0
  while (dSum < dTotal * sigma3):
    nIndex -= 1
    dSum += sortlik[nIndex]
  level3 = sortlik[nIndex]

  return level1, level2, level3




"""
Given a collection of (x,y) values, possibly with a weight function, a 2D plot
is made (not shown yet, call plt.show().
Extra options:
  w= weight
  xmin, xmax, ymin, ymax, title

"""
def make2dplot(x, y, w=None, **kwargs):
  title = r'Red noise credible regions'

  # Create a 2d contour for red noise
  xmin = x.min()
  xmax = x.max()
  ymin = y.min()
  ymax = y.max()

  for key in kwargs:
    if key.lower() == 'title':
      title = kwargs[key]
    if key.lower() == 'xmin':
      xmin = kwargs[key]
    if key.lower() == 'xmax':
      xmax = kwargs[key]
    if key.lower() == 'ymin':
      ymin = kwargs[key]
    if key.lower() == 'ymax':
      ymax = kwargs[key]


  hist2d,xedges,yedges = np.histogram2d(x, y, weights=w,\
      bins=40,range=[[xmin,xmax],[ymin,ymax]])
  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]

  xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
  yedges = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])

  level1, level2, level3 = getsigmalevels(hist2d)

  # Set the attributes for the plot
  contourlevels = (level1, level2, level3)
  #contourcolors = ('darkblue', 'darkblue', 'darkblue')
  contourcolors = ('black', 'black', 'black')
  contourlinestyles = ('-', '--', ':')
  contourlinewidths = (3.0, 3.0, 3.0)
  contourlabels = [r'1 $\sigma$', r'2 $\sigma$',r'3 $\sigma$']

  plt.figure()

  line1 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[0], \
      linestyle=contourlinestyles[0], color=contourcolors[0])
  line2 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[1], \
      linestyle=contourlinestyles[1], color=contourcolors[1])
  line3 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[2], \
      linestyle=contourlinestyles[2], color=contourcolors[2])

  contall = (line1, line2, line3)
  contlabels = (contourlabels[0], contourlabels[1], contourlabels[2])

  c1 = plt.contour(xedges,yedges,hist2d.T,contourlevels, \
      colors=contourcolors, linestyles=contourlinestyles, \
      linewidths=contourlinewidths, \
      zorder=2)

  plt.legend(contall, contlabels, loc='upper right',\
      fancybox=True, shadow=True, scatterpoints=1)
  plt.grid(True)

  plt.title(title)
  plt.xlabel(r'Amplitude [$10^{-15}$]')
  plt.ylabel(r'Spectral index $\gamma$ []')
  plt.legend()



"""
Given an mcmc chain file, plot the credible region for the GWB

"""
def makechainplot2d(chainfilename, par1=72, par2=73, xmin=None, xmax=None, ymin=None, ymax=None, title=r"GWB credible regions"):
  emceechain = np.loadtxt(chainfilename)

  if xmin is None:
    #xmin = 0
    xmin = min(emceechain[:,par1+2])
  if xmax is None:
    #xmax = 70
    xmax = max(emceechain[:,par1+2])
  if ymin is None:
    #ymin = 1
    ymin = min(emceechain[:,par2+2])
  if ymax is None:
    #ymax = 7
    ymax = max(emceechain[:,par2+2])

  # Process the parameters

  make2dplot(emceechain[:,par1+2], emceechain[:,par2+2], title=title, \
	  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

"""
Given an mcmc chain file, plot the credible region for the GWB

"""
def makechainplot1d(chainfilename, par=72, xmin=None, xmax=None, title=r"GWB marginalised posterior"):
  emceechain = np.loadtxt(chainfilename)

  if xmin is None:
    xmin = 0
  if xmax is None:
    xmax = 70

  plt.figure()
  plt.hist(emceechain[:, par+2], 100, color="k", histtype="step", range=(xmin, xmax))
  plt.title(title)


"""
Perform a simple scan for two parameters. Fix all parameters to their "start"
values, vary two parameters within their domain

"""
def ScanParameters(likob, scanfilename, par1=0, par2=1):
  ndim = len(likob.pmin)

  p1 = np.linspace(likob.pmin[par1], likob.pmax[par1], 50)
  p2 = np.linspace(likob.pmin[par2], likob.pmax[par2], 50)

  lp1 = np.zeros(shape=(50,50))
  lp2 = np.zeros(shape=(50,50))
  llik = np.zeros(shape=(50,50))
  parameters = likob.pstart.copy()
  for i in range(50):
      for j in range(50):
          lp1[i,j] = p1[i]
          lp2[i,j] = p2[j]
          parameters[par1] = p1[i]
          parameters[par2] = p2[j]
          llik[i,j] = likob.logposterior(parameters)
	  percent = (i*50+j) * 100.0 / (50*50)
	  sys.stdout.write("\rScan: %d%%" %percent)
	  sys.stdout.flush()
  sys.stdout.write("\n")

  col1 = lp1.reshape(50*50)
  col2 = lp2.reshape(50*50)
  col3 = llik.reshape(50*50)
  #col3 = np.exp(lcol3 - np.max(lcol3))

  np.savetxt(scanfilename, np.array([col1, col2, col3]).T)

"""
Perform a simple scan for one parameters. Fix all parameters to their "true"
values, vary one parameters within its domain

"""
def ScanParameter(likob, scanfilename, par1=0):
  ndim = len(likob.pmin)

  p1 = np.linspace(likob.pmin[par1], likob.pmax[par1], 50)

  lp1 = np.zeros(shape=(50))
  llik = np.zeros(shape=(50))
  parameters = likob.pstart.copy()
  for i in range(50):
      lp1[i] = p1[i]
      parameters[par1] = p1[i]
      llik[i] = likob.logposterior(parameters)
      percent = (i) * 100.0 / (50)
      sys.stdout.write("\rScan: %d%%" %percent)
      sys.stdout.flush()
  sys.stdout.write("\n")

  col1 = lp1
  col3 = llik
  #col3 = np.exp(lcol3 - np.max(lcol3))

  np.savetxt(scanfilename, np.array([col1, col3]).T)



"""
Given a scan file, plot the important credible regions

Todo: use make2dplot

"""
def makescanplot(scanfilename):
  likscan = np.loadtxt(scanfilename)

  x = likscan[:,0].reshape(int(np.sqrt(likscan[:,0].size)), \
      int(np.sqrt(likscan[:,0].size)))
  y = likscan[:,1].reshape(int(np.sqrt(likscan[:,1].size)), \
      int(np.sqrt(likscan[:,1].size)))
  ll = likscan[:,2].reshape(int(np.sqrt(likscan[:,2].size)), \
      int(np.sqrt(likscan[:,2].size)))

  lik = np.exp(ll - np.max(ll))

  level1, level2, level3 = getsigmalevels(lik)

  xedges = np.linspace(np.min(x), np.max(x), np.sqrt(x.size))
  yedges = np.linspace(np.min(y), np.max(y), np.sqrt(y.size))

  # Set the attributes for the plot
  contourlevels = (level1, level2, level3)
  #contourcolors = ('darkblue', 'darkblue', 'darkblue')
  contourcolors = ('black', 'black', 'black')
  contourlinestyles = ('-', '--', ':')
  contourlinewidths = (3.0, 3.0, 3.0)
  contourlabels = [r'1 $\sigma$', r'2 $\sigma$',r'3 $\sigma$']

  plt.figure()

  c1 = plt.contour(xedges,yedges,lik.T,contourlevels, \
      colors=contourcolors, linestyles=contourlinestyles, \
      linewidths=contourlinewidths, \
      zorder=2)

  line1 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[0], \
      linestyle=contourlinestyles[0], color=contourcolors[0])
  line2 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[1], \
      linestyle=contourlinestyles[1], color=contourcolors[1])
  line3 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[2], \
      linestyle=contourlinestyles[2], color=contourcolors[2])

  contall = (line1, line2, line3)
  contlabels = (contourlabels[0], contourlabels[1], contourlabels[2])
  plt.legend(contall, contlabels, loc='upper right',\
      fancybox=True, shadow=True, scatterpoints=1)

  plt.grid(True)

  plt.title("GWB noise credible regions")
  plt.xlabel(r'Amplitude [$10^{-15}$]')
  plt.ylabel(r'Spectral index $\gamma$ []')
  plt.legend()


"""
Given a MultiNest file, plot the important credible regions

"""
def makemnplots(mnchainfilename, par1=72, par2=73, minmaxfile=None, xmin=0, xmax=70, ymin=1, ymax=7, title='MultiNest credible regions'):
  mnchain = np.loadtxt(mnchainfilename)

  if minmaxfile is not None:
    minmax = np.loadtxt(minmaxfile)

  nDimensions = mnchain.shape[1]-2

  # Rescale the hypercube parameters
  if minmaxfile is not None:
    for i in range(nDimensions):
      mnchain[:,i+2] = minmax[i,0] + mnchain[:,i+2] * (minmax[i,1] - minmax[i,0])


  # The list of 1D parameters we'd like to check:
#  list1d = np.array([0, 4, 5, nDimensions-2, nDimensions-1])
  list1d = np.array([par1, par2])

  # Create 1d histograms
  for i in list1d:
#    plt.figure()
#    plt.hist(mnchain[:,i+2], 100, color="k", histtype="step")
#    plt.title("Dimension {0:d} (No weight)".format(i))
    plt.figure()
    plt.hist(mnchain[:,i+2], 100, weights=mnchain[:,0], color="k", histtype="step")
    plt.title("Dimension {0:d}".format(i))

  make2dplot(mnchain[:,par1+2], mnchain[:,par2+2], mnchain[:,0], title=title)



"""
Given a DNest file, plot the credible regions

"""
def makednestplots(par1=72, par2=73, xmin=0, xmax=70, ymin=1, ymax=7, title='DNest credible regions'):
  pydnest.dnestresults()

  samples = np.loadtxt('sample.txt')
  weights = np.loadtxt('weights.txt')

  maxlen = len(weights)

  list1d = np.array([par1, par2])

  # Create 1d histograms
  for i in list1d:
    plt.figure()
    plt.hist(samples[:maxlen,i], 100, weights=weights, color="k", histtype="step")
    plt.title("Dimension {0:d}".format(i))

  make2dplot(samples[:maxlen,par1], samples[:maxlen,par2], w=weights, title=title)




"""
Given an mcmc chain file, plot the log-spectrum

"""
def makespectrumplot(chainfilename, parstart=1, numfreqs=10, freqs=None, \
        Apl=None, gpl=None, Asm=None, asm=None, fcsm=0.1, plotlog=False, \
        lcolor='black', Tmax=None):
    if freqs is None:
        ufreqs = np.log10(np.arange(1, 1+numfreqs))
    else:
        ufreqs = np.log10(np.sort(np.array(list(set(freqs)))))

    #ufreqs = np.array(list(set(freqs)))
    yval = np.zeros(len(ufreqs))
    yerr = np.zeros(len(ufreqs))

    emceechain = np.loadtxt(chainfilename)

    if len(ufreqs) != (numfreqs):
        print "WARNING: parameter range does not correspond to #frequencies"

    for ii in range(numfreqs):
        fmin, fmax = confinterval(emceechain[:, parstart+2+ii], sigmalevel=1)
        yval[ii] = (fmax + fmin) * 0.5
        yerr[ii] = (fmax - fmin) * 0.5

    fig = plt.figure()

    # For plotting reference spectra
    spd = 24 * 3600.0
    spy = 365.25 * spd
    pfreqs = 10 ** ufreqs
    ypl = None
    ysm = None

    if plotlog:
        plt.errorbar(ufreqs, yval, yerr=yerr, fmt='.', c=lcolor)
        # outmatrix = np.array([ufreqs, yval, yerr]).T
        # np.savetxt('spectrumplot.txt', outmatrix)

        if Apl is not None and gpl is not None and Tmax is not None:
            Apl = 10**Apl
            ypl = (Apl**2 * spy**3 / (12*np.pi*np.pi * (Tmax))) * ((pfreqs * spy) ** (-gpl))
            plt.plot(np.log10(pfreqs), np.log10(ypl), 'g--', linewidth=2.0)

        if Asm is not None and asm is not None and Tmax is not None:
            Asm = 10**Asm
            fcsm = fcsm / spy
            ysm = (Asm * spy**3 / Tmax) * ((1 + (pfreqs/fcsm)**2)**(-0.5*asm))
            plt.plot(np.log10(pfreqs), np.log10(ysm), 'r--', linewidth=2.0)


        #plt.axis([np.min(ufreqs)-0.1, np.max(ufreqs)+0.1, np.min(yval-yerr)-1, np.max(yval+yerr)+1])
        plt.xlabel("Frequency [log(f/Hz)]")
        #if True:
        #    spd = 24 * 3600.0
        #    spy = 365.25 * spd
        #    #freqs = likobhy.ptapsrs[0].Ffreqs
        #    Tmax = 156038571.88061461
        #    Apl = 10**-13.3 ; Asm = 10**-24
        #    apl = 4.33 ; asm = 4.33
        #    fc = (10**-1.0)/spy

        #    pcsm = (Asm * spy**3 / Tmax) * ((1 + (freqs/fc)**2)**(-0.5*asm))
        #    pcpl = (Apl**2 * spy**3 / (12*np.pi*np.pi * Tmax)) * (freqs*spy) ** (-apl)
        #    plt.plot(np.log10(freqs), np.log10(pcsm), 'r--', linewidth=2.0)
        #    plt.plot(np.log10(freqs), np.log10(pcpl), 'g--', linewidth=2.0)

    else:
        plt.errorbar(10**ufreqs, yval, yerr=yerr, fmt='.', c='black')
        if Aref is not None:
            plt.plot(10**ufreqs, np.log10(yinj), 'k--')
        plt.axis([np.min(10**ufreqs)*0.9, np.max(10**ufreqs)*1.01, np.min(yval-yerr)-1, np.max(yval+yerr)+1])
        plt.xlabel("Frequency [Hz]")

    plt.title("Power spectrum")
    plt.ylabel("Power [log(r)]")
    plt.grid(True)


"""
Given a MultiNest chain file, plot the log-spectrum

"""
def makemnspectrumplot(mnchainfilename, minmaxfile=None, parstart=1, parstop=10, freqs=None):
    ufreqs = np.log10(np.sort(np.array(list(set(freqs)))))
    #ufreqs = np.array(list(set(freqs)))
    yval = np.zeros(parstop-parstart)
    yerr = np.zeros(parstop-parstart)

    if len(ufreqs) != (parstop - parstart):
        print "WARNING: parameter range does not correspond to #frequencies"

    spd = 24 * 3600.0
    spy = 365.25 * spd
    pfreqs = 10 ** ufreqs
    #Aing = 5.0e-14
    Aing = 10**(-13.00)
    Aing = 10**(-10.00)
    #yinj = (Aing**2 * spy**3 / (12*np.pi*np.pi * (5*spy))) * ((pfreqs * spy) ** (-13.0/3.0))
    yinj = (Aing**2 * spy**3 / (12*np.pi*np.pi * (5*spy))) * ((pfreqs * spy) ** (-2.33))
    #print pfreqs * spy
    #print np.log10(yinj)

    mnchain = np.loadtxt(mnchainfilename)
    #emceechain = np.loadtxt(chainfilename)

    if minmaxfile is not None:
        minmax = np.loadtxt(minmaxfile)

    nDimensions = mnchain.shape[1]-2

    # Rescale the hypercube parameters
    if minmaxfile is not None:
        for i in range(nDimensions):
            mnchain[:,i+2] = minmax[i,0] + mnchain[:,i+2] * (minmax[i,1] - minmax[i,0])


    for ii in range(parstop - parstart):
        fmin, fmax = confinterval(mnchain[:, parstart+2+ii], sigmalevel=1, weights=mnchain[:,0])
        yval[ii] = (fmax + fmin) * 0.5
        yerr[ii] = (fmax - fmin) * 0.5

    fig = plt.figure()

    #plt.plot(ufreqs, yval, 'k.-')
    plt.errorbar(ufreqs, yval, yerr=yerr, fmt='.', c='black')
    plt.plot(ufreqs, np.log10(yinj), 'k--')
    plt.title("Periodogram")
    plt.xlabel("Frequency [log(f)]")
    plt.ylabel("Power [log(r)]")
    plt.grid(True)


"""
Given a MultiNest file, plot the credible region for the GWB

"""
def makemnplot2d(mnchainfilename, par1=26, par2=27, xmin=0, xmax=70, ymin=1, ymax=7):
  mnchain = np.loadtxt(mnchainfilename)

  nDimensions = mnchain.shape[1]-2

  # Create 1d histograms
#  for i in list1d[np.where(list1d < nDimensions)]:
#    plt.figure()
#    plt.hist(mnchain[:,i+2], 100, color="k", histtype="step")
#    plt.title("Dimension {0:d} (No weight)".format(i))
#    plt.figure()
#    plt.hist(mnchain[:,i+2], 100, weights=mnchain[:,0], color="k", histtype="step")
#    plt.title("Dimension {0:d}".format(i))

  # make2dplot(emceechain[:,2], emceechain[:,3], title=r'Red noise credible regions')
#  make2dplot(mnchain[:,nDimensions], mnchain[:,nDimensions+1], title=r'GWB credible regions (No weights)')

  make2dplot(mnchain[:,par1], mnchain[:,par2], title=r'credible regions', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)



"""
Given an mcmc chain file, plot the upper limit of one variable as a function of
another

"""
def upperlimitplot2d(chainfilename, par1=72, par2=73, ymin=None, ymax=None):
  emceechain = np.loadtxt(chainfilename)

  if ymin is None:
    #ymin = 1
    ymin = min(emceechain[:,par2+2])
  if ymax is None:
    #ymax = 7
    ymax = max(emceechain[:,par2+2])

  bins = 40
  yedges = np.linspace(ymin, ymax, bins+1)
  deltay = yedges[1] - yedges[0]
  yvals = np.linspace(ymin+deltay, ymax-deltay, bins)
  sigma1 = yvals.copy()
  sigma2 = yvals.copy()
  sigma3 = yvals.copy()

  for i in range(bins):
    # Obtain the indices in the range of the bin
    indices = np.flatnonzero(np.logical_and(
      emceechain[:,par2+2] > yedges[i],
      emceechain[:,par2+2] < yedges[i+1]))

    # Obtain the 1-sided x-sigma upper limit
    a, b = confinterval(emceechain[:,par1+2][indices], sigmalevel=1, onesided=True)
    sigma1[i] = np.exp(b)
    a, b = confinterval(emceechain[:,par1+2][indices], sigmalevel=2, onesided=True)
    sigma2[i] = np.exp(b)
    a, b = confinterval(emceechain[:,par1+2][indices], sigmalevel=3, onesided=True)
    sigma3[i] = np.exp(b)

  plt.figure()

  plt.plot(yvals, sigma1, 'k-', yvals, sigma2, 'k--', yvals, sigma3, 'k:',
      linewidth=3.0)

#  outmatrix = np.array([yvals, sigma1, sigma2, sigma3]).T
#  np.savetxt('eptalimit-sotiris.txt', outmatrix)
#
  plt.legend((r'1 $\sigma$', r'2 $\sigma$',r'3 $\sigma$',), loc='upper right',\
      fancybox=True, shadow=True)
  plt.grid(True)

#
#"""
#Given 2 sample arrays, (and a weight array), plot the upper limit of one
#variable as a function of the other
#"""
#def upperlimitplot2dfromfile(chainfilename, par1=72, par2=73, ymin=None, ymax=None, exponentiate=True):
#


"""
Given a mcmc chain file, this function returns the maximum posterior value, and
the parameters
"""
def getmlfromchain(chainfilename):
  emceechain = np.loadtxt(chainfilename)

  mlind = np.argmax(emceechain[:,1])
  mlpost = emceechain[mlind, 1]
  mlparameters = emceechain[mlind, 2:]

  return (mlpost, mlparameters)


"""
Given a mcmc chain file, plot the log-likelihood values. If it is an emcee
chain, plot the different walkers independently

Maximum number of figures is an optional parameter (for emcee can be large)

"""
def makellplot(chainfilename, numfigs=2):
  emceechain = np.loadtxt(chainfilename)

  uniquechains = set(emceechain[:,0])

  styles = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-',
      'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--',
      'b:', 'g:', 'r:', 'c:', 'm:', 'y:', 'k:']

  # For each chain, plot the ll range
  for i in uniquechains:
      if i < numfigs*len(styles):
          if i % len(styles) == 0:
              plt.figure()
              plt.plot(np.arange(emceechain[(emceechain[:,0]==i),1].size), \
                      emceechain[(emceechain[:,0]==i),1], styles[int(i % len(styles))])

  plt.xlabel("Sample number")
  plt.ylabel("Log-likelihood")
  plt.title("Log-likelihood vs sample number")

"""
Given a rjmcmc chain file, run with the aim of selecting the number of Fourier
modes for both DM and Red noise, this function lets you plot the distributinon
of the number of Fourier coefficients.
"""
def makefouriermodenumberplot(chainfilename, incDM=True):
    rjmcmcchain = np.loadtxt(chainfilename)
    chainmode1 = rjmcmcchain[:,-2]
    chainmode2 = rjmcmcchain[:,-1]

    xmin = np.min(chainmode1)
    xmax = np.max(chainmode1)
    ymin = np.min(chainmode2)
    ymax = np.max(chainmode2)
    totmin = np.min([xmin, ymin])
    totmax = np.max([xmax, ymax])

    xx = np.arange(totmin-1, totmax+2)
    xy = np.zeros(len(xx))
    for ii in range(len(xx)):
        xy[ii] = np.sum(chainmode1==xx[ii]) / len(chainmode1)

    yx = np.arange(totmin-1, totmax+2)
    yy = np.zeros(len(yx))
    for ii in range(len(yx)):
        yy[ii] = np.sum(chainmode2==yx[ii]) / len(chainmode2)

    plt.figure()
    # TODO: why does drawstyle 'steps' shift the x-value by '1'?? Makes no
    #       sense.. probably a bug in the matplotlib package. Check '+0.5'
    plt.plot(xx+0.5, xy, 'r-', drawstyle='steps', linewidth=3.0)
    plt.grid(True, which='major')

    if incDM:
        plt.plot(yx+0.5, yy, 'b-', drawstyle='steps', linewidth=3.0)
        plt.legend((r'Red noise', r'DMV',), loc='upper right',\
                fancybox=True, shadow=True)
    else:
        plt.legend((r'Red noise',), loc='upper right',\
                fancybox=True, shadow=True)

    plt.xlabel('Nr. of frequencies')
    plt.ylabel('Probability')
    plt.grid(True, which='major')


"""
Given a likelihood object, a 'normal' MCMC chain file, and an output directory,
this function spits out a lot of plots summarising all relevant results of the
MCMC
"""
def makeresultsplot(likob, chainfilename, outputdir):
    emceechain = np.loadtxt(chainfilename)

    # List all varying parameters
    dopar = np.array([1]*likob.dimensions, dtype=np.bool)

    # First make a plot of all efac's
    efacparind, efacpsrind, efacnames = likob.getEfacNumbers()

    dopar[efacparind] = False

    if len(efacparind) > 0:
        maxplotpars = 20
        pages = int(1 + len(efacparind) / maxplotpars)
        for pp in range(pages):
            minpar = pp * maxplotpars
            maxpar = min(len(efacparind), minpar + maxplotpars)
            fileout = outputdir+'/efac-page-' + str(pp)

            # Create the plotting data for this plot
            x = np.arange(maxpar-minpar)
            yval = np.zeros(maxpar-minpar)
            yerr = np.zeros(maxpar-minpar)

            for ii in range(maxpar-minpar):
                fmin, fmax = confinterval(emceechain[:, efacparind[ii]+2], sigmalevel=1)
                yval[ii] = (fmax + fmin) * 0.5
                yerr[ii] = (fmax - fmin) * 0.5

            # Now make the plot
            fig = plt.figure()

            #fig = plt.figure(figsize=(10,6))   # Figure size can be adjusted if it gets big
            ax = fig.add_subplot(111)

            #plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
            plt.subplots_adjust(left=0.115, right=0.95, top=0.9, bottom=0.25)

            resp = ax.errorbar(x, yval, yerr=yerr, fmt='.', c='blue')

            ax.axis([-1, max(x)+1, 0, max(yval+yerr)+1])
            ax.xaxis.grid(True, which='major')

            #ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
            #              alpha=1.0)

            ax.set_xticks(np.arange(maxpar-minpar))

            ax.set_title(r'Efac values, page ' + str(pp))
            ax.set_ylabel(r'EFAC')
            #ax.legend(('Rutger', 'Rutger ML', 'Lindley', 'Steve',), shadow=True, fancybox=True, numpoints=1)
            ax.set_yscale('log')

            xtickNames = plt.setp(ax, xticklabels=efacnames[minpar:maxpar])
            #plt.getp(xtickNames)
            plt.setp(xtickNames, rotation=45, fontsize=8, ha='right')

            for ii in range(len(efacnames)):
                print str(efacnames[ii]) + ":  " + str(yval[ii]) + " +/- " + str(yerr[ii])

            plt.savefig(fileout+'.png')
            plt.savefig(fileout+'.eps')

    # Make a plot of the spectra of all pulsars
    spectrumname, spectrumnameshort, spmin, spmax, spfreqs = likob.getSpectraNumbers()

    for ii in range(len(spectrumname)):
        minpar = spmin[ii]
        maxpar = spmax[ii]
        dopar[range(minpar, maxpar)] = False
        fileout = outputdir+'/'+spectrumnameshort[ii]

        # Create the plotting data for this plot
        x = np.log10(np.array(spfreqs[ii]))
        yval = np.zeros(len(x))
        yerr = np.zeros(len(x))

        if (maxpar-minpar) != len(x):
            raise ValueError("ERROR: len(freqs) != maxpar-minpar")

        for jj in range(maxpar-minpar):
            fmin, fmax = confinterval(emceechain[:, minpar+jj+2], sigmalevel=1)
            yval[jj] = (fmax + fmin) * 0.5
            yerr[jj] = (fmax - fmin) * 0.5

        fig = plt.figure()

        plt.errorbar(x, yval, yerr=yerr, fmt='.', c='blue')

        plt.axis([np.min(x)-0.1, np.max(x)+0.1, np.min(yval-yerr)-1, np.max(yval+yerr)+1])
        plt.xlabel("Frequency [log(f/Hz)]")
        plt.ylabel("Power [log(r)]")
        plt.grid(True)
        plt.title(spectrumname[ii])

        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')

    # Make a triplot of all the other parameters
    if np.sum(dopar) > 0:
        indices = np.flatnonzero(np.array(dopar == True))
        triplot(chainfilename, indices)
        fileout = outputdir+'/triplot'
        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')







"""
Run a twalk algorithm on the likelihood wrapper.
Implementation from "pytwalk".

This algorithm used two points. First starts at pstart. Second starts at pstart
+ pwidth

Only every (i % thin) step is saved. So with thin=1 all will be saved

pmin - minimum boundary of the prior domain
pmax - maximum boundary of the prior domain
pstart - starting position in parameter space
pwidth - offset of starting position for second walker
steps - number of MCMC walks to take
"""
def Runtwalk(likob, steps, chainfilename, initfile=None, thin=1, analyse=False):
    # Save the parameters to file
    likob.saveModelParameters(chainfilename + '.parameters.txt')

    # Define the support function (in or outside of domain)
    def PtaSupp(x, xmin=likob.pmin, xmax=likob.pmax):
        return np.all(xmin <= x) and np.all(x <= xmax)

    if initfile is not None:
        ndim = likob.dimensions
        # Obtain starting position from file
        print "Obtaining initial positions from '" + initfile + "'"
        burnindata = np.loadtxt(initfile)
        burnindata = burnindata[:,2:]
        nsteps = burnindata.shape[0]
        dim = burnindata.shape[1]
        if(ndim != dim):
          print "ERROR: burnin file not same dimensions!"
          print "mismatch: ", ndim, dim
          exit()

        # Get starting position
        indices = np.random.randint(0, nsteps, 1)
        p0 = burnindata[indices[0]]
        indices = np.random.randint(0, nsteps, 1)
        p1 = burnindata[indices[0]]

        del burnindata
    else:
        # Obtain starting position from pstart
        p0 = likob.pstart.copy()
        p1 = likob.pstart + likob.pwidth 

    # Initialise the twalk sampler
    #sampler = pytwalk.pytwalk(n=likob.dimensions, U=np_ns_WrapLL, Supp=PtaSupp)
    #sampler = pytwalk.pytwalk(n=likob.dimensions, U=likob.nloglikelihood, Supp=PtaSupp)
    sampler = pytwalk.pytwalk(n=likob.dimensions, U=likob.nlogposterior, Supp=PtaSupp)

    # Run the twalk sampler
    sampler.Run(T=steps, x0=p0, xp0=p1)
    
    # Do some post-processing analysis like the autocorrelation time
    if analyse:
        sampler.Ana()

    indices = range(0, steps, thin)

    savechain = np.zeros((len(indices), sampler.Output.shape[1]+1))
    savechain[:,1] = -sampler.Output[indices, likob.dimensions]
    savechain[:,2:] = sampler.Output[indices, :-1]

    np.savetxt(chainfilename, savechain)


"""
Run a simple RJMCMC algorithm on the single-pulsar data to figure out what the
optimal number of Fourier modes is for both red noise and DM variations

Starting position can be taken from RJMCMC initfile, as can the covariance
matrix

"""
def RunRJMCMC(likob, steps, chainfilename, initfile=None, resize=0.088, \
        jumpprob=0.01, jumpsize1=1, jumpsize2=1, mhinitfile=False):
  # Check the likelihood object, and record the likelihood function
  if likob.likfunc == 'mark7':
      lpfn = likob.mark7logposterior
  elif likob.likfunc == 'mark8':
      lpfn = likob.mark8logposterior
  else:
      raise ValueError("ERROR: must use mark7 or mark8 likelihood functions")

  # Save the parameters to file
  likob.saveModelParameters(chainfilename + '.parameters.txt')

  ndim = likob.dimensions
  pwidth = likob.pwidth.copy()

  if initfile is not None:
    # Read the starting position of the random walkers from a file
    print "Obtaining initial positions from '" + initfile + "'"
    burnindata = np.loadtxt(initfile)
    if mhinitfile:
        burnindata = burnindata[:,2:]
    else:
        burnindata = burnindata[:,2:-2]
    nsteps = burnindata.shape[0]
    dim = burnindata.shape[1]
    if(ndim != dim):
        raise ValueError("ERROR: burnin file not same dimensions!")

    # Get starting position
    indices = np.random.randint(0, nsteps, 1)
    p0 = burnindata[indices[0]]

    # Estimate covariances as being the standarddeviation
    pwidth = resize * np.std(burnindata, axis=0)
  else:
    # Set the starting position of the random walker (add a small perturbation to
    # get away from a possible zero)
    #    p0 = np.random.rand(ndim)*pwidth+pstart
    p0 = likob.pstart + likob.pwidth
    pwidth *= resize

  # Set the covariances
  cov = np.zeros((ndim, ndim))
  for i in range(ndim):
    cov[i,i] = pwidth[i]*pwidth[i]

  # Initialise the emcee Reversible-Jump Metropolis-Hastings sampler
  sampler = rjemcee.RJMHSampler(jumpprob, jumpsize1, jumpsize2, \
          likob.afterJumpPars, likob.proposeNextDimJump, \
          likob.transDimJumpAccepted, cov, ndim, lpfn, args=[])

  mod1, mod2 = likob.proposeNextDimJump()

  # Run the sampler one sample to start the chain
  pos, mod1, mod2, lnprob, state = sampler.run_mcmc(p0, mod1, mod2, 1)

  fil = open(chainfilename, "w")
  fil.close()

  # We don't update the screen every step
  nSkip = 100
  print "Running Metropolis-Hastings sampler"
  for i in range(int(steps/nSkip)):
      for result in sampler.sample(pos, mod1, mod2, iterations=nSkip, storechain=True):
          pos = result[0]
          mod1 = result[1]
          mod2 = result[2]
          lnprob = result[3]
          state = result[4]
          
          fil = open(chainfilename, "a")
          fil.write("{0:4f} \t{1:s} \t{2:s} \t{3:s} \t{4:s}\n".format(0, \
                  str(lnprob), \
                  "\t".join([str(x) for x in pos]), \
                  str(mod1), str(mod2)))
          fil.close()

      percent = i * nSkip * 100.0 / steps
      sys.stdout.write("\rSample: {0:d} = {1:4.1f}%   acc. fr. = {2:f}   mod = {3:d} {4:d}  lnprob = {5:e}   ".format(i*nSkip, percent, \
              np.mean(sampler.acceptance_fraction),\
              mod1, mod2,\
              lnprob))
      sys.stdout.flush()
  sys.stdout.write("\n")

  print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

  try:
      print("Autocorrelation time:", sampler.acor)
  except ImportError:
      print("Install acor from github or pip: http://github.com/dfm/acor")




"""
Run a simple Metropolis-Hastings algorithm on the likelihood wrapper.
Implementation from "emcee"

Starting position can be taken from initfile (just like emcee), and if
covest=True, this file will be used to estimate the stepsize for the mcmc

"""
def RunMetropolis(likob, steps, chainfilename, initfile=None, resize=0.088):
  # Save the parameters to file
  likob.saveModelParameters(chainfilename + '.parameters.txt')

  ndim = likob.dimensions
  pwidth = likob.pwidth.copy()

  if initfile is not None:
    # Read the starting position of the random walkers from a file
    print "Obtaining initial positions from '" + initfile + "'"
    burnindata = np.loadtxt(initfile)
    burnindata = burnindata[:,2:]
    nsteps = burnindata.shape[0]
    dim = burnindata.shape[1]
    if(ndim != dim):
      print "ERROR: burnin file not same dimensions!"
      print "mismatch: ", ndim, dim
      exit()

    # Get starting position
    indices = np.random.randint(0, nsteps, 1)
    p0 = burnindata[indices[0]]

    # Estimate covariances as being the standarddeviation
    pwidth = resize * np.std(burnindata, axis=0)

    del burnindata
  else:
    # Set the starting position of the random walker (add a small perturbation to
    # get away from a possible zero)
    #    p0 = np.random.rand(ndim)*pwidth+pstart
    p0 = likob.pstart + likob.pwidth
    pwidth *= resize

  # Set the covariances
  cov = np.zeros((ndim, ndim))
  for i in range(ndim):
    cov[i,i] = pwidth[i]*pwidth[i]

  # Initialise the emcee Metropolis-Hastings sampler
#  sampler = emcee.MHSampler(cov, ndim, np_WrapLL, args=[pmin, pmax])
  #sampler = emcee.MHSampler(cov, ndim, likob.loglikelihood, args=[])
  sampler = emcee.MHSampler(cov, ndim, likob.logposterior, args=[])

  # Run the sampler one sample to start the chain
  pos, lnprob, state = sampler.run_mcmc(p0, 1)

  fil = open(chainfilename, "w")
  fil.close()

  # We don't update the screen every step
  nSkip = 100
  print "Running Metropolis-Hastings sampler"
  for i in range(int(steps/nSkip)):
      for result in sampler.sample(pos, iterations=nSkip, storechain=True):
          pos = result[0]
          lnprob = result[1]
          state = result[2]
          
          fil = open(chainfilename, "a")
          fil.write("{0:4f} \t{1:s} \t{2:s}\n".format(0, \
                  str(lnprob), \
                  "\t".join([str(x) for x in pos])))
          fil.close()

      percent = i * nSkip * 100.0 / steps
      sys.stdout.write("\rSample: {0:d} = {1:4.1f}%   acc. fr. = {2:f}   pos = {3:e} {4:e}  lnprob = {5:e}   ".format(i*nSkip, percent, \
              np.mean(sampler.acceptance_fraction),\
              pos[ndim-2], pos[ndim-1],\
              lnprob))
      sys.stdout.flush()
  sys.stdout.write("\n")

  print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

  try:
      print("Autocorrelation time:", sampler.acor)
  except ImportError:
      print("Install acor from github or pip: http://github.com/dfm/acor")




"""
Run an ensemble sampler on the likelihood wrapper.
Implementation from "emcee".
"""
def Runemcee(likob, steps, chainfilename, initfile=None, savechain=False, a=2.0):
  # Save the parameters to file
  likob.saveModelParameters(chainfilename + '.parameters.txt')

  ndim = len(likob.pstart)
  nwalkers = 6 * ndim
  mcmcsteps = steps / nwalkers

  if initfile is not None:
    # Read the starting position of the random walkers from a file
    print "Obtaining initial positions from '" + initfile + "'"
    burnindata = np.loadtxt(initfile)
    burnindata = burnindata[:,2:]
    nsteps = burnindata.shape[0]
    dim = burnindata.shape[1]
    if(ndim != dim):
      print "ERROR: burnin file not same dimensions!"
      exit()
    indices = np.random.randint(0, nsteps, nwalkers)
    p0 = [burnindata[i] for i in indices]
  else:
    # Set the starting position of the random walkers
    print "Set random initial positions"
    p0 = [np.random.rand(ndim)*likob.pwidth+likob.pstart for i in range(nwalkers)]

  print "Initialising sampler"
  sampler = emcee.EnsembleSampler(nwalkers, ndim, likob.logposterior,
      args=[], a = a)
  pos, prob, state = sampler.run_mcmc(p0, 1)
  sampler.reset()


  print "Running emcee sampler"
  fil = open(chainfilename, "w")
  fil.close()
  for i in range(mcmcsteps):
      for result in sampler.sample(pos, iterations=1, storechain=False, rstate0=state):
	  pos = result[0]
	  lnprob = result[1]
	  state = result[2]
	  fil = open(chainfilename, "a")
	  for k in range(pos.shape[0]):
	      fil.write("{0:4f} \t{1:s} \t{2:s}\n".format(k, \
		  str(lnprob[k]), \
		  "\t".join([str(x) for x in pos[k]])))
	  fil.close()
      percent = i * 100.0 / mcmcsteps
      sys.stdout.write("\rSample: {0:d} = {1:4.1f}%   acc. fr. = {2:f}   pos = {3:e} {4:e}  lnprob = {5:e}  ".format( \
	      i, percent, \
	      np.mean(sampler.acceptance_fraction), \
	      pos[0,ndim-2], \
	      pos[0,ndim-1], \
	      lnprob[0]))
      sys.stdout.flush()
  sys.stdout.write("\n")

  print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

  if savechain:
    try:
	print("Autocorrelation time:", sampler.acor)
    except ImportError:
	print("Install acor from github or pip: http://github.com/dfm/acor")

  print "Finish wrapper"


def myprior(cube, ndim, nparams):
	#print "cube before", [cube[i] for i in range(ndim)]
	for i in range(ndim):
		cube[i] = cube[i] * 10 * math.pi
	#print "python cube after", [cube[i] for i in range(ndim)]

def myloglike(cube, ndim, nparams):
	chi = 1.
	#print "cube", [cube[i] for i in range(ndim)], cube
	for i in range(ndim):
		chi *= math.cos(cube[i] / 2.)
	#print "returning", math.pow(2. + chi, 5)
	return math.pow(2. + chi, 5)



"""
Run a MultiNest algorithm on the likelihood
Implementation from "pyMultinest"

"""
def RunMultiNest(likob, chainroot, rseed=16, resume=False):
    # Save the parameters to file
    likob.saveModelParameters(chainroot + 'post_equal_weights.dat.mnparameters.txt')

    ndim = likob.dimensions

    if pymultinest is None:
        raise ImportError("pymultinest")

    # Minmax not necessary anymore with the newer multinest version
    """
    # Save the min and max values for the hypercube transform
    cols = np.array([likob.pmin, likob.pmax]).T
    np.savetxt(root+".txt.minmax.txt", cols)
    """

    # Old MultiNest call
#    pymultinest.nested.nestRun(mmodal, ceff, nlive, tol, efr, ndims, nPar, nClsPar, maxModes, updInt, Ztol, root, seed, periodic, fb, resume, likob.logposteriorhc, 0)


    pymultinest.run(likob.logposteriorhc, likob.samplefromprior, ndim,
            importance_nested_sampling = False,
            const_efficiency_mode=False,
            n_clustering_params = None,
            resume = resume,
            verbose = True,
            n_live_points = 500,
            init_MPI = False,
            multimodal = True,
            outputfiles_basename=chainroot,
            n_iter_before_update=100,
            seed=rseed,
            max_modes=100,
            evidence_tolerance=0.5,
            write_output=True,
            sampling_efficiency = 0.3)

    sys.stdout.flush()


"""
Run a DNest algorithm on the likelihood
Implementation from "pyDnest"

"""
def RunDNest(likob, mcmcFile=None, numParticles=1, newLevelInterval=500,\
        saveInterval=100, maxNumLevels=110, lamb=10.0, beta=10.0,\
        deleteParticles=True, maxNumSaves=np.inf):
    # Save the parameters to file
    #likob.saveModelParameters(chainfilename + '.parameters.txt')

    ndim = likob.dimensions

    options = pydnest.Options(numParticles=numParticles,\
            newLevelInterval=newLevelInterval, saveInterval=saveInterval,\
            maxNumLevels=maxNumLevels, lamb=lamb, beta=beta,\
            deleteParticles=deleteParticles, maxNumSaves=maxNumSaves)

    sampler = pydnest.Sampler(pydnest.LikModel, options=options,\
            mcmcFile=mcmcFile, arg=likob)

    sampler.run()

    pydnest.dnestresults()










