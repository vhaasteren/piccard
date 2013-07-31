#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
dataformat.py

Requirements:
- numpy:        pip install numpy
- emcee:        pip install emcee
- h5py:         macports, apt-get, http://h5py.googlecode.com/
- matplotlib:   macports, apt-get
- pytwalk:      (included)


"""

from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import h5py as h5
import sets as sets
import matplotlib.pyplot as plt
import os as os
import sys
import pytwalk                  # Internal module
import pydnest                  # Internal module
import anisotropygammas as ang  # Internal module
import rjmcmchammer as rjemcee  # Internal module

# Optional packages:
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
        uflags = list(sets.Set(t2pulsar.flags))

        # For every flag id, write the values for the TOAs
        print "# For every flag id, write the values for the TOAs"
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
        uniquepulsars = list(sets.Set(sourcepulsars))
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
        ans = (1./math.sqrt(2)/complex(0.,1)) * \
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
        prphi = np.linspace(0, 2*np.pi, self.priorgridbins, endpoint=False)
        prtheta = np.linspace(0, np.pi, self.priorgridbins)
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
                        intg_gamma = ang.int_Gamma_lm(mm, l, \
                                self.phiarr[aa], self.phiarr[bb], \
                                self.thetaarr[aa],self.thetaarr[bb])
                        neg_intg_gamma= (-1)**(mm) * intg_gamma  # (-1)^m Gamma_ml
                        plus_gamma_ml.append(intg_gamma)     # all gammas
                        neg_gamma_ml.append(neg_intg_gamma)  # neg m gammas

                    neg_gamma_ml = neg_gamma_ml[1:]          # Use 0 only once
                    rev_neg_gamma_ml = neg_gamma_ml[::-1]    # Reverse list direction
                    gamma_ml = rev_neg_gamma_ml+plus_gamma_ml

                    # Fill the corrcur matrices for all m
                    mindex = len(self.corr) - mmodes - 1    # Index first m mode
                    for mm in range(mmodes):
                        m = mm - ll

                        self.corr[mindex+mm][aa, bb] = \
                                ang.real_rotated_Gammas(m, ll, \
                                self.phiarr[aa], self.phiarr[bb], \
                                self.thetaarr[aa], self.thetaarr[bb], gamma_ml)

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
            raise ValueError("ERROR: len(clm) != clmlength")

        clmYlm = clm * self.SpHmat
        S = np.sum(clmYlm, axis=1) + 1.0

        return np.all(S > 0)

    # Return the full correlation matrix that depends on the clm. This
    # correlation matrix only needs to be multiplied with the signal amplitude
    # and the time-correlations
    def corrmat(self, clm):
        # Number of clm is 3 + 5 + 7 + ... (2*self.l+1)
        if len(clm) != self.clmlength():
            raise ValueError("ERROR: len(clm) != clmlength")

        corrreturn = self.corrhd.copy()
        index = 0
        for ll in range(1, self.l+1):
            for mm in range(-ll, ll+1):
                corrreturn += clm[index] * self.corr[index]
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
                            # dmspectrum, dmpowerlaw, fouriercoeff
    corr = "single"         # single, gr, uniform, dipole, anisotropicgwb...
                            # Here dipole is not the dipole in anisotropies, but
                            # in 'ephemeris' etc.

    flagname = 'efacequad'  # Name of flag this applies to
    flagvalue = 'noname'    # Flag value this applies to

    npars = 0               # Number of parameters
    ntotpars = 0            # Total number of parameters (also non-varying)
    nindex = 0              # Index in parameters array
    ntotindex = 0           # Index in total par. array. inc. non-varying pars

    bvary = None            # Which parameters are varying of this signals

    pmin = None             # Minimum bound for all parameters (also n.v.)
    pmax = None             # Maximum bound for all parameters (also n.v.)
    pwidth = None           # Stepsize bound for all parameters (also n.v.)
    pstart = None           # Start position for all parameters (also n.v.)

    # Quantities for EFAC/EQUAD
    Nvec = None             # For in the mark3 likelihood

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

mark3loglikelihood: analytically integrated over the red noise Fourier modes. DM
variations are projected on these modes, which is very suboptimal. Do not use
with DM variations. The Phi matrix inversion is not optimised per frequency. At
least one red signal must be included for each pulsar

mark4loglikelihood: analytically integrated over the red noise Fourier modes,
and the DM variation Fourier modes. The integration done separately (red noise
first). At least one red signal must be included for each pulsar, and one DM
signal must be included for each pulsar

mark6loglikelihood: As mark4loglikelihood, but now the DM and red noise Fourier
modes are integrated over simultaneously. Makes for a larger Phi matrix, but
less computational hassle. About as fast as mark4.
"""
class ptaPulsar(object):
    raj = 0
    decj = 0
    toas = None
    toaerrs = None
    residuals = None
    freqs = None
    Gmat = None
    Mmat = None
    ptmpars = []
    ptmdescription = []
    flags = None
    name = "J0000+0000"

    # The auxiliary quantities
    Fmat = None
    Fdmmat = None
    Dmat = None
    DF = None
    Ffreqs = None
    Fdmfreqs = None
    Gr = None
    GGr = None
    GtF = None
    GGtF = None
    GtD = None
    GGtD = None

    # Auxiliaries used in the likelihood
    Nvec = None             # For in the mark3 likelihood (projection approx.)

    # To select the number of Frequency modes
    bfinc = None
    bfdminc = None

    def __init__(self):
        self.raj = 0
        self.decj = 0
        self.toas = None
        self.toaerrs = None
        self.residuals = None
        self.freqs = None
        self.Gmat = None
        self.Mmat = None
        self.ptmpars = []
        self.ptmdescription = []
        self.flags = None
        self.name = "J0000+0000"

        self.Fmat = None
        self.Fdmmat = None
        self.Dmat = None
        self.DF = None
        self.Ffreqs = None
        self.Fdmfreqs = None
        self.Gr = None
        self.GGr = None
        self.GtF = None
        self.GGtF = None
        self.GtD = None
        self.GGtD = None

        self.bfinc = None
        self.bfdminc = None

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

        # See if we can find the Gmatrix
        if not "Gmatrix" in pulsarsgroup[psrname]:
            print "Gmatrix not found for " + psrname + ". Constructing it now."
            U, s, Vh = sl.svd(self.Mmat)
            self.Gmat = U[:, self.Mmat.shape[1]:].copy()
        else:
            self.Gmat = np.array(pulsarsgroup[psrname]['Gmatrix'])

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


    # Modify the design matrix to include fitting for a quadratic in the DM
    # signal.
    # TODO: Check if the DM is fit for in the design matrix
    def addDMQuadratic(self):
        # TODO: Check whether the DM quadratic (and the DM) are already present
        #       in the design matrix
        M = np.zeros((self.Mmat.shape[0], self.Mmat.shape[1]+2))
        Dmatdiag = DMk / (self.freqs**2)
        d = np.array([Dmatdiag*self.toas, Dmatdiag*(self.toas**2)]).T
        #d = np.array([Dmatdiag, Dmatdiag*self.toas, Dmatdiag*(self.toas**2)]).T

        M[:,:-2] = self.Mmat
        M[:,-2:] = d
        self.Mmat = M
        U, s, Vh = sl.svd(self.Mmat)
        self.Gmat = U[:, (self.Mmat.shape[1]):].copy()

    # The number of frequencies is not the number of modes: model = 2*freqs
    def createAuxiliaries(self, Tmax, nfreqs, ndmfreqs):
        (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
        self.Gr = np.dot(self.Gmat.T, self.residuals)
        self.GGr = np.dot(self.Gmat, self.Gr)
        self.GtF = np.dot(self.Gmat.T, self.Fmat)
        self.GGtF = np.dot(self.Gmat, self.GtF)

        # For the DM stuff
        (self.Fdmmat, self.Fdmfreqs) = fourierdesignmatrix(self.toas, 2*ndmfreqs, Tmax)
        self.Dmat = np.diag(DMk / (self.freqs**2))
        self.DF = np.dot(self.Dmat, self.Fdmmat)
        self.GtD = np.dot(self.Gmat.T, self.DF)
        self.GGtD = np.dot(self.Gmat, self.GtD)

        # DM + Red noise stuff (mark6 needs this)
        self.Emat = np.append(self.Fmat, self.DF, axis=1)
        self.GtE = np.dot(self.Gmat.T, self.Emat)
        self.GGtE = np.dot(self.Gmat, self.GtE)

    # When doing Fourier mode selection, like in mark7/mark8, we need an adjusted
    # version of the E-matrix, which only includes certain columns. Select those
    # here
    # bfinc and bfdminc are Boolean arrays indicating which Frequencies to include
    def setLimitedModeAuxiliaries(self, bfinc, bfdminc):
        bfincnp = np.array(bfinc, dtype=np.bool)
        bfdmincnp = np.array(bfdminc, dtype=np.bool)

        if not (np.all(bfincnp == self.bfinc) and np.all(bfdmincnp == self.bfdminc)):
            self.bfinc = bfincnp.copy()
            self.bfdminc = bfdmincnp.copy()

            bf = np.array([bfincnp, bfincnp]).T.flatten()
            bfdm = np.array([bfdmincnp, bfdmincnp]).T.flatten()

            # For mark8
            self.lEmat = np.append(self.Fmat[:,bf], self.DF[:,bfdm], axis=1)
            self.lGtE = np.dot(self.Gmat.T, self.lEmat)
            self.lGGtE = np.dot(self.Gmat, self.lGtE)

            # For mark7
            self.lFmat = self.Fmat[:,bf]
            self.lGtF = self.GtF[:,bf]
            self.lGGtF = self.GGtF[:,bf]

    # Just like 'setLimitedModeAuxiliaries', but now with a number as an argument
    def setLimitedModeNumber(self, nbf, nbfdm):
        bfinc = np.array([0]*self.Fmat.shape[1], dtype=np.bool)
        bfdminc = np.array([0]*self.DF.shape[1], dtype=np.bool)

        bfinc[:nbf] = True
        bfdminc[:nbfdm] = True

        self.setLimitedModeAuxiliaries(bfinc, bfdminc)


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
    phidiag = False

    # What likelihood function to use
    likfunc = 'mark3'

    # Additional informative quantities (reset after RJMCMC jump)
    npf = None      # Number of frequencies per pulsar (red noise/signal)
    npfdm = None    # Number of frequencies per pulsar (DM)
    npe = None      # Number of frequencies per pulsar (rn + DM)
    npobs = None    # Number of observations per pulsar
    npgs = None     # Number of non-projected observations per pulsar (columns Gmat)

    # The Phi, Theta, and Sigma matrices
    Phi = None          # mark1, mark3, mark4, mark6
    Thetavec = None     #               mark4, mark6
    Sigma = None        #        mark3, mark4, mark6
    GNGldet = None      # mark1, mark3, mark4, mark6

    # Other quantities that we do not want to re-initialise every likelihood call
    rGr = None          # mark1, mark3, mark4, mark6
    rGFa = None         # mark1
    aFGFa = None        # mark1
    avec = None         # mark1
    rGF = None          #        mark3, mark4
    rGE = None          #                      mark6
    FGGNGGF = None      #        mark3, mark4
    DGGNGGF = None      #               mark4
    DGGNGGD = None      #               mark4
    EGGNGGE = None      #                      mark6
    NGGF = None         #               mark4
    NGGD = None         #               mark4
    DGXr = None         #               mark4


    def __init__(self):
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

        self.phidiag = False        # If mark1, and no correlations, accelerate phi inversion

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

    def addSignalEfac(self, psrind, index, totindex, separateEfacs=False, \
            varyEfac=True, pmin=0.001, pmax=1000.0, pwidth=0.1, pstart=1.0):
        if separateEfacs:
            uflagvals = list(sets.Set(self.ptapsrs[psrind].flags))   # uniques
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

                newsignal.ntotpars = 1
                if varyEfac:
                    newsignal.npars = 1
                    newsignal.bvary = np.array([1], dtype=np.bool)
                else:
                    newsignal.npars = 0
                    newsignal.bvary = np.array([0], dtype=np.bool)

                newsignal.pmin = np.array([pmin])
                newsignal.pmax = np.array([pmax])
                newsignal.pwidth = np.array([pwidth])
                newsignal.pmax = np.array([pstart])

                newsignal.nindex = index
                newsignal.ntotindex = totindex

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

            newsignal.ntotpars = 1
            if varyEfac:
                newsignal.npars = 1
                newsignal.bvary = np.array([1], dtype=np.bool)
            else:
                newsignal.npars = 0
                newsignal.bvary = np.array([0], dtype=np.bool)

            newsignal.pmin = np.array([pmin])
            newsignal.pmax = np.array([pmax])
            newsignal.pwidth = np.array([pwidth])
            newsignal.pstart = np.array([pstart])

            newsignal.nindex = index
            newsignal.ntotindex = totindex

            self.ptasignals.append(newsignal)

    def addSignalEquad(self, psrind, index, totindex, \
            pmin=-10.0, pmax=-2.0, pwidth=0.1, pstart=-8.0):
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
        newsignal.ntotindex = totindex

        self.ptasignals.append(newsignal)

    def addSignalRedNoise(self, psrind, index, totindex, Tmax, \
            modelIndependent=False):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind

        if modelIndependent:
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        else:
            newsignal.stype = 'powerlaw'
            newsignal.npars = 2
            newsignal.ntotpars = 3
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)

            newsignal.pmin = np.array([-17.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'single'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        newsignal.ntotindex = totindex
        self.ptasignals.append(newsignal)

    def addSignalDMV(self, psrind, index, totindex, Tmax, \
            modelIndependent=False):
        newsignal = ptasignal()
        newsignal.pulsarind = psrind

        if modelIndependent:
            newsignal.stype = 'dmspectrum'
            newsignal.npars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[psrind].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -17.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -14.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        else:
            newsignal.stype = 'dmpowerlaw'
            newsignal.npars = 2
            newsignal.ntotpars = 3
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)

            newsignal.pmin = np.array([-15.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-13.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'single'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        newsignal.ntotindex = totindex
        self.ptasignals.append(newsignal)

    # TODO: use the independent GWB frequencies, instead of those of the first pulsar
    def addSignalGWB(self, index, totindex, Tmax, \
            modelIndependent=False):
        newsignal = ptasignal()
        newsignal.pulsarind = -1

        if modelIndependent:
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        else:
            newsignal.stype = 'powerlaw'
            newsignal.npars = 2
            newsignal.ntotpars = 3
            newsignal.bvary = np.array([1, 1, 0], dtype=np.bool)

            newsignal.pmin = np.array([-17.0, 1.02, 1.0e-11])
            newsignal.pmax = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'gr'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        newsignal.ntotindex = totindex
        newsignal.corrmat = hdcorrmat(self.ptapsrs)           # The H&D matrix
        self.ptasignals.append(newsignal)

    def addSignalClock(self, index, totindex, Tmax, \
            modelIndependent=False):
        newsignal = ptasignal()
        newsignal.pulsarind = -1

        if modelIndependent:
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        else:
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
        newsignal.ntotindex = totindex
        newsignal.corrmat = np.ones((len(self.ptapsrs), len(self.ptapsrs)))
        self.ptasignals.append(newsignal)

    def addSignalDipole(self, index, totindex, Tmax, \
            modelIndependent=False):
        newsignal = ptasignal()
        newsignal.pulsarind = -1

        if modelIndependent:
            newsignal.stype = 'spectrum'
            newsignal.npars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        else:
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
        newsignal.ntotindex = totindex
        newsignal.corrmat = dipolecorrmat(self.ptapsrs)
        self.ptasignals.append(newsignal)

    def addSignalAniGWB(self, index, totindex, Tmax, \
            modelIndependent=False, lAniGWB=2):
        newsignal = ptasignal()
        newsignal.pulsarind = -1
        newsignal.aniCorr = aniCorrelations(self.ptapsrs, lAniGWB)
        nclm = newsignal.aniCorr.clmlength()

        if modelIndependent:
            newsignal.stype = 'spectrum'
            newsignal.npars = nclm+int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.ntotpars = nclm+int(len(self.ptapsrs[0].Ffreqs)/2)
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)

            newsignal.pmin = np.ones(newsignal.ntotpars) * -18.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 10.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * -10.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.1
        else:
            newsignal.stype = 'powerlaw'
            newsignal.npars = nclm+2
            newsignal.ntotpars = nclm+3
            newsignal.bvary = np.array([1]*newsignal.ntotpars, dtype=np.bool)
            newsignal.bvary[-1] = False

            newsignal.pmin = np.ones(newsignal.ntotpars) * -5.0
            newsignal.pmax = np.ones(newsignal.ntotpars) * 5.0
            newsignal.pstart = np.ones(newsignal.ntotpars) * 0.0
            newsignal.pwidth = np.ones(newsignal.ntotpars) * 0.2

            newsignal.pmin[-3:] = np.array([-17.0, 1.02, 1.0e-11])
            newsignal.pmax[-3:] = np.array([-5.0, 6.98, 3.0e-9])
            newsignal.pstart[-3:] = np.array([-14.0, 2.01, 1.0e-10])
            newsignal.pwidth[-3:] = np.array([0.1, 0.1, 5.0e-11])

        newsignal.corr = 'anisotropicgwb'
        newsignal.Tmax = Tmax
        newsignal.nindex = index
        newsignal.ntotindex = totindex
        self.ptasignals.append(newsignal)

    def addSignalFourierCoeff(self, psrind, index, totindex, Tmax, isDM=False):
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
        newsignal.ntotindex = totindex

        # Since this parameter space is so large, calculate the
        # best first-estimate values of these quantities
        # We assume that many auxiliaries have been set already (is done
        # in initModel, so should be ok)
        # TODO: update this to be smarter...
        npars = newsignal.npars
        psr = self.ptapsrs[newsignal.pulsarind]

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




    # TODO: see if we can implement the RJMCMC for the Fourier modes
    def allocateAuxiliaries(self):
        # First figure out how large we have to make the arrays
        npsrs = len(self.ptapsrs)
        self.npf = np.zeros(npsrs, dtype=np.int)
        self.npfdm = np.zeros(npsrs, dtype=np.int)
        self.npobs = np.zeros(npsrs, dtype=np.int)
        self.npgs = np.zeros(npsrs, dtype=np.int)
        for ii in range(npsrs):
            self.npf[ii] = len(self.ptapsrs[ii].Ffreqs)
            self.npfdm[ii] = len(self.ptapsrs[ii].Fdmfreqs)
            self.npobs[ii] = len(self.ptapsrs[ii].toas)
            self.npgs[ii] = self.ptapsrs[ii].Gmat.shape[1]
            self.ptapsrs[ii].Nvec = np.zeros(len(self.ptapsrs[ii].toas))

        if self.likfunc == 'mark1':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGFa = np.zeros(npsrs)
            self.aFGFa = np.zeros(npsrs)
            self.avec = np.zeros(np.sum(self.npf))
        elif self.likfunc == 'mark3' or self.likfunc == 'mark7':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.Sigma = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGF = np.zeros(np.sum(self.npf))
            self.FGGNGGF = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        elif self.likfunc == 'mark4':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Sigma = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGF = np.zeros(np.sum(self.npf))
            self.FGGNGGF = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.DGGNGGF = np.zeros((np.sum(self.npfdm), np.sum(self.npf)))
            self.DGGNGGD = np.zeros((np.sum(self.npfdm), np.sum(self.npfdm)))
            self.NGGF = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
            self.NGGD = np.zeros((np.sum(self.npobs), np.sum(self.npfdm)))
            self.DGXr = np.zeros(np.sum(self.npfdm))
        elif self.likfunc == 'mark6' or self.likfunc == 'mark8':
            self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Sigma = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGE = np.zeros(np.sum(self.npf)+np.sum(self.npfdm))
            self.EGGNGGE = np.zeros((np.sum(self.npf)+np.sum(self.npfdm), np.sum(self.npf)+np.sum(self.npfdm)))


    # Initialise the model
    def initModel(self, nfreqmodes=20, ndmfreqmodes=None, \
            modelIndependentNoise=False, incRedNoise=False, \
            modelIndependentGWB=False, incGWB=False, \
            modelIndependentDM=False, incDM=False, \
            modelIndependentClock=False, incClock=False,  \
            modelIndependentDipole=False, incDipole=False, \
            modelIndependentAniGWB=False, incAniGWB=False, lAniGWB=1, \
            varyEfac=False, incEquad=False, separateEfacs=False, \
            likfunc='mark3'):
        # For every pulsar, construct the auxiliary quantities like the Fourier
        # design matrix etc
        if len(self.ptapsrs) < 1:
            raise IOError, "no pulsars found in hdf5 file"

        Tstart = np.min(self.ptapsrs[0].toas)
        Tfinish = np.max(self.ptapsrs[0].toas)
        self.likfunc = likfunc

        for m2psr in self.ptapsrs:
            Tstart = np.min([np.min(self.ptapsrs[0].toas), Tstart])
            Tfinish = np.max([np.max(self.ptapsrs[0].toas), Tfinish])

        # Total duration of the experiment
        Tmax = Tfinish - Tstart
        for m2psr in self.ptapsrs:
            if incDM:
                m2psr.addDMQuadratic()

            if ndmfreqmodes is None:
                ndmfreqmodes = nfreqmodes

            m2psr.createAuxiliaries(Tmax, nfreqmodes, ndmfreqmodes)

            # When selecting Fourier modes, like in mark7/mark8, the binclude vector
            # indicates whether or not a frequency is included in the likelihood. By
            # default they are all 'on'
            if self.likfunc == 'mark7' or self.likfunc == 'mark8':
                #m2psr.bfinc = np.array([1]*nfreqmodes, dtype=np.bool)
                #m2psr.bfdminc = np.array([1]*ndmfreqmodes, dtype=np.bool)

                m2psr.setLimitedModeAuxiliaries([1]*nfreqmodes, [1]*ndmfreqmodes)

        # Initialise the ptasignal objects
        # Currently: one efac per pulsar, and red noise
        self.ptasignals = []
        index = 0
        totindex = 0
        for ii in range(len(self.ptapsrs)):
            self.addSignalEfac(ii, index, totindex, separateEfacs, varyEfac)
            index += self.ptasignals[-1].npars
            totindex += self.ptasignals[-1].ntotpars

            if incEquad:
                self.addSignalEquad(ii, index, totindex)
                index += self.ptasignals[-1].npars
                totindex += self.ptasignals[-1].ntotpars

            if incRedNoise:
                self.addSignalRedNoise(ii, index, totindex, Tmax, modelIndependentNoise)
                index += self.ptasignals[-1].npars
                totindex += self.ptasignals[-1].ntotpars

            if incDM:
                self.addSignalDMV(ii, index, totindex, Tmax, modelIndependentDM)
                index += self.ptasignals[-1].npars
                totindex += self.ptasignals[-1].ntotpars

        if incGWB:
            self.addSignalGWB(index, totindex, Tmax, modelIndependentGWB)
            index += self.ptasignals[-1].npars
            totindex += self.ptasignals[-1].ntotpars

        if incClock:
            self.addSignalGWB(index, totindex, Tmax, modelIndependentClock)
            index += self.ptasignals[-1].npars
            totindex += self.ptasignals[-1].ntotpars

        if incDipole:
            self.addSignalDipole(index, totindex, Tmax, modelIndependentDipole)
            index += self.ptasignals[-1].npars
            totindex += self.ptasignals[-1].ntotpars

        if incAniGWB:
            self.addSignalAniGWB(index, totindex, Tmax, modelIndependentDipole, lAniGWB)
            index += self.ptasignals[-1].npars
            totindex += self.ptasignals[-1].ntotpars


        # If the frequency coefficients are included explicitly (mark1
        # likelihood), we need a couple of extra signals
        if likfunc=='mark1':
            for ii in range(len(self.ptapsrs)):
                self.addSignalFourierCoeff(ii, index, totindex, Tmax)
                index += self.ptasignals[-1].npars
                totindex += self.ptasignals[-1].ntotpars

                if incDM:
                    self.addSignalFourierCoeff(ii, index, totindex, Tmax, isDM=True)
                    index += self.ptasignals[-1].npars
                    totindex += self.ptasignals[-1].ntotpars

            if incGWB == False and incDM == False:
                # In this case, accelerate the inversion of Phi
                self.phidiag = True

        self.allocateAuxiliaries()

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
    Loop over all signals, and fill the diagonal pulsar noise covariance matrix
    (based on efac/equad)
    """
    def setPsrNoise(self, parameters):
        # For every pulsar, set the noise vector to zero
        for m2psr in self.ptapsrs:
            m2psr.Nvec[:] = 0

        # Loop over all white noise signals, and fill the pulsar Nvec
        for m2signal in self.ptasignals:
            if m2signal.stype == 'efac':
                pefac = 1.0
                if m2signal.npars == 1:
                    pefac = parameters[m2signal.nindex]
                self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pefac**2

                #if m2signal.bvary[0]:
                #    pefac = parameters[m2signal.nindex]
                #else:
                #    pefac = parameters[m2signal.ntotindex]
                #self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pefac**2
            elif m2signal.stype == 'equad':
                pequadsqr = 10**(2*parameters[m2signal.nindex])
                self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pequadsqr

                #if m2signal.bvary[0]:
                #    pequadsqr = 10**(2*parameters[m2signal.nindex])
                #else:
                #    pequadsqr = 10**(2*parameters[m2signal.ntotindex])
                #self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pequadsqr


    """
    Loop over all signals, and fill the phi matrix. This function assumes that
    the self.Phi matrix has already been allocated

    In this version, the DM variations are not included anymore
    """
    def constructPhiAndTheta(self, parameters):
        self.Phi[:] = 0         # Start with a fresh matrix
        self.Thetavec[:] = 0    # ''
        npsrs = len(self.ptapsrs)

        # Loop over all signals, and fill the phi matrix
        for m2signal in self.ptasignals:
            if m2signal.stype == 'spectrum':
                if m2signal.corr == 'single':
                    findex = np.sum(self.npf[:m2signal.pulsarind])
                    nfreq = int(self.npf[m2signal.pulsarind]/2)

                    # Pcdoubled is an array where every element of the parameters
                    # of this m2signal is repeated once (e.g. [1, 1, 3, 3, 2, 2, 5, 5, ...]

                    pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()

                    # Fill the phi matrix
                    di = np.diag_indices(2*nfreq)
                    self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += 10**pcdoubled
                elif m2signal.corr in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                    nfreq = m2signal.npars

                    if m2signal.corr in ['gr', 'uniform', 'dipole']:
                        pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()
                        corrmat = m2signal.corrmat
                    elif m2signal.corr == 'anisotropicgwb':
                        nclm = m2signal.aniCorr.clmlength()
                        pcdoubled = np.array([\
                            parameters[m2signal.nindex:m2signal.nindex+m2signal.npars-nclm],\
                            parameters[m2signal.nindex:m2signal.nindex+m2signal.npars-nclm]]).T.flatten()
                        clm = parameters[m2signal.nindex+m2signal.npars-nclm:m2signal.nindex+m2signal.npars]
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
                            indexb += self.npf[bb]
                        indexb = 0
                        indexa += self.npf[aa]
            elif m2signal.stype == 'dmspectrum':
                if m2signal.corr == 'single':
                    findex = np.sum(self.npfdm[:m2signal.pulsarind])
                    nfreq = int(self.npfdm[m2signal.pulsarind]/2)

                    pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()

                    # Fill the Theta matrix
                    self.Thetavec[findex:findex+2*nfreq] += 10**pcdoubled
            elif m2signal.stype == 'powerlaw':
                spd = 24 * 3600.0
                spy = 365.25 * spd
                Amp = 10**parameters[m2signal.nindex]
                Si = parameters[m2signal.nindex+1]

                if m2signal.corr == 'single':
                    findex = np.sum(self.npf[:m2signal.pulsarind])
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
                        clm = parameters[m2signal.nindex+m2signal.npars-nclm:m2signal.nindex+m2signal.npars]
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
                            indexb += self.npf[bb]
                        indexb = 0
                        indexa += self.npf[aa]
            elif m2signal.stype == 'dmpowerlaw':
                spd = 24 * 3600.0
                spy = 365.25 * spd
                Amp = 10**parameters[m2signal.nindex]
                Si = parameters[m2signal.nindex+1]

                if m2signal.corr == 'single':
                    findex = np.sum(self.npfdm[:m2signal.pulsarind])
                    nfreq = int(self.npfdm[m2signal.pulsarind]/2)
                    freqpy = self.ptapsrs[m2signal.pulsarind].Fdmfreqs * spy
                    # TODO: change the units of the DM signal
                    pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)

                    # Fill the Theta matrix
                    self.Thetavec[findex:findex+2*nfreq] += pcdoubled

    """
    Set the Auxiliary quantities for mark7loglikelihood in all the pulsars,
    based on the psrbfinc boolean arrays. It returns a boolean array for both
    phi and theta, indicating which elements of the covariance matrix to use in
    the likelihood
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
                        psrbfdminc[dmfind:dmfind+fdmlen])

            # Register how many modes we are including (2*number of freqs)
            self.lnpf[ii] = 2*np.sum(self.ptapsrs[ii].bfinc)
            self.lnpfdm[ii] = 2*np.sum(self.ptapsrs[ii].bfdminc)

            find += flen
            dmfind += fdmlen

        # find and dmfind now hold the total number of frequencies
        bfind = np.array([1]*find, dtype=np.bool)
        bfdmind = np.array([1]*dmfind, dtype=np.bool)
        find = 0
        dmfind = 0
        for ii in range(npsrs):
            flen = int(self.ptapsrs[ii].Fmat.shape[1]/2)
            fdmlen = int(self.ptapsrs[ii].DF.shape[1]/2)
            bfind[find:find+flen] = self.ptapsrs[ii].bfinc
            bfdmind[dmfind:dmfind+fdmlen] = self.ptapsrs[ii].bfdminc

            find += flen
            dmfind += fdmlen

        return bfind, bfdmind

    """
    Convert a number of frequencies for RN and DMV to a boolean array that
    indicates which frequencies to use
    """
    def getPsrLimFreqFromNumbers(self, psrnfinc, psrnfdminc):
        npsrs = len(self.ptapsrs)
        find = 0
        dmfind = 0
        for ii in range(npsrs):
            flen = int(self.ptapsrs[ii].Fmat.shape[1]/2)
            fdmlen = int(self.ptapsrs[ii].DF.shape[1]/2)
            find += flen
            dmfind += fdmlen

        psrbfinc = np.array([0]*flen, dtype=np.bool)
        psrbfdminc = np.array([0]*fdmlen, dtype=np.bool)
        find = 0
        dmfind = 0
        for ii in range(npsrs):
            flen = int(self.ptapsrs[ii].Fmat.shape[1]/2)
            fdmlen = int(self.ptapsrs[ii].DF.shape[1]/2)
            psrbfinc[find:find+psrnfinc[ii]] = True
            psrbfdminc[dmfind:dmfind+psrnfdminc[ii]] = True
            find += flen
            dmfind += fdmlen

        return psrbfinc, psrbfdminc

    """
    When doing an RJMCMC, this function proposed the next possible jump in both
    RN and DM spaces. This only works for a model with a single pulsar. Zero
    modes for RN or DM are not allowed.
    """
    def proposeNextDimJump(self):
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

        # Either jump in one dimension, or the other. Not both
        if np.random.rand() < 0.5:
            if maxmod1 == 1:
                propmod1 = curmod1
            elif curmod1 == 1 and curmod1 < maxmod1:
                propmod1 = curmod1 + 1
            elif curmod1 == maxmod1 and curmod1 > 1:
                propmod1 = curmod1 - 1
            elif curmod1 > 1 and curmod1 < maxmod1:
                propmod1 = curmod1 - 1 + 2*np.random.randint(2)
        else:
            if maxmod2 == 1:
                propmod2 = curmod2
            elif curmod2 == 1 and curmod2 < maxmod2:
                propmod2 = curmod2 + 1
            elif curmod2 == maxmod2 and curmod2 > 1:
                propmod2 = curmod2 - 1
            elif curmod2 > 1 and curmod1 < maxmod2:
                propmod2 = curmod2 - 1 + 2*np.random.randint(2)

        return propmod1, propmod2

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
    mark1 loglikelihood of the pta model/likelihood implementation

    This likelihood is similar to mark3, but it uses the frequency coefficients
    explicitly in the likelihood. Mark3 marginalises over them analytically.
    Therefore, this mark1 version requires some extra parameters in the model,
    all part of an extra auxiliary 'signal'.
    
    TODO: (Including DM variations not yet implemented)
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
        # algebra. First we'll invert Phi
        if self.phidiag:
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

            # One temporary quantity
            # This is equivalent to np.dot(np.diag(1.0/Nvec, GGtF))
            NGGF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].GGtF.T).T
            # This was too slow
            # NGGF = np.array([(1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].GGtF[:,i] for i in range(self.ptapsrs[ii].Fmat.shape[1])]).T

            # Fill the auxiliaries
            nobs = len(self.ptapsrs[ii].toas)
            ng = self.ptapsrs[ii].Gmat.shape[1]

            self.rGr[ii] = np.sum(self.ptapsrs[ii].GGr ** 2 / self.ptapsrs[ii].Nvec)
            self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].GGr, NGGF)
            self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) * ng / nobs
            self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].GGtF.T, NGGF)

        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi
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
        return -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD



    """
    mark4 loglikelihood of the pta model/likelihood implementation

    Like the mark3 version, but now also implements DM variation correction by
    adding a separate analytical integration over DM Fourier modes. This is
    necessary, since the DM variations and red noise work in a different set of
    basis functions
    """
    def mark4loglikelihood(self, parameters):
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
            nindex = np.sum(self.npobs[:ii])
            ncurobs = self.npobs[ii]
            nfreq = int(self.npf[ii]/2)
            nfreqdm = int(self.npfdm[ii]/2)

            # This is equivalent to np.dot(np.diag(1.0/Nvec, GGtF))
            self.NGGF[nindex:nindex+ncurobs, findex:findex+2*nfreq] = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].GGtF.T).T
            self.NGGD[nindex:nindex+ncurobs, fdmindex:fdmindex+2*nfreqdm] = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].GGtD.T).T

            # Fill the auxiliaries
            self.rGr[ii] = np.sum(self.ptapsrs[ii].GGr ** 2 / self.ptapsrs[ii].Nvec)
            self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].GGr, self.NGGF[nindex:nindex+ncurobs, findex:findex+2*nfreq])
            self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) * self.npgs[ii] / self.npobs[ii]
            self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].GGtF.T, self.NGGF[nindex:nindex+ncurobs, findex:findex+2*nfreq])
            self.DGGNGGF[fdmindex:fdmindex+2*nfreqdm, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].GGtD.T, self.NGGF[nindex:nindex+ncurobs, findex:findex+2*nfreq])
            self.DGGNGGD[fdmindex:fdmindex+2*nfreqdm, fdmindex:fdmindex+2*nfreqdm] = np.dot(self.ptapsrs[ii].GGtD.T, self.NGGD[nindex:nindex+ncurobs, fdmindex:fdmindex+2*nfreqdm])

        
        # Now that those arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi
        cfPhi = sl.cho_factor(self.Phi)
        PhiLD = 2*np.sum(np.log(np.diag(cfPhi[0])))
        Phiinv = sl.cho_solve(cfPhi, np.identity(self.Phi.shape[0]))

        self.Sigma = self.FGGNGGF + Phiinv
        # Construct and decompose Sigma
        try:
            cfSigma = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cfSigma[0])))
            SigmaGr = sl.cho_solve(cfSigma, self.rGF)
            rGSigmaGr = np.dot(self.rGF, SigmaGr)

            SigDGGNGGF = sl.cho_solve(cfSigma, self.DGGNGGF.T)
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(self.Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            SigmaGr = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF)))
            rGSigmaGr = np.dot(self.rGF, SigmaGr)

            SigDGGNGGF = np.dot(U, np.dot(np.diag(1.0/s), np.dot(Vh, self.DGGNGGF.T)))

        for ii in range(npsrs):
            fdmindex = np.sum(self.npfdm[:ii])
            nindex = np.sum(self.npobs[:ii])
            ncurobs = self.npobs[ii]
            nfreqdm = int(self.npf[ii]/2)
            self.DGXr[fdmindex:fdmindex+2*nfreqdm] = np.dot(self.ptapsrs[ii].GGtD.T,\
                    self.ptapsrs[ii].GGr / self.ptapsrs[ii].Nvec) - \
                    np.dot(self.ptapsrs[ii].GGtD.T, \
                    np.dot(self.NGGF[nindex:nindex+ncurobs, findex:findex+2*nfreq], \
                    SigmaGr[findex:findex+2*nfreq]))

        # For the DM spectrum, we'll construct the matrix Y
        # (Need Sigma for this)
        di = np.diag_indices(self.DGGNGGD.shape[0])
        Ymat = self.DGGNGGD - np.dot(self.DGGNGGF, SigDGGNGGF)
        Ymat[di] += 1.0/self.Thetavec
        ThetaLD = np.sum(np.log(self.Thetavec))
        try:
            cfY = sl.cho_factor(Ymat)
            YLD = 2*np.sum(np.log(np.diag(cfY[0])))

            # Finally, the last inner product
            rDGYGDr = np.dot(self.DGXr, sl.cho_solve(cfY, self.DGXr))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Ymat)
            YLD = np.sum(np.log(s))

            # Finally, the last inner product
            rDGYGDr = np.dot(self.DGXr, np.dot(Vh.T, np.dot(np.diag(1.0/s), \
                    np.dot(U.T, self.DGXr))))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.rGr) + 0.5*rGSigmaGr + 0.5*rDGYGDr \
               -0.5*np.sum(self.GNGldet) - 0.5*PhiLD - 0.5*SigmaLD \
               -0.5*ThetaLD - 0.5*YLD


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

            # One temporary quantity
            NGGE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].GGtE.T).T

            # Fill the auxiliaries
            nobs = len(self.ptapsrs[ii].toas)
            ng = self.ptapsrs[ii].Gmat.shape[1]
            self.rGr[ii] = np.sum(self.ptapsrs[ii].GGr ** 2 / self.ptapsrs[ii].Nvec)
            self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].GGr, NGGE)
            self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) * ng / nobs
            self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].GGtE.T, NGGE)

        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi
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

        ThetaLD = np.sum(np.log(self.Thetavec))

        # Construct and decompose Sigma
        di = np.diag_indices(np.sum(self.npf))
        didm = np.diag_indices(np.sum(self.npfdm))
        Sigma = self.EGGNGGE
        Sigma[0:np.sum(self.npf), 0:np.sum(self.npf)] += Phiinv
        Sigma[np.sum(self.npf):, np.sum(self.npf):][didm] += 1.0 / self.Thetavec
        try:
            cf = sl.cho_factor(Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGE, sl.cho_solve(cf, self.rGE))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGE, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGE))))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD - 0.5*ThetaLD


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
        bfind, bfdmind = self.prepareLimFreqIndicators(psrbfinc, psrbfdminc)

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
            #fdmindex = np.sum(self.lnpfdm[:ii])
            nfreq = int(self.lnpf[ii]/2)
            #nfreqdm = int(self.lnpfdm[ii]/2)

            # One temporary quantity
            NGGF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].lGGtF.T).T

            # Fill the auxiliaries
            nobs = len(self.ptapsrs[ii].toas)
            ng = self.ptapsrs[ii].Gmat.shape[1]

            self.rGr[ii] = np.sum(self.ptapsrs[ii].GGr ** 2 / self.ptapsrs[ii].Nvec)
            self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].GGr, NGGF)
            self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) * ng / nobs
            self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].lGGtF.T, NGGF)

        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi
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

        #ThetaLD = np.sum(np.log(lThetavec))

        # Construct and decompose Sigma
        #didm = np.diag_indices(np.sum(self.lnpfdm))
        Sigma = self.FGGNGGF[:lenphi, :lenphi]
        Sigma[0:np.sum(self.lnpf), 0:np.sum(self.lnpf)] += Phiinv
        #Sigma[np.sum(self.lnpf):, np.sum(self.lnpf):][didm] += 1.0 / lThetavec
        try:
            cf = sl.cho_factor(Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGE[:lenphi], sl.cho_solve(cf, self.rGE[:lenphi]))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGF[:lenphi], np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF[:lenphi]))))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD - 0.5*ThetaLD



    """
    mark8 loglikelihood of the pta model/likelihood implementation

    This likelihood is the same as mark6loglikelihood, except that it allows for
    a variable number of Fourier modes to be included, both for DM and for red
    noise

    psrbfinc, psrbfdminc: a boolean array, indicating which frequencies to
                          include.
    psrnfinc, psrnfdminc: integer array, indicating how many frequencies per
                          pulsar to include. Overrides psrbfinc and psrbfdminc
    """
    def mark8loglikelihood(self, parameters, psrbfinc=None, psrbfdminc=None, \
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
        bfind, bfdmind = self.prepareLimFreqIndicators(psrbfinc, psrbfdminc)

        # Double up the frequency indicators to get the mode indicators
        bfmind = np.array([bfind, bfind]).T.flatten()
        bfmdmind = np.array([bfdmind, bfdmind]).T.flatten()

        # Select the limited range Phi and Theta
        #lPhi = self.Phi[bfmind, bfmind]
        lPhi = self.Phi[:, bfmind][bfmind, :]
        lThetavec = self.Thetavec[bfmdmind]
        lenphi = np.sum(bfmind)
        lentheta = np.sum(bfmdmind)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.lnpf[:ii])
            fdmindex = np.sum(self.lnpfdm[:ii])
            nfreq = int(self.lnpf[ii]/2)
            nfreqdm = int(self.lnpfdm[ii]/2)

            # One temporary quantity
            NGGE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].lGGtE.T).T

            # Fill the auxiliaries
            nobs = len(self.ptapsrs[ii].toas)
            ng = self.ptapsrs[ii].Gmat.shape[1]
            self.rGr[ii] = np.sum(self.ptapsrs[ii].GGr ** 2 / self.ptapsrs[ii].Nvec)
            self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].GGr, NGGE)
            self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) * ng / nobs
            self.EGGNGGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm, findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].lGGtE.T, NGGE)

        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi
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

        ThetaLD = np.sum(np.log(lThetavec))

        # Construct and decompose Sigma
        didm = np.diag_indices(np.sum(self.lnpfdm))
        Sigma = self.EGGNGGE[:lenphi+lentheta, :lenphi+lentheta]
        Sigma[0:np.sum(self.lnpf), 0:np.sum(self.lnpf)] += Phiinv
        Sigma[np.sum(self.lnpf):, np.sum(self.lnpf):][didm] += 1.0 / lThetavec
        try:
            cf = sl.cho_factor(Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGE[:lenphi+lentheta], sl.cho_solve(cf, self.rGE[:lenphi+lentheta]))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGE[:lenphi+lentheta], np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGE[:lenphi+lentheta]))))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD - 0.5*ThetaLD




    def loglikelihood(self, parameters):
        ll = 0.0

        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            if self.likfunc == 'mark1':
                ll = self.mark1loglikelihood(parameters)
            elif self.likfunc == 'mark3':
                ll = self.mark3loglikelihood(parameters)
            elif self.likfunc == 'mark4':
                ll = self.mark4loglikelihood(parameters)
            elif self.likfunc == 'mark6':
                ll = self.mark6loglikelihood(parameters)
            elif self.likfunc == 'mark7':
                ll = self.mark7loglikelihood(parameters)
            elif self.likfunc == 'mark8':
                ll = self.mark8loglikelihood(parameters)
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
                lp += parameters[m2signal.nindex]
                clm = parameters[m2signal.nindex+m2signal.npars-nclm:m2signal.nindex+m2signal.npars]
                if m2signal.aniCorr.priorIndicator(clm) == False:
                    lp -= 1e99
            elif m2signal.stype == 'powerlaw':
                lp += parameters[m2signal.nindex]
            elif m2signal.stype == 'spectrum' and m2signal.corr == 'anisotropicgwb':
                nclm = m2signal.aniCorr.clmlength()
                lp += np.sum(parameters[m2signal.nindex:m2signal.nindex+m2signal.npars])
                clm = parameters[m2signal.nindex+m2signal.npars-nclm:m2signal.nindex+m2signal.npars]
                if m2signal.aniCorr.priorIndicator(clm) == False:
                    lp -= 1e99
            elif m2signal.stype == 'spectrum':
                lp += np.sum(parameters[m2signal.nindex:m2signal.nindex+m2signal.npars])

            # Divide by the prior range
            if np.sum(m2signal.bvary) > 0:
                lp -= np.sum(np.log(m2signal.pmax[m2signal.bvary]-m2signal.pmin[m2signal.bvary]))
        return lp

    # Note: the inclusion of a uniform-amplitude part can have a big influence
    def mark7logprior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        #return self.mark4logprior(parameters)
        lp = 0.0

        if psrnfinc != None and psrnfdminc != None:
            psrbfinc, psrbfdminc = self.getPsrLimFreqFromNumbers(psrnfinc, psrnfdminc)

        # Obtain the frequency selectors, and set the psr frequencies
        bfind, bfdmind = self.prepareLimFreqIndicators(psrbfinc, psrbfdminc)

        # Loop over all signals
        for m2signal in self.ptasignals:
            if m2signal.stype == 'spectrum' and m2signal.corr == 'single':
                # Red noise, see if we need to include it
                findex = int(np.sum(self.npf[:m2signal.pulsarind])/2)
                nfreq = int(self.npf[m2signal.pulsarind]/2)
                inc = bfind[findex:findex+nfreq]

                if np.sum(inc) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[inc] - m2signal.pmin[inc]))
            elif m2signal.stype == 'dmspectrum' and m2signal.corr == 'single':
                fdmindex = int(np.sum(self.npfdm[:m2signal.pulsarind])/2)
                nfreqdm = int(self.npfdm[m2signal.pulsarind]/2)
                inc = bfdmind[fdmindex:fdmindex+nfreqdm]

                if np.sum(inc) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[inc] - m2signal.pmin[inc]))
            else:
                if np.sum(m2signal.bvary) > 0:
                    lp -= np.sum(np.log(m2signal.pmax[m2signal.bvary]-m2signal.pmin[m2signal.bvary]))

        return lp

    def mark8logprior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        return self.mark7logprior(parameters, psrbfinc, psrbfdminc, \
                psrnfinc, psrnfdminc)

    def mark7logposterior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        lp = self.mark7logprior(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        if lp > -1e98:
            lp += self.mark7loglikelihood(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        return lp

    def mark8logposterior(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        lp = self.mark8logprior(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        if lp > -1e98:
            lp += self.mark8loglikelihood(parameters, psrbfinc, psrbfdminc, psrnfinc, psrnfdminc)

        return lp

    def logprior(self, parameters):
        lp = 0.0

        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            if self.likfunc == 'mark1':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark3':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark4':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark6':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark6':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'mark7':  # Mark7 should be called differently of course
                lp = self.mark7logprior(parameters)
            elif self.likfunc == 'mark8':  # Mark8 ''
                lp = self.mark8logprior(parameters)
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

    def logposteriorhc(self, cube):
        parameters = self.pmin + cube * (self.pmax - self.pmin)
        return self.logposterior(parameters)

    def samplefromprior(self):
        return self.pmin + np.random.rand(self.dimensions) * (self.pmax - self.pmin)


    """
    Simple signal generation, use frequency domain for power-law signals by
    default
    """
    def gensig(self, parameters=None, filename=None, timedomain=False):
        if parameters == None:
            parameters = self.pstart

        npsrs = len(self.ptapsrs)
        npf = np.zeros(npsrs, dtype=np.int)
        nobs = np.zeros(npsrs, dtype=np.int)
        ngs = np.zeros(npsrs, dtype=np.int)
        for ii in range(npsrs):
            npf[ii] = len(self.ptapsrs[ii].Ffreqs)
            nobs[ii] = len(self.ptapsrs[ii].toas)
            ngs[ii] = self.ptapsrs[ii].Gmat.shape[1]

        Phi = np.zeros((np.sum(npf), np.sum(npf)))      # Freq. domain red signals
        Theta = np.zeros((np.sum(npf), np.sum(npf)))    # Freq. domain DM signals
        Cr = np.zeros((np.sum(nobs), np.sum(nobs)))     # Time domain red signals
        Cdm = np.zeros((np.sum(nobs), np.sum(nobs)))     # Time domain red signals

        # For every pulsar, set the noise vector to zero
        for m2psr in self.ptapsrs:
            m2psr.Nvec = np.zeros(len(m2psr.toas))

        # Loop over all white noise signals, and fill the pulsar Nvec
        for m2signal in self.ptasignals:
            if m2signal.stype == 'efac':
                pefac = 1.0
                if m2signal.npars == 1:
                    pefac = parameters[m2signal.nindex]
                self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pefac**2
            elif m2signal.stype == 'equad':
                pequadsqr = 10**(2*parameters[m2signal.nindex])
                self.ptapsrs[m2signal.pulsarind].Nvec += m2signal.Nvec * pequadsqr


        # Now that we have the white noise, only thing left is to fill the Phi
        # matrix
        for m2signal in self.ptasignals:
            if m2signal.stype == 'spectrum':
                if m2signal.corr == 'single':
                    findex = np.sum(npf[:m2signal.pulsarind])
                    nfreq = int(npf[m2signal.pulsarind]/2)

                    # Pcdoubled is an array where every element of the parameters
                    # of this m2signal is repeated once (e.g. [1, 1, 3, 3, 2, 2, 5, 5, ...]
                    pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()

                    # Fill the phi matrix
                    di = np.diag_indices(2*nfreq)
                    Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += 10**pcdoubled
                elif m2signal.corr in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                    nfreq = m2signal.npars

                    if m2signal.corr in ['gr', 'uniform', 'dipole']:
                        pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()
                        corrmat = m2signal.corrmat
                    elif m2signal.corr == 'anisotropicgwb':
                        nclm = m2signal.aniCorr.clmlength()
                        pcdoubled = np.array([\
                            parameters[m2signal.nindex:m2signal.nindex+m2signal.npars-nclm],\
                            parameters[m2signal.nindex:m2signal.nindex+m2signal.npars-nclm]]).T.flatten()
                        clm = parameters[m2signal.nindex+m2signal.npars-nclm:m2signal.nindex+m2signal.npars]
                        corrmat = m2signal.aniCorr.corrmat(clm)

                    indexa = 0
                    indexb = 0
                    for aa in range(npsrs):
                        for bb in range(npsrs):
                            # Some pulsars may have fewer frequencies than
                            # others (right?). So only use overlapping ones
                            nof = np.min([npf[aa], npf[bb], 2*nfreq])
                            di = np.diag_indices(nof)
                            Phi[indexa:indexa+nof,indexb:indexb+nof][di] += 10**pcdoubled[:nof] * corrmat[aa, bb]
                            indexb += npf[bb]
                        indexb = 0
                        indexa += npf[aa]
            if m2signal.stype == 'dmspectrum':
                findex = np.sum(npf[:m2signal.pulsarind])
                nfreq = int(npf[m2signal.pulsarind]/2)

                # Pcdoubled is an array where every element of the parameters
                # of this m2signal is repeated once (e.g. [1, 1, 3, 3, 2, 2, 5, 5, ...]
                pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()

                # Fill the phi matrix: transformed DM power spectrum
                Theta[findex:findex+2*nfreq, findex:findex+2*nfreq] += np.diag(10**pcdoubled)

            elif m2signal.stype == 'powerlaw':
                spd = 24 * 3600.0
                spy = 365.25 * spd
                Amp = 10**parameters[m2signal.nindex]
                Si = parameters[m2signal.nindex+1]

                if m2signal.corr == 'single':
                    nindex = np.sum(nobs[:m2signal.pulsarind])
                    npobs = nobs[m2signal.pulsarind]
                    findex = np.sum(npf[:m2signal.pulsarind])
                    nfreq = int(npf[m2signal.pulsarind]/2)

                    if timedomain:
                        Cr[nindex:nindex+npobs,nindex:nindex+npobs] +=\
                                Cred_sec(self.ptapsrs[m2signal.pulsarind].toas,\
                                alpha=0.5*(3-Si),\
                                fL=1.0/40) * (Amp**2)
                    else:
                        freqpy = self.ptapsrs[m2signal.pulsarind].Ffreqs * spy
                        pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)

                        # Fill the phi matrix
                        di = np.diag_indices(2*nfreq)
                        Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                elif m2signal.corr in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                    freqpy = self.ptapsrs[0].Ffreqs * spy
                    pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)
                    nfreq = len(freqpy)

                    if m2signal.corr in ['gr', 'uniform', 'dipole']:
                        corrmat = m2signal.corrmat
                    elif m2signal.corr == 'anisotropicgwb':
                        nclm = m2signal.aniCorr.clmlength()
                        clm = parameters[m2signal.nindex+m2signal.npars-nclm:m2signal.nindex+m2signal.npars]
                        corrmat = m2signal.aniCorr.corrmat(clm)

                    indexa = 0
                    indexb = 0
                    for aa in range(npsrs):
                        for bb in range(npsrs):
                            # Some pulsars may have fewer frequencies than
                            # others (right?). So only use overlapping ones
                            nof = np.min([npf[aa], npf[bb]])
                            if nof > nfreq:
                                raise IOError, "ERROR: nof > nfreq. Adjust GWB freqs"

                            di = np.diag_indices(nof)
                            Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * corrmat[aa, bb]
                            indexb += npf[bb]
                        indexb = 0
                        indexa += npf[aa]
            elif m2signal.stype == 'dmpowerlaw':
                spd = 24 * 3600.0
                spy = 365.25 * spd
                Amp = 10**parameters[m2signal.nindex]
                Si = parameters[m2signal.nindex+1]

                nindex = np.sum(nobs[:m2signal.pulsarind])
                npobs = nobs[m2signal.pulsarind]
                findex = np.sum(npf[:m2signal.pulsarind])
                nfreq = int(npf[m2signal.pulsarind]/2)

                if timedomain:
                    Cdm[nindex:nindex+npobs,nindex:nindex+npobs] +=\
                            Cred_sec(self.ptapsrs[m2signal.pulsarind].toas,\
                            alpha=0.5*(3-Si),\
                            fL=1.0/40) * (Amp**2)
                else:
                    freqpy = self.ptapsrs[m2signal.pulsarind].Ffreqs * spy
                    pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)
                    
                    Theta[findex:findex+2*nfreq, findex:findex+2*nfreq] += np.diag(pcdoubled)


        # We have both the white noise, and the red noise. Construct the total
        # time-domain covariance matrix.
        Cov = np.zeros((np.sum(nobs), np.sum(nobs)))
        totFmat = np.zeros((np.sum(nobs), np.sum(npf)))
        totDFmat = np.zeros((np.sum(nobs), np.sum(npf)))
        totDmat = np.zeros((np.sum(nobs), np.sum(nobs)))
        totG = np.zeros((np.sum(nobs), np.sum(ngs)))
        tottoas = np.zeros(np.sum(nobs))
        tottoaerrs = np.zeros(np.sum(nobs))
        for ii in range(npsrs):
            nindex = np.sum(nobs[:ii])
            findex = np.sum(npf[:ii])
            gindex = np.sum(ngs[:ii])
            npobs = nobs[ii]
            nppf = npf[ii]
            npgs = ngs[ii]
            Cov[nindex:nindex+npobs, nindex:nindex+npobs] = np.diag(self.ptapsrs[ii].Nvec)
            totFmat[nindex:nindex+npobs, findex:findex+nppf] = self.ptapsrs[ii].Fmat
            totDFmat[nindex:nindex+npobs, findex:findex+nppf] = self.ptapsrs[ii].DF
            totDmat[nindex:nindex+npobs, nindex:nindex+npobs] = self.ptapsrs[ii].Dmat

            totG[nindex:nindex+npobs, gindex:gindex+npgs] = self.ptapsrs[ii].Gmat
            tottoas[nindex:nindex+npobs] = self.ptapsrs[ii].toas
            tottoaerrs[nindex:nindex+npobs] = self.ptapsrs[ii].toaerrs

        Cov += np.dot(totFmat, np.dot(Phi, totFmat.T))
        Cov += np.dot(totDFmat, np.dot(Theta, totDFmat.T))
        Cov += Cr
        Cov += np.dot(totDmat, np.dot(Cdm, totDmat))

        GCG = np.dot(totG.T, np.dot(Cov, totG))

        cf = sl.cholesky(GCG).T

        xi = np.random.randn(GCG.shape[0])
        ygen = np.dot(totG, np.dot(cf, xi))
        #ygen = np.dot(cf, xi)

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
  if weights is None or sm is None:
    # Create the ecdf function
    ecdf = sm.distributions.ECDF(samples)

    # Create the binning
    x = np.linspace(xmin, xmax, bins)
    y = ecdf(x)
  else:
    # MultiNest chain with weights or no statsmodel.api package
    hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights, density=True)
    x = np.delete(xedges, -1) + 1.5*(xedges[1] - xedges[0])
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
  parameters = likob.pstart
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
  parameters = likob.pstart
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
def makespectrumplot(chainfilename, parstart=1, parstop=10, freqs=None):
    ufreqs = np.log10(np.sort(np.array(list(sets.Set(freqs)))))
    #ufreqs = np.array(list(sets.Set(freqs)))
    yval = np.zeros(parstop-parstart)
    yerr = np.zeros(parstop-parstart)

    spd = 24 * 3600.0
    spy = 365.25 * spd
    pfreqs = 10 ** ufreqs
    #Aing = 5.0e-14
    Aing = 10**(-13.00)
    #yinj = (Aing**2 * spy**3 / (12*np.pi*np.pi * (5*spy))) * ((pfreqs * spy) ** (-13.0/3.0))
    yinj = (Aing**2 * spy**3 / (12*np.pi*np.pi * (5*spy))) * ((pfreqs * spy) ** (-5.33))
    #print pfreqs * spy
    #print np.log10(yinj)

    emceechain = np.loadtxt(chainfilename)

    if len(ufreqs) != (parstop - parstart):
        print "WARNING: parameter range does not correspond to #frequencies"

    for ii in range(parstop - parstart):
        fmin, fmax = confinterval(emceechain[:, parstart+2+ii], sigmalevel=1)
        yval[ii] = (fmax + fmin) * 0.5
        yerr[ii] = (fmax - fmin) * 0.5

    fig = plt.figure()

    #plt.plot(ufreqs, yval, 'k.-')
    plt.errorbar(ufreqs, yval, yerr=yerr, fmt='.', c='black')
    #plt.plot(ufreqs, np.log10(yinj), 'k--')
    plt.title("Periodogram")
    plt.xlabel("Frequency [log(f)]")
    plt.ylabel("Power [log(r)]")
    plt.grid(True)


"""
Given a MultiNest chain file, plot the log-spectrum

"""
def makemnspectrumplot(mnchainfilename, minmaxfile=None, parstart=1, parstop=10, freqs=None):
    ufreqs = np.log10(np.sort(np.array(list(sets.Set(freqs)))))
    #ufreqs = np.array(list(sets.Set(freqs)))
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
def makemnplot2d(mnchainfilename, minmaxfile=None, par1=26, par2=27, xmin=0, xmax=70, ymin=1, ymax=7):
  mnchain = np.loadtxt(mnchainfilename)

  if minmaxfile is not None:
    minmax = np.loadtxt(minmaxfile)

  nDimensions = mnchain.shape[1]-2

  # Rescale the hypercube parameters
  if minmaxfile is not None:
    for i in range(nDimensions):
      mnchain[:,i+2] = minmax[i,0] + mnchain[:,i+2] * (minmax[i,1] - minmax[i,0])

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

  make2dplot(mnchain[:,par1+2], mnchain[:,par2+2], mnchain[:,0], title=r'GWB credible regions', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)



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
Given a mcmc chain file, plot the log-likelihood values. If it is an emcee
chain, plot the different walkers independently

Maximum number of figures is an optional parameter (for emcee can be large)

"""
def makellplot(chainfilename, numfigs=2):
  emceechain = np.loadtxt(chainfilename)

  uniquechains = sets.Set(emceechain[:,0])

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
    # TODO: why does drawstyle 'steps' shift the x-value by '1'?? Makes no sense..
    plt.plot(xx+0.5, xy, 'r-', drawstyle='steps', linewidth=3.0)
    plt.grid(True, which='major')

    if incDM:
        plt.plot(yx+0.5, yy, 'b-', drawstyle='steps', linewidth=3.0)
        plt.legend((r'Red noise', r'DMV',), loc='upper right',\
                fancybox=True, shadow=True)
    else:
        plt.legend((r'Red noise',), loc='upper right',\
                fancybox=True, shadow=True)


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
def Runtwalk(likob, steps, chainfilename, thin=1):
  # Define the support function (in or outside of domain)
  def PtaSupp(x, xmin=likob.pmin, xmax=likob.pmax):
    return np.all(xmin <= x) and np.all(x <= xmax)

  p0 = likob.pstart
  p1 = likob.pstart + likob.pwidth 

  # Initialise the twalk sampler
  #sampler = pytwalk.pytwalk(n=likob.dimensions, U=np_ns_WrapLL, Supp=PtaSupp)
  #sampler = pytwalk.pytwalk(n=likob.dimensions, U=likob.nloglikelihood, Supp=PtaSupp)
  sampler = pytwalk.pytwalk(n=likob.dimensions, U=likob.nlogposterior, Supp=PtaSupp)

  # Run the twalk sampler
  sampler.Run(T=steps, x0=p0, xp0=p1)
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
        jumpprob=0.01, mhinitfile=False):
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
  sampler = rjemcee.RJMHSampler(jumpprob, likob.afterJumpPars, \
          likob.proposeNextDimJump, cov, ndim, \
          likob.mark7logposterior, args=[])

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
Run a simple Metropolis-Hastings algorithm on the likelihood wrapper.
Implementation from "emcee"

Starting position can be taken from initfile (just like emcee), and if
covest=True, this file will be used to estimate the stepsize for the mcmc

"""
def RunMetropolis(likob, steps, chainfilename, initfile=None, resize=0.088):
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




"""
Run a MultiNest algorithm on the likelihood
Implementation from "pyMultinest"

"""
def RunMultiNest(likob, chainroot):
    ndim = likob.dimensions

    if pymultinest is None:
        raise ImportError("pymultinest")

    mmodal = True       # search for multiple modes
    ceff = 0            # SMBH tuning
    nlive = 500        # number of "live" points
    tol = 0.5           # final tolerance in computing evidence
    efr = 0.80          # sampling efficiency (0.8 and 0.3 are recommended for parameter estimation & evidence evaluation)
    ndims = ndim        # number of search parameters
    nPar = ndims        # number of reported parameters (see above)
    nClsPar = 1         # number of parameters on which to attempt mode separation (first nClsPar parameters used)
    maxModes = 40       # maximum number of modes
    updInt = 100        # interval of updates
    Ztol = -1.e90       # threshold for reporting evidence about modes
    root = chainroot    # prefix for output files
    seed = 16           # seed for random numbers
    periodic = np.ones(ndims)    # period conditions
    fb = True           # output status updates?
    resume = False      # resume from previous run?

    # Save the min and max values for the hypercube transform
    cols = np.array([likob.pmin, likob.pmax]).T
    np.savetxt(root+"minmax.txt", cols)

    pymultinest.nested.nestRun(mmodal, ceff, nlive, tol, efr, ndims, nPar, nClsPar, maxModes, updInt, Ztol, root, seed, periodic, fb, resume, likob.logposteriorhc, 0)

    sys.stdout.flush()


"""
Run a DNest algorithm on the likelihood
Implementation from "pyDnest"

"""
def RunDNest(likob, mcmcFile=None, numParticles=1, newLevelInterval=500,\
        saveInterval=100, maxNumLevels=110, lamb=10.0, beta=10.0,\
        deleteParticles=True, maxNumSaves=np.inf):
    ndim = likob.dimensions

    options = pydnest.Options(numParticles=numParticles,\
            newLevelInterval=newLevelInterval, saveInterval=saveInterval,\
            maxNumLevels=maxNumLevels, lamb=lamb, beta=beta,\
            deleteParticles=deleteParticles, maxNumSaves=maxNumSaves)

    sampler = pydnest.Sampler(pydnest.LikModel, options=options,\
            mcmcFile=mcmcFile, arg=likob)

    sampler.run()

    pydnest.dnestresults()


