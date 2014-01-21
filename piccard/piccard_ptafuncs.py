#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
piccard_ptafunc.py

Simple functions that calculate some basic quantities useful for PTA projects.
Examples:
- Hellings & Downs curve
- Quadratic spindown matrix
- Fourier modes for red noise and DM variations
- Time-domain red-noise covariance matrix
- Anisotropy-correlations

"""

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
from .piccard_constants import *


# Block-wise multiplication as in G^{T}CG
def blockmul(A, B, psrobs, psrg):
    """Computes B.T . A . B, where B is a block-diagonal design matrix
        with block heights m = len(A) / len(meta) and block widths m - meta[i]['pars'].

        >>> a = N.random.randn(8,8)
        >>> a = a + a.T
        >>> b = N.zeros((8,5),'d')
        >>> b[0:4,0:2] = N.random.randn(4,2)
        >>> b[4:8,2:5] = N.random.randn(4,3)
        >>> psrobs = [4, 4]
        >>> psrg = [2, 3]
        >>> c = blockmul(a,b,psrobs, psrg) - N.dot(b.T,N.dot(a,b))
        >>> N.max(N.abs(c))
        0.0
    """

    n, p = A.shape[0], B.shape[1]    # A is n x n, B is n x p

    if (A.shape[0] != A.shape[1]) or (A.shape[1] != B.shape[0]):
        raise ValueError('incompatible matrix sizes')
    
    if (len(psrobs) != len(psrg)):
        raise ValueError('incompatible matrix description')

    res1 = np.zeros((n,p), 'd')
    res2 = np.zeros((p,p), 'd')

    npulsars = len(psrobs)
    #m = n/npulsars          # times (assumed the same for every pulsar)

    psum, isum = 0, 0
    for i in range(npulsars):
        # each A matrix is n x m, with starting column index = i * m
        # each B matrix is m x (m - p_i), with starting row = i * m, starting column s = sum_{k=0}^{i-1} (m - p_i)
        # so the logical C dimension is n x (m - p_i), and it goes to res1[:,s:(s + m - p_i)]
        res1[:,psum:psum+psrg[i]] = np.dot(A[:,isum:isum+psrobs[i]],B[isum:isum+psrobs[i], psum:psum+psrg[i]])
            
        psum += psrg[i]
        isum += psrobs[i]

    psum, isum = 0, 0
    for i in range(npulsars):
        res2[psum:psum+psrg[i],:] = np.dot(B.T[psum:psum+psrg[i], isum:isum+psrobs[i]], res1[isum:isum+psrobs[i],:])
                    
        psum += psrg[i]
        isum += psrobs[i]

    return res2

"""
Scipy 0.7.x does not yet have block_diag, and somehow I have some troubles
upgrading it on the ATLAS cluster. So for now, include the source here in
piccard as well. -- Rutger van Haasteren (December 2013)
"""
def block_diag(*arrs):
    """Create a block diagonal matrix from the provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Parameters
    ----------
    A, B, C, ... : array-like, up to 2D
        Input arrays.  A 1D array or array-like sequence with length n is
        treated as a 2D array with shape (1,n).

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

    References
    ----------
    .. [1] Wikipedia, "Block matrix",
           http://en.wikipedia.org/wiki/Block_diagonal_matrix

    Examples
    --------
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> print(block_diag(A, B, C))
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args) 

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out


"""
Calculate the daily-averaging exploder matrix, and the daily averaged site
arrival times. In the modelling, the residuals will not be changed. It is only
for calculating correlations

@param toas:        vector of site arrival times. (Seconds)
@param calcInverse: Boolean that indicates whether the pseudo-inverse of Umat needs
                    to be calculated

@return:            Either (avetoas, Umat), with avetoas the everage toas, and Umat
                    the exploder matrix. Or (avetoas, Umat, Ui), with Ui the
                    pseudo-inverse of Umat

Input is a vector of site arrival times. Returns the reduced-size average toas,
and the exploder matrix  Cfull = Umat Cred Umat^{T}
Of the output, a property of the matrices Umat and Ui is that:
np.dot(Ui, Umat) = np.eye(len(avetoas))

TODO: Make more 'Pythonic'
"""
def dailyaveragequantities(toas, calcInverse=False):
    timespan = 10       # Same observation if within 10 seconds

    processed = np.array([0]*len(toas), dtype=np.bool)  # No toas processed yet
    Umat = np.zeros((len(toas), 0))
    avetoas = np.empty(0)

    while not np.all(processed):
        npindex = np.where(processed == False)[0]
        ind = npindex[0]
        satmin = toas[ind] - timespan
        satmax = toas[ind] + timespan

        dailyind = np.where(np.logical_and(toas > satmin, toas < satmax))[0]

        newcol = np.zeros((len(toas)))
        newcol[dailyind] = 1.0

        Umat = np.append(Umat, np.array([newcol]).T, axis=1)
        avetoas = np.append(avetoas, np.mean(toas[dailyind]))
        processed[dailyind] = True

    returnvalues = (avetoas, Umat)

    # Calculate the pseudo-inverse if necessary
    if calcInverse:
        Ui = ((1.0/np.sum(Umat, axis=0)) * Umat).T
        returnvalues = (avetoas, Umat, Ui)

    return returnvalues



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

Mode 0: sin(f_0)
Mode 1: cos(f_0)
Mode 2: sin(f_1)
... etc

@param nmodes:  The number of modes that will be included (= 2*nfreq)
@param Ttot:    Total duration experiment (in case not given by t)
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
  if Ttot is None:
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

