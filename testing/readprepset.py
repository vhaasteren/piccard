#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
readprepset.py

Requirements:
- numpy:        pip install numpy
- matplotlib:   macports, apt-get

Created by vhaasteren on 2013-08-06.
Copyright (c) 2013 Rutger van Haasteren

Work that uses this code should reference van Haasteren et al. (in prep). (I'll
add the reference later).

This is a collection of tools based on the IPTA student workshop code in Krabi
2013.
It contains functions for the 'optimal correlation statistic' (Demorest et
al. 2013, arXiv:1201.664, 2013ApJ...762...94D ; Chamberlain et al. in prep)
It also contains continuous wave code written by Justin Ellis, arXiv:1305:0835

"""

from __future__ import division
import math, numpy as np, matplotlib.pyplot as plt, os as os, sys as sys, glob as glob
import scipy.linalg as sl, scipy.special as ss
import libstempo as T
from scipy.optimize import minimize_scalar


# Create both the pulsar objects and the linear model
def readPrepRealisations(path):
    psrs = []
    ntotobs, ntotpars = 0, 0
    relparpath = '/par/'
    reltimpath = '/tim/'
    reldespath = '/design/'
    relrespath = '/res/'
    relcovpath = '/cov/'

    # Read in all the pulsars through libstempo
    for infile in glob.glob(os.path.join(path+relparpath, '*.par')):
        filename = os.path.splitext(infile)
        basename = os.path.basename(filename[0])

        # Locate the par-file and the timfile
        parfile = path+relparpath+basename+'.par'
        timfile = path+reltimpath+basename+'.tim'
        desfile = path+reldespath+basename+'designMatrix.txt'

        # Make the libstempo object
        timfiletup = os.path.split(timfile)
        dirname = timfiletup[0]
        reltimfile = timfiletup[-1]
        relparfile = os.path.relpath(parfile, dirname)
        savedir = os.getcwd()
        os.chdir(dirname)
        psrs.append(T.tempopulsar(relparfile, reltimfile))
        os.chdir(savedir)

        # Read the designmatrix from the file
        #desmat = psrs[-1].designmatrix().copy()
        desmat = np.loadtxt(desfile)

        # Register how many paramters and observations we have
        #ntotobs += psr.nobs
        #ntotpars += psr.ndim+1
        ntotobs += desmat.shape[0]
        ntotpars += desmat.shape[1]
        if desmat.shape[0] != psrs[-1].nobs:
            print "For " + basename + " obs != obs ", desmat.shape[0], psrs[-1].nobs

        if desmat.shape[1] != psrs[-1].ndim+1:
            print "For " + basename + " pars != pars ", desmat.shape[1], psrs[-1].ndim+1
        
    hdmat = hdcormat(psrs)

    # Allocate memory for the model description
    toas = np.zeros(ntotobs)
    residuals = np.zeros(ntotobs)
    toaerrs = np.zeros(ntotobs)
    designmatrix = np.zeros((ntotobs, ntotpars))
    Gmatrix = np.zeros((ntotobs, ntotobs-ntotpars))
    GNGinv = []
    ptheta = []
    pphi = []
    psrobs = []
    psrpars = []
    psrg = []


    infiles = glob.glob(os.path.join(path+relparpath, '*.par'))
    indo, indp, indg = 0, 0, 0
    for i in range(len(infiles)):
        # This automatically also loops over the psrs of course
        filename = os.path.splitext(infiles[i])
        basename = os.path.basename(filename[0])

        # Locate the par-file and the timfile
        parfile = path+relparpath+basename+'.par'
        timfile = path+reltimpath+basename+'.tim'
        desfile = path+reldespath+basename+'designMatrix.txt'
        covfile = path+relcovpath+basename+'covMatrix.txt'
        resfile = path+relrespath+basename+'res.dat'

        # Read in the design matrix (again), covariance matrix, and the residuals
        desmat = np.loadtxt(desfile)
        covmat = np.loadtxt(covfile)
        resvec = np.loadtxt(resfile)
        # These comments are for when reading in from the psr object
        #desmat = psrs[i].designmatrix().copy()
        #covmat = np.diag((psrs[i].toaerrs*1e-6)**2)
        #resvec = np.array([psrs[i].toas(), psrs[i].residuals(), psrs[i].toaerrs*1e-6]).T
        
        # Determine the dimensions
        pobs = desmat.shape[0]
        ppars = desmat.shape[1]
        pgs = desmat.shape[0] - desmat.shape[1]
        psrobs.append(pobs)
        psrpars.append(ppars)
        psrg.append(pgs)

        # Load the basic quantities
        toas[indo:indo+pobs] = resvec[:,0]
        residuals[indo:indo+pobs] = resvec[:,1]
        toaerrs[indo:indo+pobs] = resvec[:,2]

        designmatrix[indo:indo+pobs, indp:indp+ppars] = desmat
        ptheta.append(0.5*np.pi - psrs[i]['DECJ'].val)
        pphi.append(psrs[i]['RAJ'].val)

        # Create the G-matrix
        U, s, Vh = sl.svd(desmat)
        Gmatrix[indo:indo+pobs, indg:indg+pgs] = U[:,ppars:].copy()

        # Create the noise matrix
        pNoise = covmat
        GNG = np.dot(U[:,ppars:].copy().T, np.dot(pNoise, U[:,ppars:].copy()))
        cf = sl.cho_factor(GNG)
        GNGinv.append(sl.cho_solve(cf, np.identity(GNG.shape[0])))

        indo += pobs
        indp += ppars
        indg += pgs
    
    model = (toas, residuals, toaerrs, designmatrix, Gmatrix, hdmat, psrobs, psrpars, psrg, GNGinv, ptheta, pphi)

    return (psrs, model)



# Function to read all par/tim files in a specific directory
def readRealisations(path):
    """ Uses the libstempo package to make tempo2 read a set of par/tim files
        in a specific directory. The package libstempo can be found at:
        https://github.com/vallis/mc3pta/tree/master/stempo
        credit: Michele Vallisneri

        The directory 'path' will be scanned for .par files and similarly named
        .tim files.
        
        @param path: path to the directory with par/tim files
        
    """

    psrs = []
    for infile in glob.glob(os.path.join(path, '*.par')):
        filename = os.path.splitext(infile)
        psrs.append(T.tempopulsar(parfile=filename[0]+'.par',timfile=filename[0]+'.tim'))
    
    return psrs

    


# Construct the Hellings & Downs correlation matrix
def hdcormat(psrs):
    """ Constructs a correlation matrix consisting of the Hellings & Downs
        correlation coefficients. See Eq. (A30) of Lee, Jenet, and
        Price ApJ 684:1304 (2008) for details.

        @param: list of libstempo pulsar objects
                (as returned by readRealisations)
        
    """
    npsrs = len(psrs)
    raj = [psrs[i]['RAJ'].val for i in range(npsrs)]
    decj = [psrs[i]['DECJ'].val for i in range(npsrs)]
    pp = np.array([np.cos(decj)*np.cos(raj), np.cos(decj)*np.sin(raj), np.sin(decj)]).T
    cosp = np.array([[np.dot(pp[i], pp[j]) for i in range(npsrs)] for j in range(npsrs)])
    cosp[cosp > 1.0] = 1.0
    xp = 0.5 * (1 - cosp)

    old_settings = np.seterr(all='ignore')
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old_settings)
    hdmat = logxp - 0.25 * xp + 0.5 + 0.5 * np.diag(np.ones(len(psrs)))

    if False: # Plot the H&D curve
        angle = np.arccos(cosp)
        x = np.array(angle.flat)
        y = np.array(hdmat.flat)
        ind = np.argsort(x)
        plt.plot(x[ind], y[ind], c='b', marker='.')

    return hdmat


# Function to get the basic quantities for a full array
def makeLinearModel(psrs):
    """ Parses the libstempo objects to obtain the linear objects that are
        used to describe the likelihood function of van Haasteren et al. (2009),
        MNRAS, 395, 1005V

        @param: a list of libstempo objects as returned by readRealisations
    """
    ntotobs, ntotpars = 0, 0
    for psr in psrs:
        ntotobs += psr.nobs
        ntotpars += psr.ndim+1
    
    toas = np.zeros(ntotobs)
    residuals = np.zeros(ntotobs)
    toaerrs = np.zeros(ntotobs)
    designmatrix = np.zeros((ntotobs, ntotpars))
    Gmatrix = np.zeros((ntotobs, ntotobs-ntotpars))
    GNGinv = []
    ptheta = []
    pphi = []

    indo, indp, indg = 0, 0, 0
    for i in range(len(psrs)):
        toas[indo:indo+psrs[i].nobs] = psrs[i].toas()
        residuals[indo:indo+psrs[i].nobs] = psrs[i].residuals()
        toaerrs[indo:indo+psrs[i].nobs] = psrs[i].toaerrs * 1.0e-6
        designmatrix[indo:indo+psrs[i].nobs, indp:indp+psrs[i].ndim+1] = psrs[i].designmatrix().copy()
        ptheta.append(0.5*np.pi - psrs[i]['DECJ'].val)
        pphi.append(psrs[i]['RAJ'].val)
        
        U, s, Vh = sl.svd(psrs[i].designmatrix())
        
        Gmatrix[indo:indo+psrs[i].nobs, indg:indg+(psrs[i].nobs-psrs[i].ndim-1)] = U[:,psrs[i].ndim+1:].copy()
        
        indo += psrs[i].nobs
        indp += psrs[i].ndim+1
        indg += psrs[i].nobs - psrs[i].ndim - 1

        # Create the noise matrix, and invert the GNG combination with Cholesky
        pNoise = np.diag((psrs[i].toaerrs*1.0e-6)**2)
        GNG = np.dot(U[:,psrs[i].ndim+1:].copy().T, np.dot(pNoise, U[:,psrs[i].ndim+1:].copy()))
        cf = sl.cho_factor(GNG)
        GNGinv.append(sl.cho_solve(cf, np.identity(GNG.shape[0])))
        
        
    hdmat = hdcormat(psrs)
    psrobs = [psrs[i].nobs for i in range(len(psrs))]
    psrpars = [psrs[i].ndim+1 for i in range(len(psrs))]
    psrg = [psrs[i].nobs - psrs[i].ndim - 1 for i in range(len(psrs))]
    
    return (toas, residuals, toaerrs, designmatrix, Gmatrix, hdmat, psrobs, psrpars, psrg, GNGinv, ptheta, pphi)


# This function returns a model for only a subset of pulsars
def modelSubset(fullmodel, subset):
    """
    With a subset of indices (e.g. [3, 5, 6]), this function converts a large
    model into a smaller model with fewer pulsars.

    @param fullmodel: the full model list
    @param subset: a list with the indices of the subset
    """
    nspsrs = len(subset)
    
    psrobs = fullmodel[6]
    psrpars = fullmodel[7]
    psrg = fullmodel[8]
    GNGinv = fullmodel[9]
    ptheta = fullmodel[10]
    pphi = fullmodel[11]
    
    spsrobs = [psrobs[i] for i in subset]
    spspars = [psrpars[i] for i in subset]
    spsrg = [psrg[i] for i in subset]
    sGNGinv = [GNGinv[i] for i in subset]
    sptheta = [ptheta[i] for i in subset]
    spphi = [pphi[i] for i in subset]

    shdmat = fullmodel[5][subset][:,subset]
    
    sobs = np.sum(spsrobs)
    spars = np.sum(spspars)
    sgs = np.sum(spsrg)
    
    subtoas = np.zeros(sobs)
    subresiduals = np.zeros(sobs)
    subtoaerrs = np.zeros(sobs)
    subdesignmatrix = np.zeros((sobs, spars))
    subGmatrix = np.zeros((sobs, sgs))
    
    indo, indp, indg = 0, 0, 0
    csobs = np.append([0], np.cumsum(psrobs))
    cspars = np.append([0], np.cumsum(psrpars))
    csgs = np.append([0], np.cumsum(psrg))

    for i in subset:
        subtoas[indo:indo+psrobs[i]] = fullmodel[0][csobs[i]:csobs[i+1]]
        subresiduals[indo:indo+psrobs[i]] = fullmodel[1][csobs[i]:csobs[i+1]]
        subtoaerrs[indo:indo+psrobs[i]] = fullmodel[2][csobs[i]:csobs[i+1]]
        
        subdesignmatrix[indo:indo+psrobs[i], indp:indp+psrpars[i]] = fullmodel[3][csobs[i]:csobs[i+1], cspars[i]:cspars[i+1]]
        subGmatrix[indo:indo+psrobs[i], indg:indg+psrg[i]] = fullmodel[4][csobs[i]:csobs[i+1], csgs[i]:csgs[i+1]]
        
        indo += psrobs[i]
        indp += psrpars[i]
        indg += psrg[i]
                   
    return (subtoas, subresiduals, subtoaerrs, subdesignmatrix, subGmatrix, shdmat, spsrobs, spspars, spsrg, sGNGinv, sptheta, spphi)




# Calculate the PTA covariance matrix (only GWB)
def Cgw_sec(model, alpha=-2.0/3.0, fL=1.0/500, approx_ksum=False, inc_cor=True):
    """ Compute the residual covariance matrix for an hc = 1 x (f year)^alpha GW background.
        Result is in units of (100 ns)^2.
        Modified from Michele Vallisneri's mc3pta (https://github.com/vallis/mc3pta)

        @param: list of libstempo pulsar objects
                (as returned by readRealisations)
        @param: the H&D correlation matrix
        @param: the TOAs
        @param: the GWB spectral index
        @param: the low-frequency cut-off
        @param: approx_ksum
    """
    psrobs = model[6]
    alphaab = model[5]
    times_f = model[0]
    
    day    = 86400.0              # seconds, sidereal (?)
    year   = 3.15581498e7         # seconds, sidereal (?)

    EulerGamma = 0.5772156649015329

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

    if inc_cor:
        # multiply by alphaab; there must be a more numpythonic way to do it
        # npsrs psrobs
        inda, indb = 0, 0
        for a in range(npsrs):
            for b in range(npsrs):
                corr[inda:inda+psrobs[a], indb:indb+psrobs[b]] *= alphaab[a, b]
                indb += psrobs[b]
            indb = 0
            inda += psrobs[a]
        
    return corr


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


# A likelihood function     return (toas, residuals, toaerrs, designmatrix, Gmatrix, hdmat, psrobs, psrpars, psrg)
def logLikelihood(model, h_c=5e-14, alpha=-2.0/3.0):
    # Obtain model information
    residuals = model[1]
    alphaab = model[5]
    times_f = model[0]
    gmat = model[4]
    psrobs = model[6]
    psrg = model[8]
    toaerrs = model[2]
    
    # Calculate the GW covariance matrix
    C = (h_c*h_c)*Cgw_sec(model, alpha=alpha, fL=1.0/500, approx_ksum=False)
    
    # Add the error bars
    C += np.diag(toaerrs*toaerrs)

    GCG = blockmul(C, gmat, psrobs, psrg)
    resid = np.dot(gmat.T, residuals)

    try:
        cf = sl.cho_factor(GCG)
        res = -0.5 * np.dot(resid, sl.cho_solve(cf, resid)) - 0.5 * len(resid) * np.log((2*np.pi)) - 0.5 * np.sum(np.log(np.diag(cf[0])**2))
    except np.linalg.LinAlgError:
        print "Problem inverting matrix at A = %s, alpha = %s:" % (A,alpha)

        raise

    return res


# For a single pulsar, obtain the estimated GWB amplitude by RMS arguments
def estimateGWRMS(psr):
    # Formula by van Haasteren & Levin (2013, equation 24)
    # sigma_gwb = 1.37e-9 * (Ah / 1e-15) * (T / yr) ^ (5/3)
    day    = 86400.0              # seconds, sidereal (?)
    year   = 3.15581498e7         # seconds, sidereal (?)
    
    gwbvar = np.absolute(np.var(psr.residuals())-psr.toaerrs[0]*psr.toaerrs[0]*1e-12)
    gwbstd = np.sqrt(gwbvar)

    Texp = (psr.toas()[-1] - psr.toas()[0]) * day
    return (gwbstd / 1.37e-9) * 1e-15 / ((Texp / year) ** (5.0/3.0))




# Calculate the crosspower between all pulsar pairs
# Use method of Demorest et al. (2012), Eqn. (9)
def crossPower(psrs):
    npsrs = len(psrs)
    angle = np.zeros(npsrs * (npsrs-1) / 2)
    crosspower = np.zeros(npsrs * (npsrs-1) / 2)
    crosspowererr = np.zeros(npsrs * (npsrs-1) / 2)
    hdcoeff = np.zeros(npsrs * (npsrs-1) / 2)

    sys.stdout.write('crossPower: calculating R...')
    sys.stdout.flush()
    
    # The full model for the full GWB covariance matrix
    fullmodel = makeLinearModel(psrs)
    Eye = np.identity(fullmodel[4].shape[1])
    R = blockmul(Eye, fullmodel[4].T, fullmodel[8], fullmodel[6])
    #R = np.dot(fullmodel[4], fullmodel[4].T)

    sys.stdout.write('\rcrossPower: calculating full covariances...')
    sys.stdout.flush()

    Cgwall = Cgw_sec(fullmodel, alpha=-2.0/3.0, fL=1.0/500, approx_ksum=False, inc_cor=False)

    sys.stdout.write('\rcrossPower: calculating matrix product...')
    sys.stdout.flush()

    #Cgw_full = np.dot(R, np.dot(Cgwall, R))
    Cgw_full = blockmul(Cgwall, R, fullmodel[6], fullmodel[6])
    del Cgwall

    csobs = np.append([0], np.cumsum(fullmodel[6]))        # To keep track of indices
    
    # For each pulsar, calculate the GW and total matrices
    Ci = []
    pos = []
    for a in range(npsrs):
        sys.stdout.write('\rcrossPower: calculating covariances pulsar ' + str(a) + '...')
        sys.stdout.flush()

        # Make the model for a single pulsar
        modela = makeLinearModel([psrs[a]])
        
        # For each pulsar, obtain the ML h_c value with Brent's method
        f = lambda x: -logLikelihood(modela, x, -2.0/3.0)
        fbounded = minimize_scalar(f, bounds=(0, estimateGWRMS(psrs[a]), 3.0e-13), method='Golden')
        hc_ml = fbounded.x
        
        Cml = (hc_ml * hc_ml) * Cgw_sec(modela, alpha=-2.0/3.0, fL=1.0/500, approx_ksum=False, inc_cor=False) + np.diag(modela[2]*modela[2])
        Ctot = blockmul(Cml , modela[4], modela[6], modela[8])
        
        # Invert the matrix with Cholesky
        cf = sl.cho_factor(Ctot)
        Cinv = sl.cho_solve(cf, np.identity(Ctot.shape[0]))
        
        Ci.append(np.dot(modela[4], np.dot(Cinv, modela[4].T)))
        pos.append(np.array([np.cos(psrs[a]['DECJ'].val)*np.cos(psrs[a]['RAJ'].val),
                             np.cos(psrs[a]['DECJ'].val)*np.sin(psrs[a]['RAJ'].val),
                             np.sin(psrs[a]['DECJ'].val)]))

    sys.stdout.write('\rcrossPower: calculating crosspower...')
    sys.stdout.flush()

    ind = 0
    for a in range(npsrs):
        for b in range(a+1, npsrs):
            # The GWB-pair correlation
            Cgw_ab = Cgw_full[csobs[a]:csobs[a+1], csobs[b]:csobs[b+1]]

            # Numerator
            num = np.dot(fullmodel[1][csobs[a]:csobs[a+1]], np.dot(Ci[a], np.dot(Cgw_ab, np.dot(Ci[b], fullmodel[1][csobs[b]:csobs[b+1]]))))
            
            # Demoninator
            den = np.trace(np.dot(Ci[a], np.dot(Cgw_ab, np.dot(Ci[b], Cgw_ab.T))))
    
            # Calculate the angular separation between the two pulsars and the H&D coeff
            angle[ind] = np.arccos(np.dot(pos[a], pos[b]))
            xp = 0.5 * (1 - np.dot(pos[a], pos[b]))
            logxp = 1.5 * xp * np.log(xp)
            hdcoeff[ind] = logxp - 0.25 * xp + 0.5
            
            # Crosspower
            crosspower[ind] = num / den
            
            # Crosspower uncertainty
            crosspowererr[ind] = 1.0 / np.sqrt(den)
            
            
            
            ind += 1

    sys.stdout.write('\rcrossPower: done...\n')
    sys.stdout.flush()

    ind = np.argsort(angle)
            
    return (angle[ind], hdcoeff[ind], crosspower[ind], crosspowererr[ind])


# Calculate the crosspower between all pulsar pairs
# Use method of Demorest et al. (2012), Eqn. (9)
# Do not fit for noise: use the pre-made noise matrices by TempoNest
def crossPowerPrep(fullmodel):
    residuals = fullmodel[1]
    Gmatrix = fullmodel[4]
    psrobs = fullmodel[6]
    psrg = fullmodel[8]
    GNGinv = fullmodel[9]
    ptheta = fullmodel[10]
    pphi = fullmodel[11]

    npsrs = len(psrobs)
    csobs = np.append([0], np.cumsum(psrobs))
    csgs = np.append([0], np.cumsum(psrg))

    angle = np.zeros(npsrs * (npsrs-1) / 2)
    crosspower = np.zeros(npsrs * (npsrs-1) / 2)
    crosspowererr = np.zeros(npsrs * (npsrs-1) / 2)
    hdcoeff = np.zeros(npsrs * (npsrs-1) / 2)

    sys.stdout.write('crossPower: calculating R...')
    sys.stdout.flush()
    
    # The full model for the full GWB covariance matrix
    #fullmodel = makeLinearModel(psrs)
    Eye = np.identity(fullmodel[4].shape[1])
    #R = np.dot(fullmodel[4], fullmodel[4].T)
    R = blockmul(Eye, fullmodel[4].T, psrg, psrobs)

    # This function take a while: give some feedback on how we are doing
    sys.stdout.write('\rcrossPower: calculating full covariances...')
    sys.stdout.flush()

    Cgwall = Cgw_sec(fullmodel, alpha=-2.0/3.0, fL=1.0/500, approx_ksum=False, inc_cor=False)

    sys.stdout.write('\rcrossPower: calculating full matrix product...')
    sys.stdout.flush()

    #Cgw_full = np.dot(R, np.dot(Cgwall, R))
    Cgw_full = blockmul(Cgwall, R, psrobs, psrobs)
    del Cgwall

    csobs = np.append([0], np.cumsum(fullmodel[6]))        # To keep track of indices
    
    # For each pulsar, calculate the GW and total matrices
    Ci = []
    pos = []
    for ii in range(npsrs):
        sys.stdout.write('\rcrossPower: calculating covariances pulsar ' + str(ii) + '...')
        sys.stdout.flush()

        pGmatrix = Gmatrix[csobs[ii]:csobs[ii+1], csgs[ii]:csgs[ii+1]]
        pGNGinv = GNGinv[ii]

        Ci.append(np.dot(pGmatrix, np.dot(pGNGinv, pGmatrix.T)))

        pos.append(np.array([np.sin(ptheta[ii])*np.cos(pphi[ii]),
                             np.sin(ptheta[ii])*np.sin(pphi[ii]),
                             np.cos(ptheta[ii])]))

    sys.stdout.write('\rcrossPower: calculating crosspower...')
    sys.stdout.flush()

    ind = 0
    for a in range(npsrs):
        for b in range(a+1, npsrs):
            # The GWB-pair correlation
            Cgw_ab = Cgw_full[csobs[a]:csobs[a+1], csobs[b]:csobs[b+1]]

            # Numerator
            num = np.dot(fullmodel[1][csobs[a]:csobs[a+1]], np.dot(Ci[a], np.dot(Cgw_ab, np.dot(Ci[b], fullmodel[1][csobs[b]:csobs[b+1]]))))
            
            # Demoninator
            den = np.trace(np.dot(Ci[a], np.dot(Cgw_ab, np.dot(Ci[b], Cgw_ab.T))))
    
            # Calculate the angular separation between the two pulsars and the H&D coeff
            angle[ind] = np.arccos(np.dot(pos[a], pos[b]))
            xp = 0.5 * (1 - np.dot(pos[a], pos[b]))
            logxp = 1.5 * xp * np.log(xp)
            hdcoeff[ind] = logxp - 0.25 * xp + 0.5
            
            # Crosspower
            crosspower[ind] = num / den
            
            # Crosspower uncertainty
            crosspowererr[ind] = 1.0 / np.sqrt(den)
            
            ind += 1

    sys.stdout.write('\rcrossPower: done...\n')
    sys.stdout.flush()

    ind = np.argsort(angle)
            
    return (angle[ind], hdcoeff[ind], crosspower[ind], crosspowererr[ind])




def createAntennaPatternFuncs(gwtheta,gwphi,ptheta,pphi):
    """ Creates Antenna Pattern Functions from Ellis et al 2012,2013 and
        return F+, Fx, and cosMu

        @param gwtheta: Polar angle of GW source in celestial coords [radians]
        @param gwphi: Azimuthal angle of GW source in celestial coords [radians]
        @param ptheta: Polar angle of pulsar in celestial coords [radians]
        @param pphi: Azimuthal angle of pulsar in celestial coords [radians]

    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = [-np.sin(gwphi), np.cos(gwphi), 0.0]
    n = [-np.cos(gwtheta)*np.cos(gwphi), -np.cos(gwtheta)*np.sin(gwphi), np.sin(gwtheta)]
    omhat = [-np.sin(gwtheta)*np.cos(gwphi), -np.sin(gwtheta)*np.sin(gwphi), -np.cos(gwtheta)]
    
    # vector pointing from earth to pusar
    phat = [np.sin(ptheta)*np.cos(pphi), np.sin(ptheta)*np.sin(pphi), np.cos(ptheta)]

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat) * np.dot(n, phat)) / (1+np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    return fplus,fcross,cosMu

# <codecell>

# function to create pulsar timing residuals (Ellis et al. 2012, 2013)
def createResiduals(gwtheta, gwphi, mc, dist, fgw, phase0, psi, inc, \
                    pdist, toas, fplus, fcross, cosMu, psrTerm=True):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013.

    @param gwtheta: Polar angle of GW source in celestial coords [radians]
    @param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    @param mc: Chirp mass of SMBMB [solar masses]
    @param dist: Luminosity distance to SMBMB [Mpc]
    @param fgw: Frequency of GW (twice the orbital frequency) [Hz]
    @param phase0: Initial Phase of GW source [radians]
    @param psi: Polarization of GW source [radians]
    @param inc: Inclination of GW source [radians]
    @param pdist: Distance to the pulsar [kpc]
    @param toas: Times at which to produce the incuced residuals [s]
    @param fplus: Plus polarization antenna pattern function 
    @param fcross: Cross polarization antenna pattern function 
    @param cosMu: Cosine of the angle between the pulsar and the GW source
    @param psrTerm: Option to include pulsar term [boolean] 

    """
    
    # convert units
    mc *= 4.9e-6         # convert from solar masses to seconds
    dist *= 1.0267e14    # convert from Mpc to seconds
    pdist *= 1.0267e11   # convert from kpc to seconds
    
    # get pulsar time
    tp = toas-pdist*(1-cosMu)

    # calculate time dependent frequency at earth and pulsar
    fdot = (96/5) * np.pi**(8/3) * mc**(5/3) * (fgw)**(11/3)
    omega = 2*np.pi*fgw*(1-8/3*fdot/fgw*toas)**(-3/8)
    omega_p = 2*np.pi*fgw*(1-256/5 * mc**(5/3) * np.pi**(8/3) * fgw**(8/3) *tp)**(-3/8)

  
    # calculate time dependent phase
    phase = phase0+ 2*np.pi/(32*np.pi**(8/3)*mc**(5./3.))*\
            (fgw**(-5/3) - (omega/2/np.pi)**(-5/3))
    phase_p = phase0+ 2*np.pi/(32*np.pi**(8/3)*mc**(5./3.))*\
            (fgw**(-5/3) - (omega_p/2/np.pi)**(-5/3))


    # define time dependent coefficients
    At = -0.5*np.sin(phase)*(3+np.cos(2*inc))
    Bt = 2*np.cos(phase)*np.cos(inc)
    At_p = -0.5*np.sin(phase_p)*(3+np.cos(2*inc))
    Bt_p = 2*np.cos(phase_p)*np.cos(inc)

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*(omega/2)**(1./3.))
    alpha_p = mc**(5./3.)/(dist*(omega_p/2)**(1./3.))


    # define rplus and rcross
    rplus = alpha*(At*np.cos(2*psi)-Bt*np.sin(2*psi))
    rcross = alpha*(At*np.sin(2*psi)+Bt*np.sin(2*psi))
    rplus_p = alpha_p*(At_p*np.cos(2*psi)-Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(At_p*np.sin(2*psi)+Bt_p*np.sin(2*psi))

    # residuals
    if psrTerm:
        res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
    else:
        res = -fplus*rplus - fcross*rcross

    return res

def cwWaveform(toas, ptheta, pphi, gwtheta=90.5*np.pi/180, gwphi=13.5*np.pi/180, mc=7e8, dist=100, \
        psi=0.25*np.pi, inc=0, fgw=2e-8, phase=0.28*np.pi, psrTerm=True):
    day = 86400.0       # Seconds per day
    pdist = 1.0         # Pulsar at 1kpc

    fplus, fcross, cosMu = createAntennaPatternFuncs(gwtheta, gwphi, ptheta, pphi)
    return np.array(createResiduals(gwtheta, gwphi, mc, dist, fgw, phase, psi, inc, pdist, toas*day, fplus, fcross, cosMu, psrTerm=psrTerm))



def makeWaveformsFromModel(model, gwtheta=90.5*np.pi/180, gwphi=13.5*np.pi/180, mc=7e8, dist=100, \
        psi=0.25*np.pi, inc=0, fgw=2e-8, phase=0.28*np.pi, psrTerm=True):
    psrobs = model[6]
    toas = model[0]
    ptheta = model[10]
    pphi = model[11]

    # To keep track of indices
    npsrs = len(psrobs)
    csobs = np.append([0], np.cumsum(psrobs))

    waveforms = []

    for ii in range(npsrs):
        waveforms.append(cwWaveform(toas[csobs[ii]:csobs[ii+1]], ptheta[ii], pphi[ii], gwtheta, gwphi, mc, dist, psi, inc, fgw, phase, psrTerm))

    return waveforms


def makeWaveforms(psrs, gwtheta=90.5*np.pi/180, gwphi=13.5*np.pi/180, mc=7e8, dist=100, \
        psi=0.25*np.pi, inc=0, fgw=2e-8, phase=0.28*np.pi, psrTerm=True):
    day = 86400.0

    waveforms = []
    for psr in psrs:
        ptheta = 0.5*np.pi - psr['DECJ'].val
        pphi = psr['RAJ'].val

        fplus, fcross, cosMu = createAntennaPatternFuncs(gwtheta, gwphi, ptheta, pphi)
        waveforms.append(cwWaveform(psr.toas(), ptheta, pphi, gwtheta, gwphi, mc, dist, psi, inc, fgw, phase, psrTerm))

    return waveforms

# A likelihood function for continuous waves
def logLikelihood2(psrs, gwtheta=90.5*np.pi/180, gwphi=13.5*np.pi/180, mc=7e8, dist=100, \
        psi=0.25*np.pi, inc=0, fgw=2e-8, phase=0.28*np.pi):

    waveforms = makeWaveforms(psrs, gwtheta, gwphi, mc, dist, psi, inc, fgw, phase)

    return  -0.5 * np.sum([((psrs[i].residuals()-waveforms[i])/(psrs[i].toaerrs*1e-6))**2 for i in range(len(psrs))])

# A likelihood function for continuous waves including the timing model
def logLikelihoodCW(model, gwtheta=90.5*np.pi/180, gwphi=13.5*np.pi/180, mc=7e8, dist=100, psi=0.25*np.pi, inc=0, fgw=2e-8, phase=0.28*np.pi, psrTerm=True):
    waveforms = makeWaveformsFromModel(model, gwtheta, gwphi, mc, dist, psi, inc, fgw, phase, psrTerm)

    residuals = model[1]
    Gmatrix = model[4]
    psrobs = model[6]
    psrg = model[8]
    GNGinv = model[9]

    # To keep track of indices
    npsrs = len(psrobs)
    csobs = np.append([0], np.cumsum(psrobs))
    csgs = np.append([0], np.cumsum(psrg))

    # Calculate the likelihood for each pulsar
    ploglik = np.zeros(npsrs)

    for ii in range(npsrs):
        pGmatrix = Gmatrix[csobs[ii]:csobs[ii+1], csgs[ii]:csgs[ii+1]]
        pGNGinv = GNGinv[ii]
        predresiduals = np.dot(pGmatrix.T, residuals[csobs[ii]:csobs[ii+1]] - waveforms[ii])

        ploglik[ii] = -0.5 * np.dot(predresiduals, np.dot(pGNGinv, predresiduals))

    return np.sum(ploglik)
