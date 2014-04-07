from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import h5py as h5
import matplotlib.pyplot as plt
import os as os
import glob
import sys
import json
import tempfile


try:
    import statsmodels.api as smapi
    sm = smapi
except ImportError:
    sm = None





"""
Given a collection of samples, return the 2-sigma confidence intervals
samples: an array of samples
sigmalevel: either 1, 2, or 3. Which sigma limit must be given
onesided: Give one-sided limits (useful for setting upper or lower limits)

"""
def confinterval(samples, sigmalevel=2, onesided=False, weights=None):
  # The probabilities for different sigmas
  sigma = [0.68268949, 0.95449974, 0.99730024, 0.90]

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
    # hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights, density=True)
    hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights)
    x = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])     # This was originally 1.5*, but turns out this is a bug plotting of 'stepstyle' in matplotlib
    y = np.cumsum(hist) / np.sum(hist)

  # Find the intervals
  if(onesided):
    bound = 1 - sigma[sigmalevel-1]
  else:
    bound = 0.5*(1-sigma[sigmalevel-1])

  x2min = x[0]
  for i in range(len(y)):
    if y[i] >= bound:
      x2min = x[i]
      break

  if(onesided):
    bound = sigma[sigmalevel-1]
  else:
    bound = 1 - 0.5 * (1 - sigma[sigmalevel-1])

  x2max = x[-1]
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
Obtain the MCMC chain as a numpy array, and a list of parameter indices

@param chainfile: name of the MCMC file
@param parametersfile: name of the file with the parameter labels
@param mcmctype: what method was used to generate the mcmc chain (auto=autodetect)
                    other options are: 'emcee', 'MultiNest', 'ptmcmc'
@param nolabels: set to true if ok to print without labels
@param incextra:    Whether or not we need to return the stype, pulsar, and
                    pso ML as well

@return: logposterior (1D), loglikelihood (1D), parameter-chain (2D), parameter-labels(1D)
"""
def ReadMCMCFile(chainfile, parametersfile=None, sampler='auto', nolabels=False,
        incextra=False):
    parametersfile = chainfile+'.parameters.txt'
    mnparametersfile = chainfile+'.mnparameters.txt'
    mnparametersfile2 = chainfile+'post_equal_weights.dat.mnparameters.txt'
    ptparametersfile = chainfile+'/ptparameters.txt'
    psofile = chainfile + '/pso.txt'

    if not os.path.exists(psofile):
        psofile = None

    if sampler.lower() == 'auto':
        # Auto-detect the sampler
        if os.path.exists(mnparametersfile2):
            chainfile = chainfile + 'post_equal_weights.dat'
            mnparametersfile = mnparametersfile2

        # Determine the type of sampler we've been using through the parameters
        # file
        if os.path.exists(mnparametersfile):
            sampler = 'MultiNest'
            parametersfile = mnparametersfile
            chainfile = chainfile
            figurefileeps = chainfile+'.fig.eps'
            figurefilepng = chainfile+'.fig.png'
        elif os.path.exists(ptparametersfile):
            sampler = 'PTMCMC'
            parametersfile = ptparametersfile
            if os.path.exists(chainfile+'/chain_1.0.txt'):
                figurefileeps = chainfile+'/chain_1.0.fig.eps'
                figurefilepng = chainfile+'chain_1.0.fig.png'
                chainfile = chainfile+'/chain_1.0.txt'
            elif os.path.exists(chainfile+'/chain_1.txt'):
                figurefileeps = chainfile+'/chain_1.fig.eps'
                figurefilepng = chainfile+'chain_1.fig.png'
                chainfile = chainfile+'/chain_1.txt'
            else:
                raise IOError, "No valid chain found for PTMCMC_Generic"
        elif os.path.exists(parametersfile):
            sampler = 'emcee'
            chainfile = chainfile
            figurefileeps = chainfile+'.fig.eps'
            figurefilepng = chainfile+'.fig.png'
        else:
            if not nolabels:
                raise IOError, "No valid parameters file found!"

            else:
                chainfile = chainfile
                figurefileeps = chainfile+'.fig.eps'
                figurefilepng = chainfile+'.fig.png'
                sampler = 'emcee'
    elif sampler.lower() == 'multinest':
        if os.path.exists(mnparametersfile2):
            chainfile = chainfile + 'post_equal_weights.dat'
            mnparametersfile = mnparametersfile2

        parametersfile = mnparametersfile
        figurefileeps = chainfile+'.fig.eps'
        figurefilepng = chainfile+'.fig.png'
    elif sampler.lower() == 'emcee':
        figurefileeps = chainfile+'.fig.eps'
        figurefilepng = chainfile+'.fig.png'
    elif sampler.lower() == 'ptmcmc':
        parametersfile = ptparametersfile
        if os.path.exists(chainfile+'/chain_1.0.txt'):
            figurefileeps = chainfile+'/chain_1.0.fig.eps'
            figurefilepng = chainfile+'chain_1.0.fig.png'
            chainfile = chainfile+'/chain_1.0.txt'
        elif os.path.exists(chainfile+'/chain_1.txt'):
            figurefileeps = chainfile+'/chain_1.fig.eps'
            figurefilepng = chainfile+'chain_1.fig.png'
            chainfile = chainfile+'/chain_1.txt'

    if not nolabels:
        # Read the parameter labels
        if os.path.exists(parametersfile):
            parfile = open(parametersfile)
            lines=[line.strip() for line in parfile]
            parlabels=[]
            stypes=[]
            pulsars=[]
            pulsarnames=[]
            for i in range(len(lines)):
                lines[i]=lines[i].split()

                if int(lines[i][0]) >= 0:
                    # If the parameter has an index
                    parlabels.append(lines[i][5])
                    stypes.append(lines[i][2])
                    pulsars.append(int(lines[i][1]))

                    if len(lines[i]) > 6:
                        pulsarnames.append(lines[i][6])
                    else:
                        pulsarnames.append("Pulsar {0}".format(lines[i][1]))

            parfile.close()
        else:
            raise IOError, "No valid parameters file found!"
    else:
        parlabels = None
        stypes = []
        pulsars = []

    if os.path.exists(parametersfile):
        chain = np.loadtxt(chainfile)
    else:
        raise IOError, "No valid chain-file found!"

    if psofile is not None:
        mldat = np.loadtxt(psofile)
        mlpso = mldat[0]
        mlpsopars = mldat[1:]
    else:
        mlpso = None
        mlpsopars = None

    if sampler.lower() == 'emcee':
        logpost = chain[:,1]
        loglik = None
        samples = chain[:,2:]
    elif sampler.lower() == 'multinest':
        loglik = chain[:,-1]
        logpost = None
        samples = chain[:,:-1]
    elif sampler.lower() == 'ptmcmc':
        logpost = chain[:,0]
        loglik = chain[:,1]
        samples = chain[:,3:]

    if incextra:
        retvals = (logpost, loglik, samples, parlabels, \
                pulsars, pulsarnames, stypes, mlpso, mlpsopars)
    else:
        retvals = (logpost, loglik, samples, parlabels)

    return retvals


def eptaOutput(chainfile, outputdir, burnin=0, thin=1, \
        parametersfile=None, sampler='auto', make1dplots=True, \
        maxpages=-1):
    """
    Given an MCMC chain file, and an output directory, produce an output file
    with all the results EPTA-style

    @param chainfile:   name of the MCMC file
    @param outputdir:   output directory where all the plots will be saved
    @param burnin:      Number of steps to be considered burn-in
    @param thin:        Number of steps to skip in between samples (thinning)
    @param parametersfile:  name of the file with the parameter labels
    @param sampler:     What method was used to generate the mcmc chain
                        (auto=autodetect). Options:('emcee', 'MultiNest',
                        'ptmcmc')

    EPTA-style format:

    [1]        [2]      [3]        [4]    [5]    [6]      [7]     [8]         [9]             [10]   [11]

    pulsarname  parname1 parname2     maxL  low68%  up68%   low90%   up90%   Priortype(lin/log)   minP   maxP

    J****         DM      amp        4E5    1E-14  1E-13   1E-15    1E-12        log             1E-20  1E-10  
    J****         DM     slope       -3E0   -4E0   -2E0    -5E0     -1E0         lin             -1E-10 -1E1
    J****         RN      amp
    J****         RN     slope
    [J****         SingS   amp
    J****         SingS  freq
    J****         SingS  phase]<----those three lines for model2
    J****        EFAC    backname1
    J****        EFAC    backname2
    J****        EFAC    backname3
    J****        EFAC    backname4
      ....
    J****        EQUAD   backname1
    J****        EQUAD   backname2
    J****        EQUAD   backname3
    J****        EQUAD   backname4
      ....

    """
    # Read the mcmc chain
    (llf, lpf, chainf, labels, pulsarid, pulsarname, stype, mlpso, mlpsopars) = \
            ReadMCMCFile(chainfile, parametersfile=parametersfile, \
            sampler=sampler, incextra=True)

    # Remove burn-in and thin the chain
    ll = llf[burnin::thin]
    lp = lpf[burnin::thin]
    chain = chainf[burnin::thin, :]

    # Obtain the maximum from the chain
    mlind = np.argmax(lp)
    mlchain = lp[mlind]
    mlchainpars = chain[mlind, :]

    if mlpso is None:
        ml = mlchain
        mlpars = None
        mlpars2 = mlchainpars
    else:
        ml = mlpso
        mlpars = mlpsopars
        mlpars2 = mlchainpars

    # List all varying parameters
    dopar = np.array([1]*len(labels), dtype=np.bool)

    table = []
    for ll, label in enumerate(labels):
        fields = np.empty(11, dtype='a64')
        fields[0] = pulsarname[ll]
        fields[3] = '0.0'

        if stype[ll] == 'efac':
            fields[1] = 'EFAC'
            fields[8] = 'lin'
            fields[9] = '0.001'
            fields[10] = '50.0'

            lab = labels[ll][15:]
            if len(lab) > 0:
                fields[2] = labels[ll][15:]
            else:
                fields[2] = pulsarname[ll]

        elif stype[ll] == 'equad':
            fields[1] = 'EQUAD'
            fields[8] = 'log'
            fields[9] = '-10.0'
            fields[10] = '-4.0'

            lab = labels[ll][16:]
            if len(lab) > 0:
                fields[2] = labels[ll][16:]
            else:
                fields[2] = pulsarname[ll]

        elif stype[ll] == 'dmpowerlaw' and \
                labels[ll] == 'DM-Amplitude':
            fields[1] = 'DM'
            fields[2] = 'amp'
            fields[8] = 'log'
            fields[9] = '-14.0'
            fields[10] = '-6.5'
        elif stype[ll] == 'dmpowerlaw' and \
                labels[ll] == 'DM-spectral-index':
            fields[1] = 'DM'
            fields[2] = 'slope'
            fields[8] = 'lin'
            fields[9] = '0.02'
            fields[10] = '6.98'
        elif stype[ll] == 'powerlaw' and \
                labels[ll] == 'RN-Amplitude':
            fields[1] = 'RN'
            fields[2] = 'amp'
            fields[8] = 'log'
            fields[9] = '-20.0'
            fields[10] = '-10.0'
        elif stype[ll] == 'powerlaw' and \
                labels[ll] == 'RN-spectral-index':
            fields[1] = 'RN'
            fields[2] = 'slope'
            fields[8] = 'lin'
            fields[9] = '0.02'
            fields[10] = '6.98'
        else:
            continue

        fmin, fmax = confinterval(chain[:, ll], sigmalevel=1)
        fields[4] = str(fmin)
        fields[5] = str(fmax)

        fmin, fmax = confinterval(chain[:, ll], sigmalevel=4)
        fields[6] = str(fmin)
        fields[7] = str(fmin)

        table.append(fields)

    table = np.array(table)

    # Sort all the fields
    newtable = np.empty(table.shape, dtype='a64')
    newind = 0

    # Place a new sorted version of all pulsars in the new table
    psrs = set(table[:,0])
    for psr in psrs:
        psr_rows = (table[:,0] == psr)
        newpsr = table[psr_rows,:].copy()

        # First do the DM
        dm_rows = (newpsr[:,1] == 'DM')
        numpars = np.sum(dm_rows)
        newtable[newind:newind+numpars,:] = newpsr[dm_rows,:]
        newind += numpars

        # Next, the red noise
        rn_rows = (newpsr[:,1] == 'RN')
        numpars = np.sum(rn_rows)
        newtable[newind:newind+numpars,:] = newpsr[rn_rows,:]
        newind += numpars

        # Then, the EFAC (sorted)
        efac_rows = (newpsr[:,1] == 'EFAC')
        efacpars = newpsr[efac_rows,:].copy()
        numpars = np.sum(efac_rows)
        efac_ind_srt = np.argsort(efacpars[:,2])
        newtable[newind:newind+numpars,:] = efacpars[efac_ind_srt,:]
        newind += numpars

        # Finally, the EQUAD (sorted)
        efac_rows = newpsr[:,1] == 'EQUAD'
        efacpars = newpsr[efac_rows,:].copy()
        numpars = np.sum(efac_rows)
        efac_ind_srt = np.argsort(efacpars[:,2])
        newtable[newind:newind+numpars,:] = efacpars[efac_ind_srt,:]
        newind += numpars

    eptafilename = outputdir + '/eptafile.txt'
    eptafile = open(eptafilename, 'w')

    for ii in range(newtable.shape[0]):
        eptafile.write('\t'.join(['{0}'.format(newtable[ii,jj]) \
                for jj in range(newtable.shape[1])]))
        eptafile.write('\n')

    eptafile.close()
    

    return newtable
