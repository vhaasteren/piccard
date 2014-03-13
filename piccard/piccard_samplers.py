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

from .piccard import *
from . import pytwalk                  # Internal module
from . import pydnest                  # Internal module
from . import rjmcmchammer as rjemcee  # Internal module
from . import PTMCMC_generic as ptmcmc
from .triplot import *

try:
    import statsmodels.api as smapi
    sm = smapi
except ImportError:
    sm = None

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
    # hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights, density=True)
    hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights)
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
def makechainplot2d(chain, par1=72, par2=73, xmin=None, xmax=None, ymin=None, ymax=None, title=r"GWB credible regions"):
  if xmin is None:
    #xmin = 0
    xmin = min(chain[:,par1])
  if xmax is None:
    #xmax = 70
    xmax = max(chain[:,par1])
  if ymin is None:
    #ymin = 1
    ymin = min(chain[:,par2])
  if ymax is None:
    #ymax = 7
    ymax = max(chain[:,par2])

  # Process the parameters

  make2dplot(chain[:,par1], chain[:,par2], title=title, \
	  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

"""
Given an mcmc chain file, plot the credible region for the GWB

"""
def makechainplot1d(chain, par=72, xmin=None, xmax=None, title=r"GWB marginalised posterior"):
  if xmin is None:
    xmin = 0
  if xmax is None:
    xmax = 70

  plt.figure()
  plt.hist(chain[:, par], 100, color="k", histtype="step", range=(xmin, xmax))
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
def makemnplots(mnchain, par1=72, par2=73, xmin=0, xmax=70, ymin=1, ymax=7, title='MultiNest credible regions'):
  nDimensions = mnchain.shape[1]

  # The list of 1D parameters we'd like to check:
  list1d = np.array([par1, par2])

  # Create 1d histograms
  for i in list1d:
    plt.figure()
    plt.hist(mnchain[:,i], 100, color="k", histtype="step")
    plt.title("Dimension {0:d}".format(i))

  make2dplot(mnchain[:,par1], mnchain[:,par2], title=title)



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
Given an mcmc chain, plot the log-spectrum

"""
def makespectrumplot(chain, parstart=1, numfreqs=10, freqs=None, \
        Apl=None, gpl=None, Asm=None, asm=None, fcsm=0.1, plotlog=False, \
        lcolor='black', Tmax=None, Aref=None):
    if freqs is None:
        ufreqs = np.log10(np.arange(1, 1+numfreqs))
    else:
        ufreqs = np.log10(np.sort(np.array(list(set(freqs)))))

    #ufreqs = np.array(list(set(freqs)))
    yval = np.zeros(len(ufreqs))
    yerr = np.zeros(len(ufreqs))

    if len(ufreqs) != (numfreqs):
        print "WARNING: parameter range does not correspond to #frequencies"

    for ii in range(numfreqs):
        fmin, fmax = confinterval(chain[:, parstart+ii], sigmalevel=1)
        yval[ii] = (fmax + fmin) * 0.5
        yerr[ii] = (fmax - fmin) * 0.5

    fig = plt.figure()

    # For plotting reference spectra
    pfreqs = 10 ** ufreqs
    ypl = None
    ysm = None

    if plotlog:
        plt.errorbar(ufreqs, yval, yerr=yerr, fmt='.', c=lcolor)
        # outmatrix = np.array([ufreqs, yval, yerr]).T
        # np.savetxt('spectrumplot.txt', outmatrix)

        if Apl is not None and gpl is not None and Tmax is not None:
            Apl = 10**Apl
            ypl = (Apl**2 * pic_spy**3 / (12*np.pi*np.pi * (Tmax))) * ((pfreqs * pic_spy) ** (-gpl))
            plt.plot(np.log10(pfreqs), np.log10(ypl), 'g--', linewidth=2.0)

        if Asm is not None and asm is not None and Tmax is not None:
            Asm = 10**Asm
            fcsm = fcsm / pic_spy
            ysm = (Asm * pic_spy**3 / Tmax) * ((1 + (pfreqs/fcsm)**2)**(-0.5*asm))
            plt.plot(np.log10(pfreqs), np.log10(ysm), 'r--', linewidth=2.0)


        #plt.axis([np.min(ufreqs)-0.1, np.max(ufreqs)+0.1, np.min(yval-yerr)-1, np.max(yval+yerr)+1])
        plt.xlabel("Frequency [log(f/Hz)]")
        #if True:
        #    #freqs = likobhy.ptapsrs[0].Ffreqs
        #    Tmax = 156038571.88061461
        #    Apl = 10**-13.3 ; Asm = 10**-24
        #    apl = 4.33 ; asm = 4.33
        #    fc = (10**-1.0)/pic_spy

        #    pcsm = (Asm * pic_spy**3 / Tmax) * ((1 + (freqs/fc)**2)**(-0.5*asm))
        #    pcpl = (Apl**2 * pic_spy**3 / (12*np.pi*np.pi * Tmax)) * \
        #    (freqs*pic_spy) ** (-apl)
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
Given an mcmc chain file, plot the upper limit of one variable as a function of
another

"""
def upperlimitplot2d(chain, par1=72, par2=73, ymin=None, ymax=None):
  if ymin is None:
    #ymin = 1
    ymin = min(chain[:,par2])
  if ymax is None:
    #ymax = 7
    ymax = max(chain[:,par2])

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
      chain[:,par2] > yedges[i],
      chain[:,par2] < yedges[i+1]))

    # Obtain the 1-sided x-sigma upper limit
    a, b = confinterval(chain[:,par1][indices], sigmalevel=1, onesided=True)
    sigma1[i] = np.exp(b)
    a, b = confinterval(chain[:,par1][indices], sigmalevel=2, onesided=True)
    sigma2[i] = np.exp(b)
    a, b = confinterval(chain[:,par1][indices], sigmalevel=3, onesided=True)
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

NOTE: Deprecated
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

NOTE: Do not use, except for emcee nowadays
"""
def makellplot(chainfilename, numfigs=2, emceesort=False):
  emceechain = np.loadtxt(chainfilename)

  if emceesort:
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
  else:
      plt.figure()
      plt.plot(np.arange(emceechain[:,1].size), \
              emceechain[:,1], 'b-')

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


def makeLogLikelihoodPlot(ax, ll, \
        xlabel="Sample number", ylabel="Log-posterior", title="Log-likelihood"):
    """
    Make a plot of the log-likelihood versus sample number
    """
    ax.plot(np.arange(len(ll)), ll, 'b-')
    ax.grid(True)
    ax.set_label(xlabel)
    ax.set_ylabel(ylabel)

def makeEfacPage(fig, samples, labels, mlchain, mlpso, txtfilename, \
        ylabel='EFAC', title='Efac values'):
    """
    Make a 1-D plot of all efacs (or equad/jitter)

    TODO: Add ML estimates
    """

    npars = len(mlchain)

    # Create the plotting data for this plot
    x = np.arange(npars)
    yval = np.zeros(npars)
    yerr = np.zeros(npars)

    #for ii in range(maxpar-minpar):
    for ii in range(npars):
        fmin, fmax = confinterval(samples[:, ii], sigmalevel=1)
        yval[ii] = (fmax + fmin) * 0.5
        yerr[ii] = (fmax - fmin) * 0.5

    ax = fig.add_subplot(111)

    plt.subplots_adjust(left=0.115, right=0.95, top=0.9, bottom=0.25)

    resp = ax.errorbar(x, yval, yerr=yerr, fmt=None, c='blue')
    if mlpso is not None:
        try:
            ress = ax.scatter(x, mlpso, s=50, c='r', marker='*')
        except ValueError:
            ress = ax.scatter(x, mlpso, s=50, c='r', marker='x')
    else:
        try:
            ress = ax.scatter(x, mlchain, s=50, c='r', marker='*')
        except ValueError:
            ress = ax.scatter(x, mlchain, s=50, c='r', marker='+')

    #ax.axis([-1, max(x)+1, 0, max(yval+yerr)+1])
    ax.axis([-1, max(x)+1, min(yval-yerr)-1, max(yval+yerr)+1])
    ax.xaxis.grid(True, which='major')

    ax.set_xticks(np.arange(npars))

    #ax.set_title(r'Efac values, page ' + str(pp))
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    xtickNames = plt.setp(ax, xticklabels=labels)

    plt.setp(xtickNames, rotation=45, fontsize=8, ha='right')

    #fileout = open(outputdir+'/efac-page-'+str(pp)+'.txt', 'w')
    fileout = open(txtfilename, 'w')
    for ii in range(npars):
        print str(labels[ii]) + ":  " + str(yval[ii]) + " +/- " + str(yerr[ii])
        fileout.write(str(labels[ii]) + "  " + str(yval[ii]) + \
                        "  " + str(yerr[ii]) + "\n")
    fileout.close()


def makeSpectrumPage(ax, samples, freqs, mlchain, mlpso, \
        xlabel='Frequency [log10(f/Hz)]', ylabel='PSD', \
        title='Power Spectral Density'):
    """
    Make a 1-D plot of the power spectrum

    TODO: Add ML estimates
    """

    npars = len(mlchain)

    # Create the plotting data for this plot
    x = np.arange(npars)
    yval = np.zeros(npars)
    yerr = np.zeros(npars)

    #for ii in range(maxpar-minpar):
    for ii in range(npars):
        fmin, fmax = confinterval(samples[:, ii], sigmalevel=1)
        yval[ii] = (fmax + fmin) * 0.5
        yerr[ii] = (fmax - fmin) * 0.5

    resp = ax.errorbar(x, yval, yerr=yerr, fmt='.', c='blue')

    ax.axis([min(x)-0.2, max(x)+0.2, min(yval-yerr)-1, max(yval+yerr)+1])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(True)


def makeResidualsPlot(ax, toas, residuals, toaerrs, flags, \
        xlabel=r'TOA [MJD]', \
        ylabel=r'Residual [$\mu$s]', \
        title='Timing residuals'):
    """
    Plot the timing residuals
    """
    becolours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    backends = list(set(flags))

    for bb, backend in enumerate(backends):
        toaind = np.array(np.array(flags) == backend, dtype=np.bool)

        respl = ax.errorbar(toas[toaind], \
                residuals[toaind]*1e6, \
                yerr=toaerrs[toaind]*1e6, fmt='.', \
                c=becolours[bb % len(becolours)])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)





def makeAllPlots(chainfile, outputdir, burnin=0, thin=1, \
        parametersfile=None, sampler='auto', make1dplots=True):
    """
    Given an MCMC chain file, and an output directory, make all the results
    plots

    @param chainfile:   name of the MCMC file
    @param outputdir:   output directory where all the plots will be saved
    @param burnin:      Number of steps to be considered burn-in
    @param thin:        Number of steps to skip in between samples (thinning)
    @param parametersfile:  name of the file with the parameter labels
    @param sampler:     What method was used to generate the mcmc chain
                        (auto=autodetect). Options:('emcee', 'MultiNest',
                        'ptmcmc')
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

    # Log-likelihood plot
    fileout = outputdir+'/logpost'
    title = "Log-likelihood '{0}'".format(outputdir)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    makeLogLikelihoodPlot(ax, lp, title=title)
    plt.savefig(fileout+'.png')
    plt.savefig(fileout+'.eps')
    plt.close(fig)

    # Plot the residuals for the pulsars
    for infile in glob.glob(os.path.join(outputdir+'/', 'residuals-*.txt') ):
        fileout = infile[:-4]
        residualsfile = open(infile)
        lines=[line.strip() for line in residualsfile]
        toas = []
        residuals = []
        toaerrs = []
        flags = []
        for ii in range(len(lines)):
            line = lines[ii].split()
            toas.append(float(line[0]))
            residuals.append(float(line[1]))
            toaerrs.append(float(line[2]))
            flags.append(line[3])

        toas = np.array(toas)
        residuals = np.array(residuals)
        toaerrs = np.array(toaerrs)
        residualsfile.close()

        title = 'Timing residuals of ' + infile[12:-4]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        makeResidualsPlot(ax, toas, residuals, toaerrs, flags, title=title)

        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')
        plt.close(fig)

    # Plot the efacs on pages
    efacparind = (np.array(stype) == 'efac')
    numefacs = np.sum(efacparind)
    if numefacs > 1 and make1dplots:
        # With more than one efac, we'll make a separate efac page
        dopar[efacparind] = False

        maxplotpars = 20
        pages = int(1 + numefacs / maxplotpars)
        for pp in range(pages):
            minpar = pp * maxplotpars
            maxpar = min(numefacs, minpar + maxplotpars)
            fileout = outputdir+'/efac-page-' + str(pp)

            efacchain = chain[:, efacparind][:, minpar:maxpar]
            efacnames = np.array(labels)[efacparind][minpar:maxpar]
            efacmlchain = mlchainpars[efacparind][minpar:maxpar]
            if mlpsopars is not None:
                efacmlpso = mlpsopars[efacparind][minpar:maxpar]
            else:
                efacmlpso = None

            fig = plt.figure()
            #ax = fig.add_subplot(111)
            makeEfacPage(fig, efacchain, efacnames, efacmlchain, efacmlpso, \
                    fileout + '.txt', 'EFAC', \
                    'Efac values {0}, page {1}'.format(pulsarname[pp], pp))

            plt.savefig(fileout+'.png')
            plt.savefig(fileout+'.eps')
            plt.close(fig)

    # Plot the equads on pages
    equadparind = (np.array(stype) == 'equad')
    numequads = np.sum(equadparind)
    if numequads > 1 and make1dplots:
        # With more than one equad, we'll make a separate equad page
        dopar[equadparind] = False

        maxplotpars = 20
        pages = int(1 + numequads / maxplotpars)
        for pp in range(pages):
            minpar = pp * maxplotpars
            maxpar = min(numequads, minpar + maxplotpars)
            fileout = outputdir+'/equad-page-' + str(pp)

            equadchain = chain[:, equadparind][:, minpar:maxpar]
            equadnames = np.array(labels)[equadparind][minpar:maxpar]
            equadmlchain = mlchainpars[equadparind][minpar:maxpar]
            if mlpsopars is not None:
                equadmlpso = mlpsopars[equadparind][minpar:maxpar]
            else:
                equadmlpso = None

            fig = plt.figure()
            #ax = fig.add_subplot(111)
            makeEfacPage(fig, equadchain, equadnames, equadmlchain, equadmlpso, \
                    fileout + '.txt', 'Equad', \
                    'Equad values {0}, page {1}'.format(pulsarname[pp], pp))

            plt.savefig(fileout+'.png')
            plt.savefig(fileout+'.eps')
            plt.close(fig)

    # Plot the jitters on pages
    jitterparind = (np.array(stype) == 'jitter')
    numjitters = np.sum(jitterparind)
    if numjitters > 1 and make1dplots:
        # With more than one jitter, we'll make a separate jitter page
        dopar[jitterparind] = False

        maxplotpars = 20
        pages = int(1 + numjitters / maxplotpars)
        for pp in range(pages):
            minpar = pp * maxplotpars
            maxpar = min(numjitters, minpar + maxplotpars)
            fileout = outputdir+'/jitter-page-' + str(pp)

            jitterchain = chain[:, jitterparind][:, minpar:maxpar]
            jitternames = np.array(labels)[jitterparind][minpar:maxpar]
            jittermlchain = mlchainpars[jitterparind][minpar:maxpar]
            if mlpsopars is not None:
                jittermlpso = mlpsopars[jitterparind][minpar:maxpar]
            else:
                jittermlpso = None

            fig = plt.figure()
            #ax = fig.add_subplot(111)
            makeEfacPage(fig, jitterchain, jitternames, jittermlchain, jittermlpso, \
                    fileout + '.txt', 'Correlated Equad', \
                    'Correlated Equad values {0}, page {1}'. \
                    format(pulsarname[pp], pp))

            plt.savefig(fileout+'.png')
            plt.savefig(fileout+'.eps')
            plt.close(fig)

    # Plot power-spectra here
    for psr in list(set(pulsarid)):
        for signal in ['spectrum', 'dmspectrum']:
            sigind = (np.array(stype) == signal)
            psrind = (np.array(pulsarid) == psr)
            ind = np.logical_and(sigind, psrind)

            if np.sum(ind) > 0:
                fileout = outputdir+'/'+pulsarname[ind[0]]+'-'+signal

                dopar[jitterparind] = False
                freqs = np.log10(np.float(np.array(labels[ind])))
                spectrumchain = chain[:, ind]
                spectrummlchain = mlchainpars[ind]
                if mlpsopars is not None:
                    spectrummlpso = mlpsopars[ind]
                else:
                    spectrummlpso = None

                if signal == 'spectrum':
                    title = 'Power Spectral Density noise {0}'.format(\
                            pulsarname[ind[0]])
                elif signal == 'dmspectrum':
                    title = 'Power Spectral Density DM variations {0}'.format(\
                            pulsarname[ind[0]])

                fig = plt.figure()
                ax = fig.add_subplot(111)
                makeSpectrumPage(ax, spectrumchain, freqs, spectrummlchain, \
                        spectrummlpso, title=title)



    # Make a triplot of all the other parameters
    if np.sum(dopar) > 1:
        indices = np.flatnonzero(np.array(dopar == True))
        fileout = outputdir+'/triplot'
        triplot(chain, parlabels=labels, plotparameters=indices, ml=mlpars,
                ml2=mlpars2)
        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')
        plt.close(fig)

    if np.sum(dopar) == 1:
        # Make a single plot
        indices = np.flatnonzero(np.array(dopar == True))
        f, axarr = plt.subplots(nrows=1, ncols=1)
        makesubplot1d(axarr, emceechain[:,indices[0]])
        fileout = outputdir+'/triplot'
        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')
        plt.close(fig)




"""
Given a likelihood object, a 'normal' MCMC chain file, and an output directory,
this function spits out a lot of plots summarising all relevant results of the
MCMC

Note: will be deprecated some next version
"""
def makeresultsplot(likob, chainfilename, outputdir, burnin=0, thin=1):
    (logpost_long, loglik_long, emceechain_long, labels) = ReadMCMCFile(chainfilename)
    (logpost, loglik, emceechain) = (logpost_long[burnin::thin], \
            loglik_long[burnin::thin], emceechain_long[burnin::thin])
    if logpost is None:
        lp = loglik
    else:
        lp = logpost

    # Make a ll-plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(lp)), lp, 'b-')
    ax.grid(True)
    ax.set_xlabel('Sample number')
    ax.set_ylabel('Log-posterior')
    fileout = outputdir+'/logpost'
    plt.savefig(fileout+'.png')
    plt.savefig(fileout+'.eps')
    plt.close(fig)

    # List all varying parameters
    dopar = np.array([1]*likob.dimensions, dtype=np.bool)

    # First plot the timing residuals, coloured by backend system, for each pulsar
    becolours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for pp, psr in enumerate(likob.ptapsrs):
        fileout = outputdir+'/residuals-' + psr.name
        fig = plt.figure()
        ax = fig.add_subplot(111)

        backends = list(set(psr.flags))
        for bb, backend in enumerate(backends):
            toaind = np.array(np.array(psr.flags) == backend, dtype=np.bool)
            respl = ax.errorbar(psr.toas[toaind]/pic_spd, \
                    psr.residuals[toaind]*1e6, \
                    yerr=psr.toaerrs[toaind]*1e6, fmt='.', \
                    c=becolours[bb % len(becolours)])

        ax.set_xlabel('TOA [MJD] - 53000')
        ax.set_ylabel(r'Residual [$\mu$s]')
        ax.set_title(psr.name)
        ax.grid(True)

        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')

        plt.close(fig)
        

    # Make a plot of all efac's
    efacparind, efacpsrind, efacnames = likob.getEfacNumbers()

    if len(efacparind) > 0:
        dopar[efacparind] = False
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

            #for ii in range(maxpar-minpar):
            for ii in range(minpar, maxpar):
                fmin, fmax = confinterval(emceechain[:, efacparind[ii]], sigmalevel=1)
                yval[ii-minpar] = (fmax + fmin) * 0.5
                yerr[ii-minpar] = (fmax - fmin) * 0.5

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
            #ax.legend(('One', 'Rutger ML', 'Two', 'Three',), shadow=True, fancybox=True, numpoints=1)
            #ax.set_yscale('log')

            xtickNames = plt.setp(ax, xticklabels=efacnames[minpar:maxpar])
            #plt.getp(xtickNames)
            plt.setp(xtickNames, rotation=45, fontsize=8, ha='right')

            efacfileout = open(outputdir+'/efac-page-'+str(pp)+'.txt', 'w')
            for ii in range(minpar, maxpar):
                print str(efacnames[ii]) + ":  " + str(yval[ii-minpar]) + " +/- " + str(yerr[ii-minpar])
                efacfileout.write(str(efacnames[ii]) + "  " + str(yval[ii-minpar]) + \
                                "  " + str(yerr[ii-minpar]) + "\n")

            efacfileout.close()

            plt.savefig(fileout+'.png')
            plt.savefig(fileout+'.eps')
            plt.close(fig)


    # Make a plot of all equad's
    equadparind, equadpsrind, equadnames = likob.getEfacNumbers('equad')
    dopar[equadparind] = False

    if len(equadparind) > 1:
        maxplotpars = 20
        pages = int(1 + len(equadparind) / maxplotpars)
        for pp in range(pages):
            minpar = pp * maxplotpars
            maxpar = min(len(equadparind), minpar + maxplotpars)
            fileout = outputdir+'/equad-page-' + str(pp)

            # Create the plotting data for this plot
            x = np.arange(maxpar-minpar)
            yval = np.zeros(maxpar-minpar)
            yerr = np.zeros(maxpar-minpar)

            #for ii in range(maxpar-minpar):
            for ii in range(minpar, maxpar):
                fmin, fmax = confinterval(emceechain[:, equadparind[ii]], sigmalevel=1)
                yval[ii-minpar] = (fmax + fmin) * 0.5
                yerr[ii-minpar] = (fmax - fmin) * 0.5

            # Now make the plot
            fig = plt.figure()

            #fig = plt.figure(figsize=(10,6))   # Figure size can be adjusted if it gets big
            ax = fig.add_subplot(111)

            #plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
            plt.subplots_adjust(left=0.115, right=0.95, top=0.9, bottom=0.25)

            resp = ax.errorbar(x, yval, yerr=yerr, fmt='.', c='blue')

            #ax.axis([-1, max(x)+1, 0, max(yval+yerr)+1])
            ax.axis([-1, max(x)+1, min(yval-yerr)-1, max(yval+yerr)+1])
            ax.xaxis.grid(True, which='major')

            #ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
            #              alpha=1.0)

            ax.set_xticks(np.arange(maxpar-minpar))

            ax.set_title(r'Equad values, page ' + str(pp))
            ax.set_ylabel(r'Equad')
            #ax.legend(('One', 'Rutger ML', 'Two', 'Three',), shadow=True, fancybox=True, numpoints=1)
            #ax.set_yscale('log')

            xtickNames = plt.setp(ax, xticklabels=equadnames[minpar:maxpar])
            #plt.getp(xtickNames)
            plt.setp(xtickNames, rotation=45, fontsize=8, ha='right')

            equadfileout = open(outputdir+'/equad-page-'+str(pp)+'.txt', 'w')
            for ii in range(minpar, maxpar):
                print str(equadnames[ii]) + ":  " + str(yval[ii-minpar]) + " +/- " + str(yerr[ii-minpar])
                equadfileout.write(str(equadnames[ii]) + "  " + str(yval[ii-minpar]) + \
                                "  " + str(yerr[ii-minpar]) + "\n")

            equadfileout.close()

            plt.savefig(fileout+'.png')
            plt.savefig(fileout+'.eps')
            plt.close(fig)


    # Make a plot of all cequad's
    cequadparind, cequadpsrind, cequadnames = likob.getEfacNumbers('jitter')
    dopar[cequadparind] = False

    if len(cequadparind) > 1:
        maxplotpars = 20
        pages = int(1 + len(cequadparind) / maxplotpars)
        for pp in range(pages):
            minpar = pp * maxplotpars
            maxpar = min(len(cequadparind), minpar + maxplotpars)
            fileout = outputdir+'/cequad-page-' + str(pp)

            # Create the plotting data for this plot
            x = np.arange(maxpar-minpar)
            yval = np.zeros(maxpar-minpar)
            yerr = np.zeros(maxpar-minpar)

            #for ii in range(maxpar-minpar):
            for ii in range(minpar, maxpar):
                fmin, fmax = confinterval(emceechain[:, cequadparind[ii]], sigmalevel=1)
                yval[ii-minpar] = (fmax + fmin) * 0.5
                yerr[ii-minpar] = (fmax - fmin) * 0.5

            # Now make the plot
            fig = plt.figure()

            #fig = plt.figure(figsize=(10,6))   # Figure size can be adjusted if it gets big
            ax = fig.add_subplot(111)

            #plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
            plt.subplots_adjust(left=0.115, right=0.95, top=0.9, bottom=0.25)

            resp = ax.errorbar(x, yval, yerr=yerr, fmt='.', c='blue')

            #ax.axis([-1, max(x)+1, 0, max(yval+yerr)+1])
            ax.axis([-1, max(x)+1, min(yval-yerr)-1, max(yval+yerr)+1])
            ax.xaxis.grid(True, which='major')

            #ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
            #              alpha=1.0)

            ax.set_xticks(np.arange(maxpar-minpar))

            ax.set_title(r'Correlated equad values, page ' + str(pp))
            ax.set_ylabel(r'Correlated equad')
            #ax.legend(('One', 'Rutger ML', 'Two', 'Three',), shadow=True, fancybox=True, numpoints=1)
            #ax.set_yscale('log')

            xtickNames = plt.setp(ax, xticklabels=cequadnames[minpar:maxpar])
            #plt.getp(xtickNames)
            plt.setp(xtickNames, rotation=45, fontsize=8, ha='right')

            cequadfileout = open(outputdir+'/cequad-page-'+str(pp)+'.txt', 'w')
            for ii in range(minpar, maxpar):
                print str(cequadnames[ii]) + ":  " + str(yval[ii-minpar]) + " +/- " + str(yerr[ii-minpar])
                cequadfileout.write(str(cequadnames[ii]) + "  " + str(yval[ii-minpar]) + \
                                "  " + str(yerr[ii-minpar]) + "\n")

            cequadfileout.close()

            plt.savefig(fileout+'.png')
            plt.savefig(fileout+'.eps')
            plt.close(fig)



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
            fmin, fmax = confinterval(emceechain[:, minpar+jj], sigmalevel=1)
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
        plt.close(fig)

    # Make a triplot of all the other parameters
    if np.sum(dopar) > 1:
        indices = np.flatnonzero(np.array(dopar == True))
        fileout = outputdir+'/triplot'
        triplot(emceechain, parlabels=labels, plotparameters=indices)
        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')
        plt.close(fig)
    if np.sum(dopar) == 1:
        # Make a single plot
        indices = np.flatnonzero(np.array(dopar == True))
        f, axarr = plt.subplots(nrows=1, ncols=1)
        makesubplot1d(axarr, emceechain[:,indices[0]])
        fileout = outputdir+'/triplot'
        plt.savefig(fileout+'.png')
        plt.savefig(fileout+'.eps')
        plt.close(fig)


"""
Given a likelihood object, an MCMC chain file, and a JSON model file, this
function will process the MCMC chain and finds the upper-limit noise values that
should be used when deciding how many frequencies should be used in the
modelling. These values should later be given to 'numfreqsFromSpectrum', but for
now are written to the JSON file.
For now, this is only done for noise powerlaw signals, and DM powerlaw signals

@param likob:           The likelihood object belonging to the MCMC chain
@param chainfilename:   Name of the MCMC file that we will read in
@param jsonfile:        Name of the JSON file we will write to
"""
def calculateCompressionSpectrum(likob, chainfilename, jsonfile):
    # Find the power-law signals, and the DM signals
    rnsigs = likob.getSignalNumbersFromDict(likob.ptasignals, stype='powerlaw', \
            corr='single')
    dmsigs = likob.getSignalNumbersFromDict(likob.ptasignals, stype='dmpowerlaw', \
            corr='single')
    allsigs = np.append(rnsigs, dmsigs)

    # Read in the MCMC chain
    (logpost, loglik, chain, labels) = ReadMCMCFile(chainfilename)

    # Find the maximum likelihood value
    mlind = np.argmax(loglik)

    # For each of these, find the maximum amplitude signal to use for
    # compression
    for ss in allsigs:
        pi = likob.ptasignals[ss]['parindex']

        # Figure out the spectral index
        #if likob.ptasignals[ss]['stype'] == 'dmpowerlaw':
        #    si = 2.0
        #else:
        #    si = 3.3

        # Figure out which chain points to use
        if likob.ptasignals[ss]['bvary'][0] and \
                likob.ptasignals[ss]['bvary'][1]:
            si = chain[mlind, pi+1]

            # Both amplitude and spectral index in chain
            chainindices = np.flatnonzero(np.logical_and( \
                    chain[:,pi+1] > si-0.1, \
                    chain[:,pi+1] < si+0.1))
        elif likob.ptasignals[ss]['bvary'][0]:
            chainindices = np.arange(chain.shape[0])
        else:
            continue

        # Find the two-sigma upper-limit on the amplitude
        minamp, maxamp = confinterval(chain[:,pi][chainindices], \
                sigmalevel=3, onesided=True)

        # Add the maximum amplitude and spectral index to the dictionary
        likob.ptasignals[ss]['compressionSpectrum'] = [maxamp, si]

    # Write the JSON file to disk
    likob.writeModelToFile(jsonfile)



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
            raise ValueError("ERROR: burnin file not same dimensions. Mismatch {0} {1}".format(ndim, dim))

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
        raise ValueError("ERROR: burnin file not same dimensions. Mismatch {0} {1}".format(ndim, dim))

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
        raise ValueError("ERROR: burnin file not same dimensions. Mismatch {0} {1}".format(ndim, dim))
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


    pymultinest.run(likob.loglikelihoodhc, likob.samplefromprior, ndim,
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




"""
Run a generic PTMCMC algorithm.
"""
def RunPTMCMC(likob, steps, chainsdir, covfile=None, burnin=10000):
    # Save the parameters to file
    likob.saveModelParameters(chainsdir + '/ptparameters.txt')

    ndim = likob.dimensions
    pwidth = likob.pwidth.copy()

    if not covfile is None:
        cov = np.load(covfile)
        p0 = likob.pstart #+ 0.001*likob.pwidth
    else:
        cov = np.diag(pwidth**2)
        p0 = likob.pstart #+ likob.pwidth


    sampler = ptmcmc.PTSampler(ndim, likob.loglikelihood, likob.logprior, cov=cov, \
            outDir=chainsdir, verbose=True)

    sampler.sample(p0, steps, thin=1, burn=burnin)

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
