#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab
"""
Created by vhaasteren on 2013-08-06.
Copyright (c) 2013 Rutger van Haasteren

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator, NullFormatter, NullLocator
import matplotlib as mpl
import matplotlib.ticker
import os as os
import scipy.ndimage.filters as snf
from distutils.version import LooseVersion

"""
font = {'family' : 'serif',
        'serif'  : 'Computer modern Roman',
        'weight' : 'bold',
        'size'   : 8}

mpl.rc('font', **font)
#mpl.rc('text', usetex=True)
mpl.rc('text')
"""

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



def makesubplot2d(ax, samples1, samples2, weights=None):
    xmin = np.min(samples1)
    xmax = np.max(samples1)
    ymin = np.min(samples2)
    ymax = np.max(samples2)

    hist2d,xedges,yedges = np.histogram2d(samples1, samples2, weights=weights, \
            bins=40,range=[[xmin,xmax],[ymin,ymax]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]
    
    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
    yedges = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])
    
    level1, level2, level3 = getsigmalevels(hist2d)
    
    contourlevels = (level1, level2, level3)
    
    #contourcolors = ('darkblue', 'darkblue', 'darkblue')
    contourcolors = ('black', 'black', 'black')
    contourlinestyles = ('-', '--', ':')
    contourlinewidths = (2.0, 2.0, 2.0)
    contourlabels = [r'1 $\sigma$', r'2 $\sigma$',r'3 $\sigma$']
    
    line1 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[0], \
            linestyle=contourlinestyles[0], color=contourcolors[0])
    line2 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[1], \
            linestyle=contourlinestyles[1], color=contourcolors[1])
    line3 = plt.Line2D(range(10), range(10), linewidth=contourlinewidths[2], \
            linestyle=contourlinestyles[2], color=contourcolors[2])
    
    contall = (line1, line2, line3)
    contlabels = (contourlabels[0], contourlabels[1], contourlabels[2])

    c1 = ax.contour(xedges,yedges,hist2d.T,contourlevels, \
            colors=contourcolors, linestyles=contourlinestyles, \
            linewidths=contourlinewidths, zorder=2)

def makesubplot2denh(ax, samples1, samples2, weights=None):
    bins = 50

    xmin = np.min(samples1)
    xmax = np.max(samples1)
    ymin = np.min(samples2)
    ymax = np.max(samples2)
    hrange = [[xmin, xmax], [ymin, ymax]]

    [h, xs, ys] = np.histogram2d(samples1, samples2, bins=bins, normed=True, range=hrange)

    ax.contourf(0.5*(xs[1:]+xs[:-1]),0.5*(ys[1:]+ys[:-1]),h.T,cmap=plt.get_cmap('YlOrBr'))
    #ax.contourf(0.5*(xs[1:]+xs[:-1]),0.5*(ys[1:]+ys[:-1]),h.T,cmap=plt.get_cmap('jet'))
    #plt.hold(True)

    H,tmp1,tmp2 = np.histogram2d(samples1, samples2 ,bins=bins, range=hrange)

    H = snf.gaussian_filter(H,sigma=1.5)

    H = H / len(samples1)           # this is not correct with weights!
    Hflat = -np.sort(-H.flatten())  # sort highest to lowest
    cumprob = np.cumsum(Hflat)      # sum cumulative probability

    levels = [np.interp(level, cumprob, Hflat) for level in (0.6826,0.9547,0.9973)]

    xs = np.linspace(hrange[0][0], hrange[0][1], bins)
    ys = np.linspace(hrange[1][0], hrange[1][1], bins)

    ax.contour(xs,ys,H.T,levels,colors='k',linestyles=('-','--','-.'),linewidths=2)
    #plt.hold(False)

    
def makesubplot1d(ax, samples, weights=None):
    bins = 100
    xmin = np.min(samples)
    xmax = np.max(samples)

    if LooseVersion(np.__version__) >= LooseVersion('1.6.1'):
        hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights, density=True)
    else:
        hist, xedges = np.histogram(samples[:], bins=bins, range=(xmin,xmax), weights=weights)

    x = np.delete(xedges, -1) + 1.5*(xedges[1] - xedges[0])     # This should be 0.5*, but turns out this is a bug plotting of 'stepstyle' in matplotlib

    ax.plot(x, hist, 'k-', drawstyle='steps', linewidth=2.0)

    #ax.hist(samples, 100, color='k', histtype='bar', linewidth=2.0)



"""
Make a tri-plot of an MCMC chain.

@param chain: mcmc chain, with all columns the parameters of the chain
@param parlabels: the names of all parameters in the columns of the chain
@param plotparameters: list of the parameter indices to be plotter (None=all)
@param name: name of the plot (and of the figure file output)

Writes an eps and a png file as well
"""
def triplot(chain, parlabels=None, plotparameters=None, name=None):
    # Need chain, and parlabels

    # Figure out which parameters to plot
    samples = chain
    fileparameters = samples.shape[1]
    if plotparameters is None:
        parameters = np.arange(fileparameters)
    else:
        plotparameters = np.array(plotparameters)
        parameters = plotparameters[plotparameters < fileparameters]
        

    # Create the plot array

    # "plt.figure(figsize=(10,20))    # Resize the total figure\n",
    f, axarr = plt.subplots(nrows=len(parameters), ncols=len(parameters), \
            figsize=(15, 10))

    for i in range(len(parameters)):
        # for j in len(parameters[np.where(i <= parameters)]:
        for j in range(len(parameters)):
            ii = i
            jj = len(parameters) - j - 1

            xmajorLocator = matplotlib.ticker.MaxNLocator(nbins=4,prune='both')#LinearLocator(3)
            ymajorLocator = matplotlib.ticker.MaxNLocator(nbins=4,prune='both')#LinearLocator(3)

            if j <= len(parameters)-i-1:
                axarr[jj][ii].xaxis.set_minor_locator(NullLocator())
                axarr[jj][ii].yaxis.set_minor_locator(NullLocator())
                axarr[jj][ii].xaxis.set_major_locator(NullLocator())
                axarr[jj][ii].yaxis.set_major_locator(NullLocator())

                axarr[jj][ii].xaxis.set_minor_formatter(NullFormatter())
                axarr[jj][ii].yaxis.set_minor_formatter(NullFormatter())
                axarr[jj][ii].xaxis.set_major_formatter(NullFormatter())
                axarr[jj][ii].yaxis.set_major_formatter(NullFormatter())
                xmajorFormatter = FormatStrFormatter('%g')
                ymajorFormatter = FormatStrFormatter('%g')

                if ii == jj:
                    # Make a 1D plot
                    makesubplot1d(axarr[ii][ii], samples[:,parameters[ii]])
                else:
                    # Make a 2D plot
                    makesubplot2denh(axarr[jj][ii], samples[:,parameters[ii]], \
                            samples[:,parameters[jj]])

                axarr[jj][ii].xaxis.set_major_locator(xmajorLocator)
                axarr[jj][ii].yaxis.set_major_locator(ymajorLocator)
            else:
                axarr[jj][ii].set_visible(False)
                #axarr[jj][ii].axis('off')

            if jj == len(parameters)-1:
                axarr[jj][ii].xaxis.set_major_formatter(xmajorFormatter)
                axarr[jj][ii].set_xlabel(parlabels[parameters[ii]])

            if ii == 0:
                if jj == 0:
                    axarr[jj][ii].yaxis.set_major_locator(NullLocator())
                    axarr[jj][ii].set_ylabel('Post.')
                else:
                    axarr[jj][ii].yaxis.set_major_formatter(ymajorFormatter)
                    axarr[jj][ii].set_ylabel(parlabels[parameters[jj]])


    if name is not None:
        #f.suptitle(shortname[-10:])
        f.suptitle(name)

        #f.subplots_adjust(hspace=0)
        #plt.setp([a.get_xticklabels() for a in f.axes[:-0-2]], visible=False)
        #plt.tight_layout() # Or equivalently,  "plt.tight_layout()"

        #plt.savefig('pulsar-' + str(psr) + '.png')
        plt.savefig(name+'.fig.png')
        plt.savefig(name+'.fig.eps')
        #plt.show()

    """
    # Fine-tune figure: make subplots close to each other and hide x ticks for all
    # but the bottom plot
    # Also add some space for the legend below the plots
    f.subplots_adjust(bottom=0.22)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-0-2]], visible=False)
    """
