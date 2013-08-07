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
    
    
def makesubplot1d(ax, samples, weights=None):
    ax.hist(samples, 100, color='k', histtype='bar', linewidth=2.0)



# The mcmc chain (ASCII file, with columns the values of the parameters each step)
# Note that the first two columns are walker index, and loglikelihood value
# So the indices are used '+2' in the chain below
# Layout ASCII-file
#     Col 1        Col 2        Col 3           Col 4
#   walker id  loglikelihood  parameter 1   parameter 2
#   walker id  loglikelihood  parameter 1   parameter 2
#  .... etc.


def triplot(chainfilename, plotparameters=None, minmaxfile=None):
    #shortname=options.root
    chainfilename = chainfilename
    parametersfilename = chainfilename+'.parameters.txt'
    figurefilenameeps = chainfilename+'.fig.eps'
    figurefilenamepng = chainfilename+'.fig.png'
    chain = np.loadtxt(chainfilename)

    if minmaxfile==None:
        minmaxfile = chainfilename+'.minmax.txt'

    # Check if we are making MultiNest triplots (rescale the parameters, and use
    # posterior weights
    weights=None
    if os.path.exists(minmaxfile):
        minmax = np.loadtxt(minmaxfile)
        for ii in range(chain.shape[1]-2):
          chain[:,ii+2] = minmax[ii,0] + chain[:,ii+2] * (minmax[ii,1] - minmax[ii,0])
        weights = chain[:,0]

    #print "shortname = ", shortname
    print "parametersfilename = ", parametersfilename
    print "figurefilename = ", figurefilenameeps
    print "chainfilename = ", chainfilename

    # Read in the labels for all parameters from parametersfilename
    parfile = open(parametersfilename)
    lines=[line.strip() for line in parfile]
    parlabels=[]
    for i in range(len(lines)):
        lines[i]=lines[i].split(" ")
        if lines[i][0] >= 0:
            # If the parameter has an index
            parlabels.append(lines[i][5])
        #parlabels.append(lines[i])

    #print "Parameter labels", parlabels

    # Figure out which parameters to plot
    fileparameters = chain.shape[1]-2
    if plotparameters is None:
        parameters = np.arange(fileparameters)
    else:
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
                    makesubplot1d(axarr[ii][ii], chain[:,parameters[ii]+2], \
                            weights=weights)
                else:
                    # Make a 2D plot
                    makesubplot2d(axarr[jj][ii], chain[:,parameters[ii]+2], \
                            chain[:,parameters[jj]+2], weights=weights)

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


    #f.suptitle(shortname[-10:])
    f.suptitle(chainfilename)

    #f.subplots_adjust(hspace=0)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-0-2]], visible=False)
    #plt.tight_layout() # Or equivalently,  "plt.tight_layout()"

    #plt.savefig('pulsar-' + str(psr) + '.png')
    plt.savefig(figurefilenameeps)
    plt.savefig(figurefilenamepng)
    #plt.show()

    """
    # Fine-tune figure: make subplots close to each other and hide x ticks for all
    # but the bottom plot
    # Also add some space for the legend below the plots
    f.subplots_adjust(bottom=0.22)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-0-2]], visible=False)
    """
