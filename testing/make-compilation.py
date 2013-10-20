import numpy as np
import matplotlib.pyplot as plt
import glob, os
import re

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, elevenSeventeen

# Set the page lay-out based on the assumption we have seven models per tabloid
w, h = landscape(elevenSeventeen)

size1, size2 = 0.33*h, 0.66*h
ratio1, ratio2 = 1500.0/1000, 1152.0/432
#ratio1, ratio2 = 1.0, 1.0

col2 = size1*ratio1
size3, size4, size5 = h/2, h/4, h/6

# Per pulsar, we are going to collect all the png files+evidence values, and put
# the figures in a single file
pngfiles = glob.glob(os.path.join('./', '*.png'))
pulsars = list(set([pngfile[2:7] for pngfile in pngfiles]))
#models = ['efacequad', 'cont0', 'cont0coarse', 'cont1', 'cont1coarse', 'cont2', 'cont2coarse']
models = ['FQ', 'FQP', 'FCP', 'FQCP', 'FQP1', 'FCP1', 'FQCP1']

figx = ['']*len(models)
figy = ['']*len(models)
figw = ['']*len(models)
figh = ['']*len(models)
figx[0], figy[0], figw[0], figh[0] = 0,     h-size1, size1*ratio1, size1
figx[1], figy[1], figw[1], figh[1] = 0,     h-2*size1, size1*ratio1, size1
figx[2], figy[2], figw[2], figh[2] = 0,     h-3*size1, size1*ratio1, size1
figx[3], figy[3], figw[3], figh[3] = col2,      h-3*size1, size1*ratio1, size1
figx[4], figy[4], figw[4], figh[4] = 2*col2,      h-1*size1, size1*ratio1, size1
figx[5], figy[5], figw[5], figh[5] = 2*col2,      h-2*size1, size1*ratio1, size1
figx[6], figy[6], figw[6], figh[6] = 2*col2,      h-3*size1, size1*ratio1, size1


# Make a tabloid for all pulsars
for psr in pulsars:
    print "Preparing compilation of pulsar {0}...".format(psr)
    c = canvas.Canvas("compilations/{0}.pdf".format(psr), pagesize=landscape(elevenSeventeen))

    #evidences = [0.0] * len(models)
    evidences = []
    for ii in range(len(models)):
        pngfile = psr + '-' + models[ii] + '-post_equal_weights.dat.fig.png'
        statsfile = psr + '-' + models[ii] + '-stats.dat'

        # if the result exists, place the figure
        if os.path.isfile(pngfile):
            print "  Model: {0} with file '{1}'".format(models[ii], pngfile)

            c.drawInlineImage(pngfile, figx[ii], figy[ii], height=figh[ii], width=figw[ii])

        if os.path.isfile(statsfile):
            # Read in the evidence for this model
            lines = open(statsfile).readlines()
            #evidences[ii] = float(re.search(r'Global Log-Evidence           :\s*(\S*)\s*\+/-\s*(\S*)',lines[0]).group(1))
            evidences.append(float(re.search(r'Global Log-Evidence           :\s*(\S*)\s*\+/-\s*(\S*)',lines[0]).group(1)))

    sortind = np.array(evidences).argsort()
    lineind = 0
    for ind in sortind[::-1]:
        c.setFont("Courier",14)
        #c.drawString(col2, h-15-16*(ii+1), models[ii]+': ' + str(evidences[ii]))
        c.drawString(1.2*col2, h-25-16*(lineind+1), "{0:>7}: {1}".format(models[ind], str(evidences[ind])))
        lineind += 1

    c.save()


