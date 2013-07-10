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

import numpy as np
import scipy.linalg as sl, scipy.special as ss
import h5py as h5
import sets as sets
#import bangwrapper as bw
import matplotlib.pyplot as plt
import os as os
import sys
import libstempo as t2
import pytwalk
import emcee
import statsmodels.api as sm


# In order to parametrise the correlations of a GWB, or any other signal, we
# need to have a numbering scheme between parameter number, and correlation
# coefficient. This particular scheme stems from a much earlier implementation,
# and it is currently not used in the bang-legacy interface. However, we include
# it here since these parameters are present in the HDF5 format. Any other
# transformation of these parameters (e.g. Lentati et al., 2013 in prep.) can of
# course be used as well.
#
# The GWB is a is a power-law signal, correlated between all pulsars. Because we
# also want to be able to construct these correlation coefficients from the
# data, we need to be able to include these correlations as model parameters as
# well. Therefore, a GWB that applies to k pulsars has k*(k+1)/2 extra
# parameters: the H&D correlation parameters. For convenience, we define a
# numbering scheme for these correlation parameters.  The correlation index
# between pulsar i and j, with 0 <= j <= i < k is n_{i,j} = j + i(i+1)/2
# Example:
# 
# i =  0  1  2  3  4         j
#    _______________________ =
#    | 0  1  3  6  10  ... | 0
#    |    2  4  7  11  ... | 1
#    |       5  8  12  ... | 2
#    |          9  13  ... | 3
#    |             14  ... | 4
#
#  So for n_{3,2} = 8
#
# TODO: Is it necessary to implement this scheme ourselves? Use for now, with
# these functions:
# Calculate n(i,j)
def cn(i, j):
    return np.int32(np.float32(j) + np.float32(i)*(np.float32(i)+1)/2)
    
# Calculate i(n)
def ci(n):
    return np.int32(-0.5 + math.sqrt(2*np.float32(n)+0.25))

# Calculate j(n)
def cj(n):
    return np.int32(n - cn(ci(n), 0))

# Given i and j, what is the parameter number
def cparfromij(i, j):
    return cn(i, j) + 3

# Given the parameter number, what is i and j
def cijfrompar(p):
    return ci(p-3), cj(p-3)

# Given two coordinate pairs, what is the GWB correlation? Note: this assumes
# that no two pulsars have exactly the same coordinates
# TODO: Check consistency of the arguments
def gwbcorr(raja, decja, rajb, decjb):
    dcor = np.zeros(len(raja))

    # Calculate the dot-product of the two pulsar positions
    kax = np.cos(decja)*np.cos(raja)
    kay = np.cos(decja)*np.sin(raja)
    kaz = np.sin(decja)
    kbx = np.cos(decjb)*np.cos(rajb)
    kby = np.cos(decjb)*np.sin(rajb)
    kbz = np.sin(decjb)
    dotprod = kax*kbx+kay*kby+kaz*kbz

    # This is the Hellings & Downs correlation
    dx = 0.5 * (1.0 - dotprod)

    # Find out which indices to set ourselves due to log function
    inda = dx <= 0.0
    indc = np.logical_not(inda)

    # Those correlations are 0.5
    dcor[inda] = 0.5

    # The others are given by the Hellings and Downs function
    dcor[indc] = 1.5 * dx[indc] * np.log(dx[indc]) - 0.25 * dx[indc] + 0.5

    # If the two locations are the same (same pulsar), add 0.5 to correlation
    addition = np.logical_and(raja==rajb, decja==decjb)
    dcor += addition * np.array([0.5] * len(raja))

    return dcor



"""
The DataFile class is a very first attempt to support the HDF5 file format. It
most likely needs to be re-designed, since it does not provide a universal
interface, and it is not completely independent from the bang-legacy wrapper
(the data compression is done with calls to the bang-legacy code)

Without calls to the bang-legacy wrapper, it can already be used to
import pulsars from tempo2 into the HDF5 file format, and create the model for
datachallenge and EPTA/IPTA data analysis/GWB searches.
"""
class DataFile(object):
    filename = None
    h5file = None
    pardict = None
    allpardict = None

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

        ## Write the TOAs, residuals, and uncertainties.
        #spd = 24.0*3600     # seconds per day
        #pulsarsgroup.create_dataset('TOAs', data=t2data[:,0]*spd)       # days (MJD) * sec per day
        #pulsarsgroup.create_dataset('prefitRes', data=t2data[:,3])      # seconds
        #pulsarsgroup.create_dataset('postfitRes', data=t2data[:,1])     # seconds
        #pulsarsgroup.create_dataset('toaErr', data=t2data[:,2])         # seconds
        #pulsarsgroup.create_dataset('freq', data=t2data[:,4])           # MHz

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

        # Obtain the timing model parameters
        #tmparameters = t2pulsar.numpypars()

        # Calculate the maximum string length of the TMP name
        #maxlen = max(len(parname) for parname in tmparameters['name'])
        #maxlen = max(maxlen, len('OFFSET'))
        #offsetname = np.array(['OFFSET'], dtype='a'+str(maxlen))

        # Add the offset that is always fit for to the list of timing model parameters
        # TODO: Do not down-convert quad precision to double precision here
        # TODO: Scale with the value of jumps etc. Is there something useful
        #       that can be done here?
        #tmpname = np.append(offsetname, tmparameters['name'])
        #tmpvalpre = np.double(np.append(np.array([0.0]), tmparameters['val']))
        #tmpvalpost = np.double(np.append(np.array([0.0]), tmparameters['pval']))
        #tmperrpre = np.double(np.append(np.array([0.0]), tmparameters['err']))
        #tmperrpost = np.double(np.append(np.array([0.0]), tmparameters['perr']))

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

        # Close the hdf5 file
        self.h5file.close()
        self.h5file = None

    """
    Create a model folder, and process all the data with all the flags
    """
    def preprocessmodeldata(self):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the data group
        if not "Data" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no data group in hdf5 file"

        datagroup = self.h5file["Data"]

        # Retrieve the list of pulsars
        if not "Pulsars" in datagroup:
            raise IOError, "no pulsar group in hdf5 file"

        pulsarsgroup = datagroup["Pulsars"]

        # Create the model group
        if not "Models" in self.h5file:
            self.h5file.create_group("Models")

        modelsgroup = self.h5file["Models"]

        # Create the model, named "rgp-ban-legacy"
        modelname = "rgp-ban-legacy"
        if "rgp-ban-legacy" in modelsgroup:
            print "WARNING: deleting already existing model (%s)" % (modelname)
            del modelsgroup[modelname]

        modelgroup = modelsgroup.create_group(modelname)

        # Write an identifier that it is a van haasteren et al. (2009) model
        #   This does not include data compression
        modelid = 'vHLML2009'
        modelgroup.create_dataset('modelid', data=np.array([modelid]), dtype='a'+str(len(modelid)))

        # Create a data folder, which contains a copy of all the data, but now in a more
        # usable format
        # TODO: make these into symbolic links?
        processeddatagroup = modelgroup.create_group("ProcessedData")

        # The relevant data in this model (quad precision numpy arrays)
        totaltoas = np.array([], dtype='f16')
        totalprefitres = np.array([], dtype='f16')
        totalpostfitres = np.array([], dtype='f16')
        totaltoaerr = np.array([], dtype='f16')
        totalfreq = np.array([], dtype='f16')

        # The flags in this model (list)
        totalflags = []

        # Loop over all pulsars, and create the datasets
        for pulsar in pulsarsgroup:
            totaltoas = np.append(totaltoas, pulsarsgroup[pulsar]['TOAs'])
            totalprefitres = np.append(totalprefitres, pulsarsgroup[pulsar]['prefitRes'])
            totalpostfitres = np.append(totalpostfitres, pulsarsgroup[pulsar]['postfitRes'])
            totaltoaerr = np.append(totaltoaerr, pulsarsgroup[pulsar]['toaErr'])
            totalfreq = np.append(totalfreq, pulsarsgroup[pulsar]['freq'])

            if not "Flags" in pulsarsgroup[pulsar]:
                raise IOError, "no flags group in hdf5 file for pulsar (%s)" % (pulsar)

            flagsgroup = pulsarsgroup[pulsar]["Flags"]
            totalflags += list(flagsgroup)

        # Write the data to the processeddatagroup 
        # TODO: Do not downconvert to double precision
        processeddatagroup.create_dataset('TOAs', data=np.double(totaltoas))
        processeddatagroup.create_dataset('prefitRes', data=np.double(totalprefitres))
        processeddatagroup.create_dataset('postfitRes', data=np.double(totalpostfitres))
        processeddatagroup.create_dataset('toaErr', data=np.double(totaltoaerr))
        processeddatagroup.create_dataset('freq', data=np.double(totalfreq))

        # Create the model group for flags
        modelflaggroup = processeddatagroup.create_group("Flags")

        # Reduce the flags to just the unique flags in the files
        uniqueflags = sets.Set(totalflags)

        print "Looping over all flags (Make this faster!)"

        # Loop over all the flags, and write the all-pulsar flag group
        for flag in uniqueflags:
            flagvalues = []
            for pulsar in pulsarsgroup:
                flagsgroup = pulsarsgroup[pulsar]["Flags"]

                if flag in flagsgroup:
                    # This line is what takes up so much time
                    flagvalues += str(flagsgroup[flag])
                else:
                    flagvalues += [""] * len(pulsarsgroup[pulsar]['TOAs'])

            # Write these flag values
            modelflaggroup.create_dataset(flag, data=flagvalues)

        print "Done with that"

        # Create a few more flags for each TOA:
        # - The pulsarid flag, an enumerated value (starting 0) indicating pulsar id
        # - The pulsar name flag, which is equal to the HDF5 used name
        # - The efacequad flag, which is a unique identifier for the efac/equad signals
        totalpulsarid = np.empty(0, dtype='u4')
        totalpulsarname = []
        totalefacequad = []
        pid = 0
        for pulsar in pulsarsgroup:
            nobs = len(pulsarsgroup[pulsar]['TOAs'])

            # Construct the flags for just this pulsar first
            pulsarname = map(str, [pulsar] * nobs)
            pulsarid = pid*np.ones(nobs, dtype='u4')

            # Concatenate the pulsar name with the sys flag, and make that the efacequad
            # flag
            if "sys" in pulsarsgroup[pulsar]["Flags"]:
                efacequad = map('-'.join, zip(pulsarname, pulsarsgroup[pulsar]["Flags"]['sys']))
            else:
                efacequad = pulsarname

            # Add these three to the total
            totalpulsarname += map(str, pulsarname)
            totalpulsarid = np.append(totalpulsarid, pulsarid)
            totalefacequad += map(str, efacequad)

            pid += 1

        modelflaggroup.create_dataset("pulsarid", data=totalpulsarid)
        modelflaggroup.create_dataset("pulsarname", data=totalpulsarname)
        modelflaggroup.create_dataset("efacequad", data=totalefacequad)

        self.h5file.close()
        self.h5file = None

    """
    Add the linear model of all the tempo2 design matrices
    """
    def addt2signals(self):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]


        # Add the linear signals
        if not 'Signals' in modelgroup:
            modelgroup.create_group("Signals")
        signalgroup = modelgroup["Signals"]

        # Add all the timing model parameters: linear timing model parameters
        lineartmp = signalgroup.create_group("linear")
        lineartempo2 = lineartmp.create_group("tempo2")

        # Figure out how many toas and timing model parameters we have
        # TODO: This should not be done through the pulsarsgroup
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        totalobs = len(processeddatagroup['TOAs'])
        totaltmps = 0
        for pulsar in pulsarsgroup:
            totaltmps += len(pulsarsgroup[pulsar]['tmp_name'])

        # Create the design matrix (this one is double precision), and the timing model
        # parameter values/descriptions
        designmatrix = np.zeros((totalobs, totaltmps), dtype='f8')
        totaltmp_name = []
        totaltmp_valpost = np.empty(0, dtype='f8')
        totaltmp_valpre = np.empty(0, dtype='f8')
        totaltmp_errpost = np.empty(0, dtype='f8')
        totaltmp_errpre = np.empty(0, dtype='f8')
        index1 = 0
        index2 = 0
        for pulsar in pulsarsgroup:
            # Do the design matrix
            nobs = len(pulsarsgroup[pulsar]['TOAs'])
            ntmps = len(pulsarsgroup[pulsar]['tmp_name'])
            # Use the comma, and not [..][..] when slicing multiple dimensions
            designmatrix[index1:(index1+nobs),index2:(index2+ntmps)] = pulsarsgroup[pulsar]['designmatrix']

            # Create the timing model parameter name
            totaltmp_name += map(str, map(':'.join, zip([pulsar]*ntmps, pulsarsgroup[pulsar]['tmp_name'])))
            totaltmp_valpost = np.append(totaltmp_valpost, pulsarsgroup[pulsar]['tmp_valpost'])
            totaltmp_valpre = np.append(totaltmp_valpre, pulsarsgroup[pulsar]['tmp_valpre'])
            totaltmp_errpost = np.append(totaltmp_errpost, pulsarsgroup[pulsar]['tmp_errpost'])
            totaltmp_errpre = np.append(totaltmp_errpre, pulsarsgroup[pulsar]['tmp_errpre'])

            index1 += nobs
            index2 += ntmps

        lineartempo2.create_dataset("designmatrix", data=designmatrix)
        lineartempo2.create_dataset("tmp_name", data=totaltmp_name)
        lineartempo2.create_dataset("tmp_valpost", data=totaltmp_valpost)
        lineartempo2.create_dataset("tmp_valpre", data=totaltmp_valpre)
        lineartempo2.create_dataset("tmp_errpost", data=totaltmp_errpost)
        lineartempo2.create_dataset("tmp_errpre", data=totaltmp_errpre)

        self.h5file.close()
        self.h5file = None

    """
    Add the error bars, and efac parameters
    """
    def addmodelerrorbars(self, vary=False, separateforeeflags=False):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        psrnames = map(str, pulsarsgroup)

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]


        # Add the linear signals
        if not 'Signals' in modelgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        signalgroup = modelgroup["Signals"]
        modelflaggroup = processeddatagroup["Flags"]

        # The stochastic signals require a 
        stochasticsignals = signalgroup.create_group("stochastic")

        # All unique efacequad flags require an efac error-bar source, and an equad
        uefacequad = psrnames
        flagname = 'pulsarname'
        if separateforeeflags:
            uefacequad = sets.Set(modelflaggroup["efacequad"])
            flagname = 'efacequad'
        else:
            uefacequad = psrnames
            flagname = 'pulsarname'

        for eeflag in uefacequad:
            eelabel = 'efac-' + str(eeflag)
            efacsource = stochasticsignals.create_group(eelabel)
            efacsource.create_dataset("type", data=['efac', 'uncor'])
            efacsource.create_dataset("flag", data=[flagname, eeflag])
            efacsource.create_dataset("parname", data=['efac'])
            efacsource.create_dataset("units", data=['log(linear)'])
            efacsource.create_dataset("min", data=[-5])
            efacsource.create_dataset("max", data=[5.0])
            efacsource.create_dataset("start", data=[0.0])
            efacsource.create_dataset("stepscale", data=[0.1])
            efacsource.create_dataset("vary", data=[vary])

        self.h5file.close()
        self.h5file = None

    """
    Add the white/jitter noise (equad)
    """
    def addwhitenoise(self, vary=False, separateforeeflags=False):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        psrnames = map(str, pulsarsgroup)

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]


        # Add the linear signals
        if not 'Signals' in modelgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        signalgroup = modelgroup["Signals"]
        modelflaggroup = processeddatagroup["Flags"]

        # The stochastic signals require a 
        if not "stochastic" in signalgroup:
            signalgroup.create_group("stochastic")

        stochasticsignals = signalgroup["stochastic"]

        # All unique efacequad flags require an efac error-bar source, and an equad
        uefacequad = psrnames
        flagname = 'pulsarname'
        if separateforeeflags:
            uefacequad = sets.Set(modelflaggroup["efacequad"])
            flagname = 'efacequad'
        else:
            uefacequad = psrnames
            flagname = 'pulsarname'

        for eeflag in uefacequad:
            eelabel = 'equad-' + str(eeflag)
            equadsource = stochasticsignals.create_group(eelabel)
            equadsource.create_dataset("type", data=['equad', 'uncor'])
            equadsource.create_dataset("flag", data=[flagname, eeflag])
            equadsource.create_dataset("parname", data=['equad'])
            equadsource.create_dataset("units", data=['log(linear:sec)'])
            equadsource.create_dataset("min", data=[np.log(1e-11)])
            equadsource.create_dataset("max", data=[np.log(1.0e-3)])
            equadsource.create_dataset("start", data=[np.log(1e-7)])
            equadsource.create_dataset("stepscale", data=[0.1])
            equadsource.create_dataset("vary", data=[vary])

        self.h5file.close()
        self.h5file = None

    """
    Add power-law red noise
    """
    def addrednoise(self, vary=False, separateforeeflags=False):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        psrnames = map(str, pulsarsgroup)

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]


        # Add the linear signals
        if not 'Signals' in modelgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        signalgroup = modelgroup["Signals"]
        modelflaggroup = processeddatagroup["Flags"]

        # The stochastic signals require a 
        if not "stochastic" in signalgroup:
            signalgroup.create_group("stochastic")

        stochasticsignals = signalgroup["stochastic"]

        # All unique efacequad flags require an efac error-bar source, and an equad
        flagname = 'pulsarname'
        pulsarflags = psrnames
        if separateforeeflags:
            pulsarflags = sets.Set(modelflaggroup["efacequad"])
            flagname = 'efacequad'
        else:
            flagname = 'pulsarname'
            pulsarflags = psrnames

        for pflag in pulsarflags:
            rnlabel = 'rednoise-' + str(pflag)
            rnsource = stochasticsignals.create_group(rnlabel)
            rnsource.create_dataset("type", data=['powerlaw', 'uncor'])
            rnsource.create_dataset("flag", data=[flagname, pflag])
            rnsource.create_dataset("parname", data=['amplitude', 'spectral index', 'low-frequency cut off'])
            # TODO: come up with better units
            rnsource.create_dataset("units", data=['log(linear:1e-15)', 'linear', 'yr-1'])
            rnsource.create_dataset("min", data=[np.log(0.1), 1.16, 0.00001])
            rnsource.create_dataset("max", data=[np.log(1.0e4), 6.82, 1.0])
            rnsource.create_dataset("start", data=[0.00, 2.16, 0.05])
            rnsource.create_dataset("stepscale", data=[0.1, 0.1, 0.01])
            rnsource.create_dataset("vary", data=[vary, vary, False])

        self.h5file.close()
        self.h5file = None

    """
    Add power-law dispersion measure variations
    """
    def adddmv(self, vary=False, separateforeeflags=False):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        psrnames = map(str, pulsarsgroup)

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]

        # Check for the linear signals
        if not 'Signals' in modelgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        signalgroup = modelgroup["Signals"]
        modelflaggroup = processeddatagroup["Flags"]

        # The stochastic signals require a 
        if not "stochastic" in signalgroup:
            signalgroup.create_group("stochastic")

        stochasticsignals = signalgroup["stochastic"]

        # All unique efacequad flags require an efac error-bar source, and an equad
        flagname = 'pulsarname'
        pulsarflags = psrnames
        if separateforeeflags:
            pulsarflags = sets.Set(modelflaggroup["efacequad"])
            flagname = 'efacequad'
        else:
            flagname = 'pulsarname'
            pulsarflags = psrnames

        for pflag in pulsarflags:
            rnlabel = 'dmv-' + str(pflag)
            rnsource = stochasticsignals.create_group(rnlabel)
            rnsource.create_dataset("type", data=['dmv', 'uncor'])
            rnsource.create_dataset("flag", data=[flagname, pflag])
            rnsource.create_dataset("parname", data=['amplitude', 'spectral index', 'low-frequency cut off'])
            # TODO: come up with better units
            rnsource.create_dataset("units", data=['log(linear)', 'linear', 'yr-1'])
            rnsource.create_dataset("min", data=[np.log(0.1), 1.16, 0.00001])
            rnsource.create_dataset("max", data=[np.log(1.0e4), 6.82, 1.0])
            rnsource.create_dataset("start", data=[0.00, 2.16, 0.05])
            rnsource.create_dataset("stepscale", data=[0.1, 0.1, 0.01])
            rnsource.create_dataset("vary", data=[vary, vary, False])

        self.h5file.close()
        self.h5file = None


    """
    Add a GWB signal
    """
    def addgwb(self, vary=False):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        psrnames = map(str, pulsarsgroup)

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]


        # Add the linear signals
        if not 'Signals' in modelgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        signalgroup = modelgroup["Signals"]
        modelflaggroup = processeddatagroup["Flags"]

        # The stochastic signals require a 
        if not "stochastic" in signalgroup:
            signalgroup.create_group("stochastic")

        stochasticsignals = signalgroup["stochastic"]

        pulsarflags = psrnames
        k = len(pulsarflags)

        # Label the pulsar indices of the upper triangular correlation matrix
        index_i, index_j = np.triu_indices(k)
        index_n = cn(index_i, index_j)

        # Retrieve the list of pulsars
        if not "Pulsars" in datagroup:
            raise IOError, "no pulsar group in hdf5 file"

        pulsarsgroup = datagroup["Pulsars"]

        # Pulsar angles
        # Note: in ban-legacy, the declination is different. Here we use the tempo2
        # convention.
        # decl_tempo2 = pi/2 - decl_ban-legacy
        # TODO: This should not depend on pulsarsgroup
        raj = np.zeros(k)
        decj = np.zeros(k)
        pid = 0
        for pulsar in pulsarsgroup:
            rajind = np.flatnonzero(np.array(pulsarsgroup[pulsar]['tmp_name']) == 'RAJ')
            decjind = np.flatnonzero(np.array(pulsarsgroup[pulsar]['tmp_name']) == 'DECJ')
            if len(rajind) != 1 or len(decjind) != 1:
                raise IOError, "RAJ or DECJ not set properly for " + pulsarname

            raj[pid] = np.array(pulsarsgroup[pulsar]['tmp_valpre'])[rajind]
            decj[pid] = np.array(pulsarsgroup[pulsar]['tmp_valpre'])[decjind]
            pid += 1


        # Make an array, k*(k+1)/2 long, of pulsar positions which list all possible
        # coordinate combinations. Also name the combinations as:
        # J0030+451:J0437-4715
        pulsarnames = np.array(list(pulsarsgroup))
        raj_index_i, decj_index_i = raj[index_i], decj[index_i]
        raj_index_j, decj_index_j = raj[index_j], decj[index_j]
        hdcoefnames = map(str, map(':'.join, zip(pulsarnames[index_i], pulsarnames[index_j])))

        # Calculate the k*(k+1)/2 Hellings & Downs coefficients
        hdcoefs = gwbcorr(raj_index_i, decj_index_i, raj_index_j, decj_index_j)

        # Set the label, and the gwb parameters
        gwblabel = 'gwb-pow-gr'
        gwbtype = ['powerlaw', 'gr']
        gwbflags = ['pulsarname'] + pulsarflags
        gwbparnames = ['amplitude', 'spectral index', 'low-frequency cut off'] + hdcoefnames
        gwbunits = ['log(linear:1e-15)', 'linear', 'yr-1'] + ['corr']*len(hdcoefs)
        gwbparsstart = np.append([0.0, 4.33, 0.05], hdcoefs)
        gwbparsmin = [np.log(0.01), 1.16, 1.0e-7] + [-1.0]*len(hdcoefs)
        gwbparsmax = [np.log(1.0e4), 6.85, 1.00] + [1.0]*len(hdcoefs)
        gwbparstepscale = [0.1, 0.1, 0.01] + [0.1]*len(hdcoefs)
        gwbparsvary = [vary, vary, False] + [False]*len(hdcoefs)

        gwbsource = stochasticsignals.create_group(gwblabel)
        gwbsource.create_dataset("type", data=gwbtype)
        gwbsource.create_dataset("flag", data=gwbflags)
        gwbsource.create_dataset("parname", data=gwbparnames)
        gwbsource.create_dataset("units", data=gwbunits)
        gwbsource.create_dataset("min", data=gwbparsmin)
        gwbsource.create_dataset("max", data=gwbparsmax)
        gwbsource.create_dataset("start", data=gwbparsstart)
        gwbsource.create_dataset("stepscale", data=gwbparstepscale)
        gwbsource.create_dataset("vary", data=gwbparsvary)

        self.h5file.close()
        self.h5file = None


    """
    Add a Time-Standard signal
    """
    def addclockerr(self, vary=False):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        psrnames = map(str, pulsarsgroup)

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]


        # Add the linear signals
        if not 'Signals' in modelgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        signalgroup = modelgroup["Signals"]
        modelflaggroup = processeddatagroup["Flags"]

        # The stochastic signals require a 
        if not "stochastic" in signalgroup:
            signalgroup.create_group("stochastic")

        stochasticsignals = signalgroup["stochastic"]

        pulsarflags = psrnames
        k = len(pulsarflags)

        # Label the pulsar indices of the upper triangular correlation matrix
        index_i, index_j = np.triu_indices(k)
        index_n = cn(index_i, index_j)

        # Retrieve the list of pulsars
        if not "Pulsars" in datagroup:
            raise IOError, "no pulsar group in hdf5 file"

        pulsarsgroup = datagroup["Pulsars"]

        # Pulsar angles
        # Note: in ban-legacy, the declination is different. Here we use the tempo2
        # convention.
        # decl_tempo2 = pi/2 - decl_ban-legacy
        # TODO: This should not depend on pulsarsgroup
        raj = np.zeros(k)
        decj = np.zeros(k)
        pid = 0
        for pulsar in pulsarsgroup:
            rajind = np.flatnonzero(np.array(pulsarsgroup[pulsar]['tmp_name']) == 'RAJ')
            decjind = np.flatnonzero(np.array(pulsarsgroup[pulsar]['tmp_name']) == 'DECJ')
            if len(rajind) != 1 or len(decjind) != 1:
                raise IOError, "RAJ or DECJ not set properly for " + pulsarname

            raj[pid] = np.array(pulsarsgroup[pulsar]['tmp_valpre'])[rajind]
            decj[pid] = np.array(pulsarsgroup[pulsar]['tmp_valpre'])[decjind]
            pid += 1


        # Make an array, k*(k+1)/2 long, of pulsar positions which list all possible
        # coordinate combinations. Also name the combinations as:
        # J0030+451:J0437-4715
        pulsarnames = np.array(list(pulsarsgroup))
        raj_index_i, decj_index_i = raj[index_i], decj[index_i]
        raj_index_j, decj_index_j = raj[index_j], decj[index_j]
        hdcoefnames = map(str, map(':'.join, zip(pulsarnames[index_i], pulsarnames[index_j])))

        # Calculate the k*(k+1)/2 Hellings & Downs coefficients
        hdcoefs = gwbcorr(raj_index_i, decj_index_i, raj_index_j, decj_index_j)

        # Set the label, and the clock parameters
        clocklabel = 'clock-pow-gr'
        clocktype = ['powerlaw', 'gr']
        clockflags = ['pulsarname'] + pulsarflags
        clockparnames = ['amplitude', 'spectral index', 'low-frequency cut off'] + hdcoefnames
        clockunits = ['log(linear:1e-15)', 'linear', 'yr-1'] + ['corr']*len(hdcoefs)
        clockparsstart = np.append([0.0, 4.33, 0.05], hdcoefs)
        clockparsmin = [np.log(0.01), 1.16, 1.0e-7] + [-1.0]*len(hdcoefs)
        clockparsmax = [np.log(1.0e4), 6.85, 1.00] + [1.0]*len(hdcoefs)
        clockparstepscale = [0.1, 0.1, 0.01] + [0.1]*len(hdcoefs)
        clockparsvary = [vary, vary, False] + [False]*len(hdcoefs)

        clocksource = stochasticsignals.create_group(clocklabel)
        clocksource.create_dataset("type", data=clocktype)
        clocksource.create_dataset("flag", data=clockflags)
        clocksource.create_dataset("parname", data=clockparnames)
        clocksource.create_dataset("units", data=clockunits)
        clocksource.create_dataset("min", data=clockparsmin)
        clocksource.create_dataset("max", data=clockparsmax)
        clocksource.create_dataset("start", data=clockparsstart)
        clocksource.create_dataset("stepscale", data=clockparstepscale)
        clocksource.create_dataset("vary", data=clockparsvary)

        self.h5file.close()
        self.h5file = None


    """
    Add an anisotropic GWB signal (up to dipole order)
    """
    def adddipolegwb(self, vary=False):
        # Open read/write, but file must exist
        self.h5file = h5.File(self.filename, 'r+')

        # Retrieve the models group
        if not "Models" in self.h5file:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        modelsgroup = self.h5file["Models"]
        datagroup = self.h5file["Data"]
        pulsarsgroup = datagroup["Pulsars"]
        psrnames = map(str, pulsarsgroup)

        modelname = "rgp-ban-legacy"
        modelgroup = modelsgroup[modelname]
        processeddatagroup = modelgroup["ProcessedData"]


        # Add the linear signals
        if not 'Signals' in modelgroup:
            self.h5file.close()
            self.h5file = None
            raise IOError, "no Models group in hdf5 file"

        signalgroup = modelgroup["Signals"]
        modelflaggroup = processeddatagroup["Flags"]

        # The stochastic signals require a 
        if not "stochastic" in signalgroup:
            signalgroup.create_group("stochastic")

        stochasticsignals = signalgroup["stochastic"]

        pulsarflags = psrnames
        k = len(pulsarflags)

        # Set the label, and the gwb parameters
        gwbflags = ['pulsarname'] + pulsarflags
        gwbparnames = ['amplitude', 'spectral index', 'low-frequency cut off']
        gwbunits = ['log(linear:1e-15)', 'linear', 'yr-1']
        gwbparsstart = [0.0, 4.33, 0.05]
        gwbparsmin = [np.log(0.01), 1.16, 1.0e-7]
        gwbparsmax = [np.log(1.0e4), 6.85, 1.00]
        gwbparstepscale = [0.1, 0.1, 0.01]
        gwbparsvary = [vary, vary, False]

        gwblabel = 'gwb-pow-dipole-00'
        gwbtype = ['powerlaw', 'dipole', '00']
        gwbsource = stochasticsignals.create_group(gwblabel)
        gwbsource.create_dataset("type", data=gwbtype)
        gwbsource.create_dataset("flag", data=gwbflags)
        gwbsource.create_dataset("parname", data=gwbparnames)
        gwbsource.create_dataset("units", data=gwbunits)
        gwbsource.create_dataset("min", data=gwbparsmin)
        gwbsource.create_dataset("max", data=gwbparsmax)
        gwbsource.create_dataset("start", data=gwbparsstart)
        gwbsource.create_dataset("stepscale", data=gwbparstepscale)
        gwbsource.create_dataset("vary", data=gwbparsvary)

        gwblabel = 'gwb-pow-dipole-1-1'
        gwbtype = ['powerlaw', 'dipole', '1-1']
        gwbsource = stochasticsignals.create_group(gwblabel)
        gwbsource.create_dataset("type", data=gwbtype)
        gwbsource.create_dataset("flag", data=gwbflags)
        gwbsource.create_dataset("parname", data=gwbparnames)
        gwbsource.create_dataset("units", data=gwbunits)
        gwbsource.create_dataset("min", data=gwbparsmin)
        gwbsource.create_dataset("max", data=gwbparsmax)
        gwbsource.create_dataset("start", data=gwbparsstart)
        gwbsource.create_dataset("stepscale", data=gwbparstepscale)
        gwbsource.create_dataset("vary", data=gwbparsvary)

        gwblabel = 'gwb-pow-dipole-10'
        gwbtype = ['powerlaw', 'dipole', '10']
        gwbsource = stochasticsignals.create_group(gwblabel)
        gwbsource.create_dataset("type", data=gwbtype)
        gwbsource.create_dataset("flag", data=gwbflags)
        gwbsource.create_dataset("parname", data=gwbparnames)
        gwbsource.create_dataset("units", data=gwbunits)
        gwbsource.create_dataset("min", data=gwbparsmin)
        gwbsource.create_dataset("max", data=gwbparsmax)
        gwbsource.create_dataset("start", data=gwbparsstart)
        gwbsource.create_dataset("stepscale", data=gwbparstepscale)
        gwbsource.create_dataset("vary", data=gwbparsvary)

        gwblabel = 'gwb-pow-dipole-11'
        gwbtype = ['powerlaw', 'dipole', '11']
        gwbsource = stochasticsignals.create_group(gwblabel)
        gwbsource.create_dataset("type", data=gwbtype)
        gwbsource.create_dataset("flag", data=gwbflags)
        gwbsource.create_dataset("parname", data=gwbparnames)
        gwbsource.create_dataset("units", data=gwbunits)
        gwbsource.create_dataset("min", data=gwbparsmin)
        gwbsource.create_dataset("max", data=gwbparsmax)
        gwbsource.create_dataset("start", data=gwbparsstart)
        gwbsource.create_dataset("stepscale", data=gwbparstepscale)
        gwbsource.create_dataset("vary", data=gwbparsvary)

        self.h5file.close()
        self.h5file = None



    """
    Create a model "rgp-ban-legacy", which describes the data as:
    Random Gaussian process, error bars for all observations, efac parameter for
    all system/pulsar combinations, power-law red noise for all pulsars,
    analytic marginalisation over all timing model parameters

    Currently done manually, so not used yet
    """
    def creategwbsearchmodel(self):
        pass


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
The bang-legacy code encodes all the different dipole modes independently.
However, the spectral index is shared between all the modes, and the dipole
modes need to be converted to real parameters (amplitude and c_lm)

This function returns the bang-parameters A00, A1-1, A10, A11, based on the
physical parametes A, and the c_lm's

Input: x is a 4D array, with elements
- A, the GWB amplitude
- c1-1
- c10
- c11

Output: another 4D array, with elements
- A00, A1-1, A10, A11
"""
def dipoleconvertfromreal(x):
    # x is a 4-D 
    return x

"""
Based on the original (wrapper) parameter dictionary, this function returns the
parameter dictionary and indices as the user will see it. Some parameters are
changed, and
"""
def dipoleampindices(origpardict):
    mode = 0
    origindex = 0
    parindex = 0

    wrapdai = np.array([0, 0, 0, 0])    # Wrapper dipole amplitude indices
    userdai = np.array([0, 0, 0, 0])    # User dipole amplitude indices
    wrapdsi = np.array([0, 0, 0, 0])    # Wrapper dipole spectral index indices
    userdsi = np.array([0])             # User dipole spectral index indices
    #pardict = [pardict[i].copy() for i in range(len(pardict))]
    pardict = []
    
    for par in origpardict:
        if par['signaltype'] == 'powerlaw' and par['signalcorrelation'] == 'dipole':
            # It is a dipole GWB, so which parameter is it?
            if par['mode'] == '00':
                # mode 0, 0
                mode = 0
            elif par['mode'] == '1-1':
                # mode 1, -1
                mode = 1
            elif par['mode'] == '10':
                # mode 1, 0
                mode = 2
            elif par['mode'] == '11':
                # mode 1, 1
                mode = 3
            else:
                # This should not happen, since we already checked it
                raise IOError, "Incorrect parameter dictionary (signal %s)" % (par['signal'])

            if par['parname'] == 'amplitude' and par['vary'] > 0:
                pardict.append(par.copy())      # Always copy amplitude
                wrapdai[mode] = origindex
                userdai[mode] = parindex
                parindex += 1

                if mode == 0:
                    pardict[-1]['parname'] = 'Amplitude'
                else:
                    pardict[-1]['parname'] = 'c' + par['mode']
                    pardict[-1]['units'] = ''
                    pardict[-1]['start'] = 0.0
                    pardict[-1]['min'] = -1.0
                    pardict[-1]['max'] = 1.0
                    pardict[-1]['stepscale'] = 0.1
            elif par['parname'] == 'spectral index' and par['vary'] > 0 and mode == 0: 
                pardict.append(par.copy())      # Copy first spectral index
                wrapdsi[mode] = origindex
                userdsi[mode] = parindex
                parindex += 1
            elif par['parname'] == 'spectral index' and par['vary'] > 0 and mode > 0: 
                wrapdsi[mode] = origindex
            else:
                # This should not happen, since we already checked it
                raise IOError, "Incorrect parameter dictionary (signal %s)" % (par['signal'])

        else:
            # No dipole parameter, append without question
            pardict.append(par.copy())
            parindex += 1

        # Increase the parameter number index
        origindex += 1

    return (pardict, wrapdai, userdai, wrapdsi, userdsi)

"""
Based on the original varying parameter dictionary, this function returns
whether or not there is one, and exactly one, dipole signal in the dictionary

Because of the mixing of real and apparent parameters, we will require that this
signal has _exactly_ 8 varying parameters.
"""
def havedipolesignal(pardict):
    dipars = np.zeros(8)  # A00, gamma, A1-1, gamma, A10, gamma, A11, gamma
    ind = np.array([0, 1])

    for par in pardict:
        if par['signaltype'] == 'powerlaw' and par['signalcorrelation'] == 'dipole':
            # It is a dipole GWB, so which parameter is it?
            if par['mode'] == '00':
                # mode 0, 0
                ind = np.array([0, 1])
            elif par['mode'] == '1-1':
                # mode 1, -1
                ind = np.array([2, 3])
            elif par['mode'] == '10':
                # mode 1, 0
                ind = np.array([4, 5])
            elif par['mode'] == '11':
                # mode 1, 1
                ind = np.array([6, 7])
            else:
                return False

            if par['parname'] == 'amplitude' and par['vary'] > 0 and dipars[ind[0]] == 0:
                dipars[ind[0]] = 1
            elif par['parname'] == 'spectral index' and par['vary'] > 0 and dipars[ind[1]] == 0: 
                dipars[ind[1]] = 1
            else:
                return False

    # All parameters must be present and varying
    return (np.sum(dipars) == 8)




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
Mark1: Very basic implementation of the model/likelihood. Although written with
multiple pulsars in mind, only really works for a single pulsar with one efac
and some red noise spectrum.

This class represents all we know about a pulsar

For now, do not bother with different back-end systems. Keep it simple
"""
class mark1Pulsar(object):
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
    Ffreqs = None
    Gr = None
    GGNGG = None
    GNGldet = None
    rGGNGGr = None
    rGGNGGF = None
    FGGNGGF = None

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
        self.Ffreqs = None
        self.Gr = None
        self.GGNGG = None
        self.GNGldet = None
        self.rGGNGGr = None
        self.rGGNGGF = None
        self.FGGNGGF = None

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

    # The number of frequencies is not the number of modes: model = 2*freqs
    def createAuxiliaries(self, nfreqs):
        (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs)
        self.Gr = np.dot(self.Gmat.T, self.residuals)
        N = np.diag(self.toaerrs**2)
        GNG = np.dot(self.Gmat.T, np.dot(N, self.Gmat))
        cf = sl.cho_factor(GNG)
        self.GNGldet = 2*np.sum(np.log(np.diag(cf[0])))
        GNGinv = sl.cho_solve(cf, np.identity(GNG.shape[0]))
        self.GGNGG = np.dot(self.Gmat, np.dot(GNGinv, self.Gmat.T))
        self.rGGNGGr = np.dot(self.residuals, np.dot(self.GGNGG, self.residuals))
        self.rGGNGGF = np.dot(self.Fmat.T, np.dot(self.GGNGG, self.residuals))
        self.FGGNGGF = np.dot(self.Fmat.T, np.dot(self.GGNGG, self.Fmat))



class mark1Likelihood(object):
    m1psrs = []

    dimensions = 0
    pmin = None
    pmax = None
    pstart = None
    pwidth = None
    pamplitudeind = None
    initialised = False

    def __init__(self):
        self.m1psrs = []

        self.dimensions = 0
        self.pmin = None
        self.pmax = None
        self.pstart = None
        self.pwidth = None
        self.pamplitudeind = None
        self.initialised = False

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
            newpsr = mark1Pulsar()
            newpsr.readFromH5(filename, psrname)
            newpsr.createAuxiliaries(45)
            self.m1psrs.append(newpsr)

    def initPrior(self):
        self.dimensions = 0
        for m1psr in self.m1psrs:
            pclength = int(len(m1psr.Ffreqs) / 2)
            self.dimensions += 1 + pclength

        self.pmin = np.zeros(self.dimensions)
        self.pmax = np.zeros(self.dimensions)
        self.pstart = np.zeros(self.dimensions)
        self.pwidth = np.zeros(self.dimensions)

        index = 0
        for m1psr in self.m1psrs:
            # Efac:
            self.pmin[index] = 0.001
            self.pmax[index] = 1000.0
            self.pwidth[index] = 0.1
            self.pstart[index] = 1.0
            index += 1

            # Spectrum coefficients.
            pclength = int(len(m1psr.Ffreqs) / 2)

            # Spectrum coefficients. First produce a least-squares fit for the
            # Fourier components
            cf = sl.cho_factor(m1psr.FGGNGGF)
            FGNGFinv = sl.cho_solve(cf, np.identity(m1psr.FGGNGGF.shape[0]))
            fest = np.dot(FGNGFinv, m1psr.rGGNGGF)

            # Produce the corresponding spectral estimates
            psest = np.zeros(pclength)
            for ii in range(pclength):
                psest[ii] = np.log10(fest[2*ii]**2 + fest[2*ii+1]**2)

            self.pmin[index:index+pclength] = psest - 20.0
            self.pmax[index:index+pclength] = psest + 10.0
            self.pmin[index:index+pclength] = -50.0
            self.pmax[index:index+pclength] = 0
            self.pstart[index:index+pclength] = psest
            self.pwidth[index:index+pclength] = 0.1

            index += pclength


            """
            A = np.dot(m1psr.Gmat.T, m1psr.Fmat)
            B = np.dot(m1psr.Fmat.T, m1psr.Fmat)
            E = np.dot(A.T, A)
            Ei = np.array(np.mat(E).I)
            P = np.dot(m1psr.Gmat, m1psr.Gmat.T)
            P4 = np.dot(m1psr.Gmat, m1psr.Gmat.T)
            B4 = np.dot(m1psr.Fmat.T, np.dot(P4, m1psr.Fmat))
            fest2 = np.dot(Ei, np.dot(A.T, np.dot(m1psr.Gmat.T, m1psr.residuals)))
            fest3 = np.dot(np.array(np.mat(B).I), np.dot(m1psr.Fmat.T, m1psr.residuals))
            fest4 = np.dot(np.array(np.mat(B4).I), np.dot(m1psr.Fmat.T, np.dot(P4, m1psr.residuals)))
            """

            """
            F, Ffreqs = fourierdesignmatrix(m1psr.toas, 2*int(len(m1psr.toas)/2))
            FF = np.dot(F.T, F)
            model = (m1psr.toas, m1psr.residuals, m1psr.toaerrs, m1psr.Mmat, m1psr.Gmat, np.array([[1]]), [len(m1psr.toas)], [m1psr.Mmat.shape[1]], [m1psr.Gmat.shape[1]], None, None, None)
            alpha = -2.0/3.0
            Cgw = Cgw_sec(model, alpha=alpha)
            GCG = np.dot(m1psr.Gmat.T, np.dot(Cgw, m1psr.Gmat))
            ratio = -2.51183978438e+89
            pcs = Ffreqs ** (3 - 2*alpha) * ratio
            Cf = np.dot(F, np.dot(np.diag(pcs), F.T))
            GCfG = np.dot(m1psr.Gmat.T, np.dot(Cf, m1psr.Gmat))
            print "Traces:", np.trace(GCG), np.trace(GCfG)
            print "Ratio:", np.trace(GCG) / np.trace(GCfG)
            """

            """
            fig = plt.figure()
            ufreqs = np.log(np.array(list(sets.Set(m1psr.Ffreqs))))
            plt.errorbar(ufreqs, psest, fmt='o', c='blue')
            #plt.errorbar(m1psr.Ffreqs, fest**2, fmt='o', c='blue')
            #plt.errorbar(m1psr.Ffreqs, fest2**2, fmt='.', c='red')
            #plt.errorbar(m1psr.Ffreqs, fest3**2, fmt='.', c='green')
            #plt.errorbar(m1psr.Ffreqs, fest4**2, fmt='.', c='black')
            plt.title("Periodogram")
            plt.xlabel("Frequency [log(f)]")
            plt.ylabel("Power [log(r)]")
            plt.grid(True)
            plt.show()
            """

            """
            # Plot the reconstructed signal
            recsig = np.dot(m1psr.Gmat, np.dot(m1psr.Gmat.T, np.dot(m1psr.Fmat, fest)))
            plt.errorbar(m1psr.toas, m1psr.residuals, yerr=m1psr.toaerrs, fmt='.', c='blue')
            plt.plot(m1psr.toas, recsig, 'r--')
            plt.grid(True)
            plt.show()
            """
            
            """
            # Plot the Fourier basis functions
            Proj = np.dot(m1psr.Gmat, m1psr.Gmat.T)
            Fproj = np.dot(Proj, m1psr.Fmat)
            plt.plot(m1psr.toas, Fproj[:,0], 'k-')
            plt.plot(m1psr.toas, m1psr.Fmat[:,0], 'k--')
            plt.plot(m1psr.toas, Fproj[:,1], 'r-')
            plt.plot(m1psr.toas, m1psr.Fmat[:,1], 'r--')
            plt.plot(m1psr.toas, Fproj[:,2], 'b-')
            plt.plot(m1psr.toas, m1psr.Fmat[:,2], 'b--')
            plt.plot(m1psr.toas, Fproj[:,3], 'g-')
            plt.plot(m1psr.toas, m1psr.Fmat[:,3], 'g--')
            plt.grid(True)
            plt.show()
            """

            """
            # Plot the basis functions of the compression
            psc = m1psr.Ffreqs ** (-13.0/3.0)
            GF = np.dot(m1psr.Gmat.T, m1psr.Fmat)
            GCps = np.dot(GF, np.dot(np.diag(psc), GF.T))
            U, s, Vh = sl.svd(GCps)
            bfuncs = np.dot(m1psr.Gmat, U)

            Proj = np.dot(m1psr.Gmat, m1psr.Gmat.T)
            Fproj = np.dot(Proj, m1psr.Fmat)
            plt.plot(m1psr.toas, bfuncs[:,0], 'k-')
            plt.plot(m1psr.toas, bfuncs[:,1], 'k--')
            plt.plot(m1psr.toas, bfuncs[:,2], 'k.')
            plt.grid(True)
            plt.show()
            """



    """
    Parameters are: efac, coefficients (for now)
    """
    def mark1loglikelihood(self, parameters):
        # First figure out what the parameters are
        efacs = []
        pcoefs = []
        index = 0
        slength = 0
        for m1psr in self.m1psrs:
            pclength = int(len(m1psr.Ffreqs) / 2)
            if len(parameters) < index + 1 + pclength:
                raise ValueError('ERROR:len(parameters) too small in loglikelihood')

            efacs.append(parameters[index])
            index += 1

            # The power spectrum coefficients enter the matrix twice: for both
            # sine and cosine basis functions.
            pcdoubled = np.array([parameters[index:index+pclength], parameters[index:index+pclength]]).T.flatten()
            #pcoefs.append(np.array(parameters[index:index+pclength]))
            pcoefs.append(pcdoubled)
            index += pclength

            slength += 2*pclength

        # Form the total correlation matrices
        Sigma = np.zeros((slength, slength))
        Phi = np.zeros((slength, slength))
        rGF = np.zeros(slength)
        rGr = np.zeros(len(self.m1psrs))
        GNGldet = np.zeros(len(self.m1psrs))
        index = 0
        for ii in range(len(self.m1psrs)):
            pclength = int(len(self.m1psrs[ii].Ffreqs) / 2)
            rGr[ii] = self.m1psrs[ii].rGGNGGr / efacs[ii] 
            rGF[index:index+2*pclength] = self.m1psrs[ii].rGGNGGF / efacs[ii]
            GNGldet[ii] = self.m1psrs[ii].GNGldet + np.log(efacs[ii]) * self.m1psrs[ii].Gmat.shape[1]

            # Fill the single pulsar part of Sigma
            Sigma[index:index+2*pclength, index:index+2*pclength] = self.m1psrs[ii].FGGNGGF / efacs[ii]

            # Add the spectrum coefficients to the diagonal indices
            di = np.diag_indices(2*pclength)
            Sigma[index:index+2*pclength, index:index+2*pclength][di] += 10**(-pcoefs[ii])
            Phi[index:index+2*pclength, index:index+2*pclength][di] = 10**pcoefs[ii]

            index += 2*pclength

        # Calculate the Cholesky decomposition of the correlation matrix
        cf = sl.cho_factor(Sigma)

        # The dot product involving Sigma
        rGPhiGr = np.dot(rGF, sl.cho_solve(cf, rGF))

        # Log of the determinant
        SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))

        # At the moment, Phi is diagonal. In the future it won't be (when GWB is
        # included). At that point, calculate this guy earlier, and add to Sp
        # above.
        PhiLD = np.sum(np.log(np.diag(Phi)))

        return -0.5*np.sum(rGr) - 0.5*np.sum(GNGldet) + 0.5*rGPhiGr - 0.5*PhiLD  - 0.5*SigmaLD

    def loglikelihood(self, parameters):
        ll = 0.0

        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            ll = self.mark1loglikelihood(parameters)
        else:
            ll = -1e99

        return ll

    def logposterior(self, parameters):
        return self.loglikelihood(parameters)

    def nlogposterior(self, parameters):
        return - self.loglikelihood(parameters)


"""
A general signal element of the mark2 model/likelihood.

This replaces most auxiliary quantities present in mark1Pulsar, and can be used
for any type of stochastic signal (efac/equad and spectral noise)

Henceforth, the Fmat fourier design matrices are assumed to be for identical
frequencies for all pulsars.
"""
class mark2signal(object):
    pulsarind = None        # pulsar nr. for EFAC/EQUAD
    stype = "none"          # EFAC, EQUAD, spectrum, powerlaw
    corr = "single"         # single, gr, uni, dipole...

    npars = 0               # Number of parameters
    nindex = 0              # index in total parameters array

    # Quantities for EFAC/EQUAD
    GNG = None

    # If this pulsar has only one efac/equad parameter, use computational
    # shortcut, using:
    Naccel = False
    GGNGG = None
    GNGldet = None
    rGGNGGr = None
    rGGNGGF = None
    FGGNGGF = None

    # Quantities for spectral noise
    Tmax = None
    hdmat = None


    



"""
Mark2: Basic implementation of the model/likelihood, based on the frequency
models as outlined in Lentati et al. (2013). Can handle multiple EFACs per
pulsar, an EQUAD, and general red noise spectra.
"""
class mark2Pulsar(object):
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
    Ffreqs = None
    Gr = None
    GtF = None

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
        self.Ffreqs = None
        self.Gr = None
        self.GtF = None

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

    # The number of frequencies is not the number of modes: model = 2*freqs
    def createAuxiliaries(self, nfreqs, Tmax):
        (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
        self.Gr = np.dot(self.Gmat.T, self.residuals)
        self.GtF = np.dot(self.Gmat.T, self.Fmat)

"""
with n the number of pulsars, return an nxn matrix representing the H&D
correlation matrix
"""
def mark2hdmat(m2psrs):
    """ Constructs a correlation matrix consisting of the Hellings & Downs
        correlation coefficients. See Eq. (A30) of Lee, Jenet, and
        Price ApJ 684:1304 (2008) for details.

        @param: list of mark2Pulsar objects
        
    """
    npsrs = len(m2psrs)
    
    raj = [m2psrs[i].raj[0] for i in range(npsrs)]
    decj = [m2psrs[i].decj[0] for i in range(npsrs)]
    pp = np.array([np.cos(decj)*np.cos(raj), np.cos(decj)*np.sin(raj), np.sin(decj)]).T
    cosp = np.array([[np.dot(pp[i], pp[j]) for i in range(npsrs)] for j in range(npsrs)])
    cosp[cosp > 1.0] = 1.0
    xp = 0.5 * (1 - cosp)

    old_settings = np.seterr(all='ignore')
    logxp = 1.5 * xp * np.log(xp)
    np.fill_diagonal(logxp, 0)
    np.seterr(**old_settings)
    hdmat = logxp - 0.25 * xp + 0.5 + 0.5 * np.diag(np.ones(npsrs))

    if False: # Plot the H&D curve
        angle = np.arccos(cosp)
        x = np.array(angle.flat)
        y = np.array(hdmat.flat)
        ind = np.argsort(x)
        plt.plot(x[ind], y[ind], c='b', marker='.')

    return hdmat




class mark2Likelihood(object):
    m2psrs = []

    dimensions = 0
    pmin = None
    pmax = None
    pstart = None
    pwidth = None
    pamplitudeind = None
    initialised = False

    # The model/signals description
    m2signals = []

    def __init__(self):
        self.m2psrs = []

        self.dimensions = 0
        self.pmin = None
        self.pmax = None
        self.pstart = None
        self.pwidth = None
        self.pamplitudeind = None
        self.initialised = False

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
            newpsr = mark2Pulsar()
            newpsr.readFromH5(filename, psrname)
            self.m2psrs.append(newpsr)

    # Initialise the model
    def initModel(self, nfreqmodes=20):
        # For every pulsar, construct the auxiliary quantities like the Fourier
        # design matrix etc
        if len(self.m2psrs) < 1:
            raise IOError, "no pulsars found in hdf5 file"

        Tstart = np.min(self.m2psrs[0].toas)
        Tfinish = np.max(self.m2psrs[0].toas)

        for m2psr in self.m2psrs:
            Tstart = np.min([np.min(self.m2psrs[0].toas), Tstart])
            Tfinish = np.max([np.max(self.m2psrs[0].toas), Tfinish])

        # Total duration of the experiment
        Tmax = Tfinish - Tstart
        for m2psr in self.m2psrs:
            m2psr.createAuxiliaries(nfreqmodes, Tmax)

        # Initialise the mark2signal objects
        # Currently: one efac per pulsar, and red noise
        self.m2signals = []
        index = 0
        for ii in range(len(self.m2psrs)):
            # Create an efac signal
            newsignal = mark2signal()
            newsignal.pulsarind = ii
            newsignal.stype = 'efac'
            newsignal.corr = 'single'
            N = np.diag(self.m2psrs[ii].toaerrs**2)
            newsignal.GNG = np.dot(self.m2psrs[ii].Gmat.T, np.dot(N, self.m2psrs[ii].Gmat))
            newsignal.npars = 0
            newsignal.nindex = index
            index += newsignal.npars

            # Create the computational-shortcut stuff
            newsignal.Naccel = True
            cf = sl.cho_factor(newsignal.GNG)
            newsignal.GNGldet = 2*np.sum(np.log(np.diag(cf[0])))
            GNGinv = sl.cho_solve(cf, np.identity(newsignal.GNG.shape[0]))
            newsignal.GGNGG = np.dot(self.m2psrs[ii].Gmat, np.dot(GNGinv, self.m2psrs[ii].Gmat.T))
            newsignal.rGGNGGr = np.dot(self.m2psrs[ii].residuals, np.dot(newsignal.GGNGG, self.m2psrs[ii].residuals))
            newsignal.rGGNGGF = np.dot(self.m2psrs[ii].Fmat.T, np.dot(newsignal.GGNGG, self.m2psrs[ii].residuals))
            newsignal.FGGNGGF = np.dot(self.m2psrs[ii].Fmat.T, np.dot(newsignal.GGNGG, self.m2psrs[ii].Fmat))

            self.m2signals.append(newsignal)

            """
            # Create a spectral signal
            newsignal = mark2signal()
            newsignal.pulsarind = ii
            newsignal.stype = 'spectrum'
            newsignal.corr = 'single'
            newsignal.npars = int(len(self.m2psrs[ii].Ffreqs)/2)
            newsignal.Tmax = Tmax
            newsignal.nindex = index
            self.m2signals.append(newsignal)
            index += newsignal.npars
            """

        # Now include a GWB
        newsignal = mark2signal()
        newsignal.pulsarind = -1
        newsignal.stype = 'spectrum'
        #newsignal.stype = 'powerlaw'
        newsignal.corr = 'gr'
        newsignal.npars = int(len(self.m2psrs[0].Ffreqs)/2)     # Check how many
        #newsignal.npars = 2
        newsignal.nindex = index
        newsignal.Tmax = Tmax
        newsignal.hdmat = mark2hdmat(self.m2psrs)           # The H&D matrix
        self.m2signals.append(newsignal)
        index += newsignal.npars

    def initPrior(self):
        self.dimensions = 0
        for m2signal in self.m2signals:
            self.dimensions += m2signal.npars

        self.pmin = np.zeros(self.dimensions)
        self.pmax = np.zeros(self.dimensions)
        self.pstart = np.zeros(self.dimensions)
        self.pwidth = np.zeros(self.dimensions)

        index = 0
        for m2signal in self.m2signals:
            if m2signal.stype == 'efac':
                if m2signal.npars == 1:
                    self.pmin[index] = 0.001
                    self.pmax[index] = 1000.0
                    self.pwidth[index] = 0.1
                    self.pstart[index] = 1.0
                    index += m2signal.npars
            elif m2signal.stype == 'equad':
                self.pmin[index] = -10.0        # 1e-10 secs
                self.pmax[index] = -2           # 1e-2 secs
                self.pwidth[index] = 0.1
                self.pstart[index] = -8.0
                index += m2signal.npars
            elif m2signal.stype == 'spectrum':
                npars = m2signal.npars
                self.pmin[index:index+npars] = -25.0
                self.pmax[index:index+npars] = 10.0
                self.pstart[index:index+npars] = -10.0
                self.pwidth[index:index+npars] = 0.1
                index += m2signal.npars
            elif m2signal.stype == 'powerlaw':
                self.pmin[index:index+2] = [-17.0, 1.02]
                self.pmax[index:index+2] = [-10.0, 6.98]
                self.pstart[index:index+2] = [-15.0, 2.01]
                self.pwidth[index:index+2] = [0.1, 0.1]
                index += m2signal.npars


    """
    Parameters are: efac, coefficients (for now)

    # For every pulsar, we assume that the power spectrum frequencies are
    # the same (if included)
    """
    def m2loglikelihood(self, parameters):
        # Calculating the log-likelihood happens in several steps. Here, we are
        # going to assume computational tricks for all efac/equad signals. It
        # 'll take too long otherwise.
        # Single-pulsar spectra could be added though.

        # For all pulsars, we will need the quantities:
        # rGGNGGr, rGGNGGF, FGGNGGF and Phi

        # For the total, we will construct the full matrix Sigma from these.
        # After that, the log-likelihood can be calculated

        # First figure out how large we have to make the arrays
        npsrs = len(self.m2psrs)
        npf = np.zeros(npsrs, dtype=np.int)
        for ii in range(npsrs):
            npf[ii] = len(self.m2psrs[ii].Ffreqs)

        # Define the total arrays
        rGr = np.zeros(npsrs)
        rGF = np.zeros(np.sum(npf))
        FGGNGGF = np.zeros((np.sum(npf), np.sum(npf)))
        Phi = np.zeros((np.sum(npf), np.sum(npf)))
        Sigma = np.zeros((np.sum(npf), np.sum(npf))) 
        GNGldet = np.zeros(npsrs)
        
        # Loop over all signals, and fill the above arrays
        for m2signal in self.m2signals:
            if m2signal.stype == 'efac':
                pefac = 1.0
                if m2signal.npars == 1:
                    pefac = parameters[m2signal.nindex]

                findex = np.sum(npf[:m2signal.pulsarind])
                nfreq = npf[m2signal.pulsarind]/2

                rGr[m2signal.pulsarind] = m2signal.rGGNGGr / pefac
                rGF[findex:findex+2*nfreq] = m2signal.rGGNGGF / pefac
                GNGldet[m2signal.pulsarind] = m2signal.GNGldet + np.log(pefac) * self.m2psrs[m2signal.pulsarind].Gmat.shape[1]
                FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] += m2signal.FGGNGGF / pefac
            elif m2signal.stype == 'equad':
                # TODO: Change this from an efac, now is the same
                findex = np.sum(npf[:m2signal.pulsarind])
                nfreq = npf[m2signal.pulsarind]/2

                rGr[m2signal.pulsarind] = m2signal.rGGNGGr / parameters[m2signal.nindex]
                rGF[findex:findex+2*nfreq] = m2signal.rGGNGGF / parameters[m2signal.nindex]
                GNGldet[m2signal.pulsarind] = m2signal.GNGldet + np.log(parameters[m2signal.nindex]) * self.m2psrs[m2signal.pulsarind].Gmat.shape[1]
                FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] += m2signal.FGGNGGF / parameters[m2signal.nindex]
            elif m2signal.stype == 'spectrum':
                if m2signal.corr == 'single':
                    findex = np.sum(npf[:m2signal.pulsarind])
                    nfreq = npf[m2signal.pulsarind]/2

                    # Pcdoubled is an array where every element of the parameters
                    # of this m2signal is repeated once (e.g. [1, 1, 3, 3, 2, 2, 5, 5, ...]
                    pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()

                    # Fill the phi matrix
                    di = np.diag_indices(2*nfreq)
                    Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += 10**pcdoubled
                elif m2signal.corr == 'gr':
                    nfreq = m2signal.npars

                    pcdoubled = np.array([parameters[m2signal.nindex:m2signal.nindex+m2signal.npars], parameters[m2signal.nindex:m2signal.nindex+m2signal.npars]]).T.flatten()

                    indexa = 0
                    indexb = 0
                    for aa in range(npsrs):
                        for bb in range(npsrs):
                            # Some pulsars may have fewer frequencies than
                            # others (right?). So only use overlapping ones
                            nof = np.min([npf[aa], npf[bb], 2*nfreq])
                            di = np.diag_indices(nof)
                            Phi[indexa:indexa+nof,indexb:indexb+nof][di] += 10**pcdoubled[:nof] * m2signal.hdmat[aa, bb]
                            indexb += npf[bb]
                        indexb = 0
                        indexa += npf[aa]
            elif m2signal.stype == 'powerlaw':
                spd = 24 * 3600.0
                spy = 365.25 * spd
                Amp = 10**parameters[m2signal.nindex]
                Si = parameters[m2signal.nindex+1]

                if m2signal.corr == 'single':
                    findex = np.sum(npf[:m2signal.pulsarind])
                    nfreq = npf[m2signal.pulsarind]/2
                    freqpy = self.m2psrs[m2signal.pulsarind].Ffreqs * spy
                    pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)

                    # Fill the phi matrix
                    di = np.diag_indices(2*nfreq)
                    Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                elif m2signal.corr == 'gr':
                    freqpy = self.m2psrs[0].Ffreqs * spy
                    pcdoubled = (Amp**2 * spy**3 / (12*np.pi*np.pi * m2signal.Tmax)) * freqpy ** (-Si)
                    nfreq = len(freqpy)

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
                            Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * m2signal.hdmat[aa, bb]
                            indexb += npf[bb]
                        indexb = 0
                        indexa += npf[aa]
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi
        cf = sl.cho_factor(Phi)
        PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
        Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))

        # Construct and decompose Sigma
        Sigma = FGGNGGF + Phiinv
        cf = sl.cho_factor(Sigma)
        SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
        rGSigmaGr = np.dot(rGF, sl.cho_solve(cf, rGF))

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(rGr) - 0.5*np.sum(GNGldet) + 0.5*rGSigmaGr - 0.5*PhiLD - 0.5*SigmaLD

    def loglikelihood(self, parameters):
        ll = 0.0

        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            ll = self.m2loglikelihood(parameters)
        else:
            ll = -1e99

        return ll

    def logposterior(self, parameters):
        return self.loglikelihood(parameters)

    def nlogposterior(self, parameters):
        return - self.loglikelihood(parameters)




"""
Given a collection of samples, return the 2-sigma confidence intervals
samples: an array of samples
sigmalevel: either 1, 2, or 3. Which sigma limit must be given
onesided: Give one-sided limits (useful for setting upper or lower limits)

"""
def confinterval(samples, sigmalevel=2, onesided=False):
  # The probabilities for different sigmas
  sigma = [0.68268949, 0.95449974, 0.99730024]

  # Create the ecdf function
  ecdf = sm.distributions.ECDF(samples)

  # Create the binning
  x = np.linspace(min(samples), max(samples), 200)
  y = ecdf(x)

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
    Aing = 5.0e-14
    yinj = (Aing**2 * spy**3 / (12*np.pi*np.pi * (5*spy))) * ((pfreqs * spy) ** (-13.0/3.0))
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
Run a simple Metropolis-Hastings algorithm on the likelihood wrapper.
Implementation from "emcee"

Starting position can be taken from initfile (just like emcee), and if
covest=True, this file will be used to estimate the stepsize for the mcmc

"""
def RunMetropolis(likob, steps, chainfilename, initfile=None, resize=0.088):
  ndim = likob.dimensions
  pwidth = likob.pwidth

  if initfile is not None:
    # Read the starting position of the random walkers from a file
    print "Obtaining initial positions from '" + initfile + "'"
    burnindata = np.loadtxt(initfile)
    burnindata = burnindata[:,2:]
    nsteps = burnindata.shape[0]
    dim = burnindata.shape[1]
    if(ndim is not dim):
      print "ERROR: burnin file not same dimensions!"
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
  for i in range(steps/nSkip):
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


