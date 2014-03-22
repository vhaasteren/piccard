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
- libstempo:    pip install libstempo (optional, required for creating HDF5
                files, and for non-linear timing model analysis
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
import json
import tempfile

from . import anisotropygammas as ang  # Internal module
from .triplot import *

try:    # If without libstempo, can still read hdf5 files
    import libstempo
    t2 = libstempo
except ImportError:
    t2 = None

# In order to keep the dictionary in order
try:
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict
except ImportError:
    OrderedDict = dict


# Some constants used in Piccard
# For DM calculations, use this constant
# See You et al. (2007) - http://arxiv.org/abs/astro-ph/0702366
# Lee et al. (in prep.) - ...
# Units here are such that delay = DMk * DM * freq^-2 with freq in MHz
pic_DMk = 4.15e3        # Units MHz^2 cm^3 pc sec

pic_spd = 86400.0       # Seconds per day
#pic_spy = 31556926.0   # Wrong definition of YEAR!!!
pic_spy =  31557600.0   # Seconds per year (yr = 365.25 days, so Julian years)
pic_T0 = 53000.0        # MJD to which all HDF5 toas are referenced



"""
The DataFile class is the class that supports the HDF5 file format. All HDF5
file interactions happen in this class.
"""
class DataFile(object):
    filename = None
    h5file = None

    """
    Initialise the structure.

    @param filename:    name of the HDF5 file
    """
    def __init__(self, filename=None):
        # Open the hdf5 file?
        self.filename = filename

    def __del__(self):
        # Delete the instance, and close the hdf5 file?
        pass

    """
    Return a list of pulsars present in the HDF5 file
    """
    def getPulsarList(self):
        # 'r' means: read file, must exist
        self.h5file = h5.File(self.filename, 'r')
        psrlist = list(self.h5file)
        self.h5file.close()

        return psrlist

    """
    Obtain the hdf5 group of pulsar psrname, create it if it does not exist. If
    delete is toggled, delete the content first. This function assumes the hdf5
    file is opened (not checked)

    @param psrname: The name of the pulsar
    @param delete:  If True, the pulsar group will be emptied before use
    """
    def getPulsarGroup(self, psrname, delete=False):
        # datagroup = h5file.require_group('Data')

        if psrname in self.h5file and delete:
            del self.h5file[psrname]

        pulsarGroup = self.h5file.require_group(psrname)

        return pulsarGroup

    """
    Add data to a specific pulsar. Here the hdf5 file is opened, and the right
    group is selected

    @param psrname:     The name of the pulsar
    @param field:       The name of the field we will be writing to
    @param data:        The data we are writing to the field
    @param overwrite:   Whether the data should be overwritten if it exists
    """
    def addData(self, psrname, field, data, overwrite=True):
        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')

        psrGroup = self.getPulsarGroup(psrname, delete=False)
        self.writeData(psrGroup, field, data, overwrite=overwrite)

        self.h5file.close()
        self.h5file = None

        

    """
    Read data from a specific pulsar. If the data is not available, the hdf5
    file is properly closed, and an exception is thrown

    @param psrname:     Name of the pulsar we are reading data from
    @param field:       Field name of the data we are requestion
    @param subgroup:    If the data is in a subgroup, get it from there
    @param dontread:    If set to true, do not actually read anything
    @param required:    If not required, do not throw an exception, but return
                        'None'
    """
    def getData(self, psrname, field, subgroup=None, \
            dontread=False, required=True):
        # Dontread is useful for readability in the 'readPulsarAuxiliaries
        if dontread:
            return None

        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # 'r' means: read file, must exist
        self.h5file = h5.File(self.filename, 'r')
        psrGroup = self.getPulsarGroup(psrname, delete=False)

        datGroup = psrGroup
        if subgroup is not None:
            if subgroup in psrGroup:
                datGroup = psrGroup[subgroup]
            else:
                self.h5file.close()
                if required:
                    raise IOError, "Field {0} not present for pulsar {1}/{2}".format(field, psrname, subgroup)

        if field in datGroup:
            data = np.array(datGroup[field])
            self.h5file.close()
        else:
            self.h5file.close()
            if required:
                raise IOError, "Field {0} not present for pulsar {1}".format(field, psrname)
            else:
                data = None

        return data

    """
    Retrieve the shape of a specific dataset

    @param psrname:     Name of the pulsar we are reading data from
    @param field:       Field name of the data we are requestion
    @param subgroup:    If the data is in a subgroup, get it from there
    """
    def getShape(self, psrname, field, subgroup=None):
        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # 'r' means: read file, must exist
        self.h5file = h5.File(self.filename, 'r')
        psrGroup = self.getPulsarGroup(psrname, delete=False)

        datGroup = psrGroup
        if subgroup is not None:
            if subgroup in psrGroup:
                datGroup = psrGroup[subgroup]
            else:
                self.h5file.close()
                raise IOError, "Field {0} not present for pulsar {1}/{2}".format(field, psrname, subgroup)

        if field in datGroup:
            shape = datGroup[field].shape
            self.h5file.close()
        else:
            self.h5file.close()
            raise IOError, "Field {0} not present for pulsar {1}".format(field, psrname)

        return shape


    """
    (Over)write a field of data for a specific pulsar/group. Data group is
    required, instead of a name.

    @param dataGroup:   Group object
    @param field:       Name of field that we are writing to
    @param data:        The data that needs to be written
    @param overwrite:   If True, data will be overwritten (default True)
    """
    def writeData(self, dataGroup, field, data, overwrite=True):
        if field in dataGroup and overwrite:
            del dataGroup[field]

        if not field in dataGroup:
            try:
                dataGroup.create_dataset(field, data=data)
            except ValueError:
                print("WARNING: h5py too old to support empty arrays: {0}".
                        format(field))

    """
    Add a pulsar to the HDF5 file, given a tempo2 par and tim file. No extra
    model matrices and auxiliary variables are added to the HDF5 file. This
    function interacts with the libstempo Python interface to Tempo2

    @param parfile:     Name of tempo2 parfile
    @param timfile:     Name of tempo2 timfile
    @param iterations:  Number of fitting iterations to do before writing
    @param mode:        Can be replace/overwrite/new. Replace first deletes the
                        entire pulsar group. Overwrite overwrites all data, but
                        does not delete the auxiliary fields. New requires the
                        pulsar not to exist, and throws an exception otherwise.
    """
    def addTempoPulsar(self, parfile, timfile, iterations=1, mode='replace'):
        # Check whether the two files exist
        if not os.path.isfile(parfile) or not os.path.isfile(timfile):
            raise IOError, "Cannot find parfile (%s) or timfile (%s)!" % (parfile, timfile)

        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # Parse the default write behaviour
        deletepsr = False
        if mode == 'replace':
            deletepsr = True
        overwrite = False
        if mode == 'overwrite':
            overwrite = True

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')
        
        # Obtain the directory name of the timfile, and change to it
        timfiletup = os.path.split(timfile)
        dirname = timfiletup[0]
        reltimfile = timfiletup[-1]
        relparfile = os.path.relpath(parfile, dirname)
        savedir = os.getcwd()

        # Change directory to the base directory of the tim-file to deal with
        # INCLUDE statements in the tim-file
        os.chdir(dirname)

        # Load pulsar data from the libstempo library
        t2pulsar = t2.tempopulsar('./'+relparfile, './'+reltimfile)

        # Load the entire par-file into memory, so that we can save it in the
        # HDF5 file
        with open(relparfile, 'r') as content_file:
            parfile_content = content_file.read()

        # Save the tim-file to a temporary file (so that we don't have to deal
        # with 'include' statements in the tim-file), and load that tim-file in
        # memory for HDF5 storage
        tempfilename = tempfile.mktemp()
        t2pulsar.savetim(tempfilename)
        with open(tempfilename, 'r') as content_file:
            timfile_content = content_file.read()
        os.remove(tempfilename)

        # Change directory back to where we were
        os.chdir(savedir)

        # Get the pulsar group
        psrGroup = self.getPulsarGroup(t2pulsar.name, delete=deletepsr)

        # Save the par-file and the tim-file to the HDF5 file
        self.writeData(psrGroup, 'parfile', parfile_content, overwrite=overwrite)
        self.writeData(psrGroup, 'timfile', timfile_content, overwrite=overwrite)

        # Iterate the fitting a few times if necessary
        if iterations > 1:
            t2pulsar.fit(iters=iterations)

        self.writeData(psrGroup, 'TOAs', np.double(np.array(t2pulsar.toas())-pic_T0)*pic_spd, overwrite=overwrite)    # Seconds
        self.writeData(psrGroup, 'prefitRes', np.double(t2pulsar.prefit.residuals), overwrite=overwrite)  # Seconds
        self.writeData(psrGroup, 'postfitRes', np.double(t2pulsar.residuals()), overwrite=overwrite)  # Seconds
        self.writeData(psrGroup, 'toaErr', np.double(1e-6*t2pulsar.toaerrs), overwrite=overwrite)    # Seconds
        self.writeData(psrGroup, 'freq', np.double(t2pulsar.freqs), overwrite=overwrite)    # MHz

        # TODO: writing the design matrix should be done irrespective of the fitting flag
        desmat = t2pulsar.designmatrix(fixunits=True)
        self.writeData(psrGroup, 'designmatrix', desmat, overwrite=overwrite)

        # Write the unit conversions for the design matrix (to timing model
        # parameters
        #unitConversion = t2pulsar.getUnitConversion()
        #self.writeData(psrGroup, 'unitConversion', unitConversion, overwrite=overwrite)

        # Do not write the (co)G-matrix anymore
        # U, s, Vh = sl.svd(desmat)
        # self.writeData(psrGroup, 'Gmatrix', U[:, desmat.shape[1]:], overwrite=overwrite)
        # self.writeData(psrGroup, 'coGmatrix', U[:, :desmat.shape[1]], overwrite=overwrite)

        # Now obtain and write the timing model parameters
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

        self.writeData(psrGroup, 'tmp_name', tmpname, overwrite=overwrite)          # TMP name
        self.writeData(psrGroup, 'tmp_valpre', tmpvalpre, overwrite=overwrite)      # TMP pre-fit value
        self.writeData(psrGroup, 'tmp_valpost', tmpvalpost, overwrite=overwrite)    # TMP post-fit value
        self.writeData(psrGroup, 'tmp_errpre', tmperrpre, overwrite=overwrite)      # TMP pre-fit error
        self.writeData(psrGroup, 'tmp_errpost', tmperrpost, overwrite=overwrite)    # TMP post-fit error

        # Get the flag group for this pulsar. Create if not there
        flagGroup = psrGroup.require_group('Flags')

        # Obtain the unique flags in this dataset, and write to file
        uflags = list(set(t2pulsar.flags))
        for flagid in uflags:
            self.writeData(flagGroup, flagid, t2pulsar.flags[flagid], overwrite=overwrite)

        if not "efacequad" in flagGroup:
            # Check if the sys-flag is present in this set. If it is, add an
            # efacequad flag with pulsarname+content of the sys-flag. If it
            # isn't, check for a be-flag and try the same. Otherwise, add an
            # efacequad flag with the pulsar name as it's elements.
            efacequad = []
            nobs = len(t2pulsar.toas())
            pulsarname = map(str, [t2pulsar.name] * nobs)

            if "group" in flagGroup:
                efacequad = map('-'.join, zip(pulsarname, flagGroup['group']))
            elif "f" in flagGroup:
                efacequad = map('-'.join, zip(pulsarname, flagGroup['f']))
            elif "sys" in flagGroup:
                efacequad = map('-'.join, zip(pulsarname, flagGroup['sys']))
            else:
                efacequad = pulsarname

            self.writeData(flagGroup, "efacequad", efacequad, overwrite=overwrite)

        if not "pulsarname" in flagGroup:
            nobs = len(t2pulsar.toas())
            pulsarname = map(str, [t2pulsar.name] * nobs)
            self.writeData(flagGroup, "pulsarname", pulsarname, overwrite=overwrite)

        # Close the HDF5 file
        self.h5file.close()
        self.h5file = None

    """
    Add pulsars to the HDF5 file, given the name of another hdf5 file and a list
    of pulsars. The data structures will be directly copied from the source file
    to this one.

    @param h5file:  The name of the other HDF5 file from which we will be adding
    @param pulsars: Which pulsars to read ('all' = all, otherwise provide a
                    list: ['J0030+0451', 'J0437-4715', ...])
                    WARNING: duplicates are _not_ checked for.
    @param mode:    Whether to just add, or overwrite (add/replace)
    """
    def addH5Pulsar(self, h5file, pulsars='all', mode='add'):
        # 'a' means: read/write if exists, create otherwise, 'r' means read
        sourceh5 = h5.File(h5file, 'r')
        self.h5file = h5.File(self.filename, 'a')

        # The pulsar names in the HDF5 files
        sourcepsrnames = list(sourceh5)
        destpsrnames = list(self.h5file)

        # Determine which pulsars we are reading in
        readpsrs = []
        if pulsars=='all':
            readpsrs = sourcepsrnames
        else:
            # Check if all provided pulsars are indeed in the HDF5 file
            if np.all(np.array([pulsars[ii] in destpsrnames for ii in range(len(pulsars))]) == True):
                readpsrs = pulsars
            elif pulsars in destpsrnames:
                pulsars = [pulsars]
                readpsrs = pulsars
            else:
                self.h5file.close()
                sourceh5.close()
                raise ValueError("ERROR: Not all provided pulsars in HDF5 file")

        # Check that these pulsars are not already in the current HDF5 file
        if not np.all(np.array([readpsrs[ii] not in destpsrnames for ii in range(len(readpsrs))]) == True) and \
                mode != 'replace':
            self.h5file.close()
            sourceh5.close()
            raise ValueError("ERROR: Refusing to overwrite pulsars in {0}".format(self.filename))

        # Ok, now we are good. Let's copy the pulsars
        for pulsar in readpsrs:
            if pulsar in self.h5file:
                # Delete the pulsar if it exists
                del self.h5file[pulsar]

            # Copy a pulsar
            self.h5file.copy(sourceh5[pulsar], pulsar)

        # Close both files
        self.h5file.close()
        sourceh5.close()

    """
    Read the basic quantities of a pulsar from an HDF5 file into a ptaPulsar
    object. No extra model matrices and auxiliary variables are added to the
    HDF5 file. If any field is not present in the HDF5 file, an IOError
    exception is raised

    @param psr:     The ptaPulsar object we need to fill with data
    @param psrname: The name of the pulsar to be read from the HDF5 file

    TODO: The HDF5 file is opened and closed every call of 'getData'. That seems
          kind of inefficient
    """
    def readPulsar(self, psr, psrname):
        psr.name = psrname

        # Read the content of the par/tim files in a string
        psr.parfile_content = str(self.getData(psrname, 'parfile', required=False))
        psr.timfile_content = str(self.getData(psrname, 'timfile', required=False))

        # Read the timing model parameter descriptions
        psr.ptmdescription = map(str, self.getData(psrname, 'tmp_name'))
        psr.ptmpars = np.array(self.getData(psrname, 'tmp_valpre'))
        psr.ptmparerrs = np.array(self.getData(psrname, 'tmp_errpre'))
        psr.flags = map(str, self.getData(psrname, 'efacequad', 'Flags'))

        # Read the position of the pulsar
        rajind = np.flatnonzero(np.array(psr.ptmdescription) == 'RAJ')
        decjind = np.flatnonzero(np.array(psr.ptmdescription) == 'DECJ')
        psr.raj = np.array(self.getData(psrname, 'tmp_valpre'))[rajind]
        psr.decj = np.array(self.getData(psrname, 'tmp_valpre'))[decjind]

        # Obtain residuals, TOAs, etc.
        psr.toas = np.array(self.getData(psrname, 'TOAs'))
        psr.toaerrs = np.array(self.getData(psrname, 'toaErr'))
        psr.prefitresiduals = np.array(self.getData(psrname, 'prefitRes'))
        psr.residuals = np.array(self.getData(psrname, 'postfitRes'))
        psr.detresiduals = np.array(self.getData(psrname, 'prefitRes'))
        psr.freqs = np.array(self.getData(psrname, 'freq'))
        psr.Mmat = np.array(self.getData(psrname, 'designmatrix'))
        #psr.unitconversion = np.array(self.getData(psrname, 'unitConversion', required=False))

        # We do not read the (co)G-matrix anymore here. Happens when
        # initialising the model


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
    def addpulsarold(self, parfile, timfile, iterations=1):
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

        # Load pulsar data from the libstempo library
        os.chdir(dirname)
        t2pulsar = t2.tempopulsar(relparfile, reltimfile)
        os.chdir(savedir)

        # Create the pulsar subgroup if it des not exist
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

        if iterations > 1:
            t2pulsar.fit(iters=iterations)

        # Create the datasets, with reference time pepoch = 53000
        pulsarsgroup.create_dataset('TOAs', data=np.double(np.array(t2pulsar.toas())-53000)*pic_spd)       # days (MJD) * sec per day
        pulsarsgroup.create_dataset('prefitRes', data=np.double(t2pulsar.prefit.residuals))      # seconds
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
            # efacequad flag with pulsarname+content of the sys-flag. If it
            # isn't, check for a be-flag and try the same. Otherwise, add an
            # efacequad flag with the pulsar name as it's elements.
            efacequad = []
            nobs = len(t2pulsar.toas())
            pulsarname = map(str, [t2pulsar.name] * nobs)
            if "sys" in flaggroup:
                efacequad = map('-'.join, zip(pulsarname, flaggroup['sys']))
            elif "be" in flaggroup:
                efacequad = map('-'.join, zip(pulsarname, flaggroup['be']))
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


def selection_to_dselection(Nvec, U):
    """
    Given a selection vector Nvec and a quantization matrix U, both with
    elements in [0, 1.0], this function returns the selection vector in the
    basis of epoch average residuals. This assumes that all observations in a
    single observation epoch are flagged identically (same backend).

    @param Nvec:    vector with elements in [0.0, 1.0] that indicates which
                    observations are selected
    @param U:       quantization matrix
    
    @returns:   Selection matrix as Nvec, but in the epoch-averaged basis
    """
    return np.array(np.sum(Nvec * U.T, axis=1) > 0.0, dtype=np.double)


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

def AntennaPatternPC(rajp, decjp, raj, decj):
    """
    Return the x,+ polarisation antenna pattern for a given source position and
    polsar position

    @param rajp:    Right ascension pulsar
    @param decj:    Declination pulsar
    @param raj:     Right ascension source
    @param dec:     Declination source
    """
    Omega = np.array([-np.cos(decj)*np.cos(raj), \
                      -np.cos(decj)*np.sin(raj), \
                      np.sin(decj)])
    
    mhat = np.array([-np.sin(raj), np.cos(raj), 0])
    nhat = np.array([-np.cos(raj)*np.sin(decj), \
                     -np.sin(decj)*np.sin(raj), \
                     np.cos(decj)])

    p = np.array([np.cos(rajp)*np.cos(decj), \
                  np.sin(rajp)*np.cos(decj), \
                  np.sin(decj)])

    Fp = 0.5 * (np.dot(mhat, p)**2 - np.dot(nhat, p)**2) / (1 + np.dot(Omega, p))
    Fc = np.dot(mhat, p) * np.dot(nhat, p) / (1 + np.dot(Omega, p))

    return Fp, Fc


# The GWB general anisotropic correlations in the pixel-basis
class pixelCorrelations(object):
    phiarr = None           # The phi pulsar position parameters
    thetaarr = None         # The theta pulsar position parameters
    cmat = None             # The correlation matrix
    Fp = None               # Plus antenna pattern
    Fc = None               # Cross antenna pattern
    npixels = 4

    def __init__(self, psrs=None, npixels=4):
        # If we have a pulsars object, initialise the angular quantities
        if psrs != None:
            self.setmatrices(psrs, npixels)
            self.npixels = npixels
        else:
            self.phiarr = None           # The phi pulsar position parameters
            self.thetaarr = None         # The theta pulsar position parameters
            self.cmat = None
            self.Fp = None
            self.Fc = None

    def setmatrices(self, psrs, npixels=4):
        # First set all the pulsar positions
        self.phiarr = np.zeros(len(psrs))
        self.thetaarr = np.zeros(len(psrs))
        self.cmat = np.zeros((len(psrs), len(psrs)))
        self.Fp = np.zeros((len(psrs), npixels))
        self.Fc = np.zeros((len(psrs), npixels))
        self.npixels = npixels

        for ii in range(len(psrs)):
            self.phiarr[ii] = psrs[ii].raj
            self.thetaarr[ii] = 0.5*np.pi - psrs[ii].decj

    # Return the full correlation matrix that depends on the gwb direction
    def corrmat(self, pixpars):
        """
        pixpars[0]: right ascension source
        pixpars[1]: declination source
        ... etc.
        """
        for ii in range(len(self.phiarr)):
            for pp in range(self.npixels):
                self.Fp[ii, pp], self.Fc[ii, pp] = AntennaPatternPC( \
                        self.phiarr[ii], 0.5*np.pi - self.thetaarr[ii], \
                        pixpars[2*pp], pixpars[1+2*pp])

        for ii in range(len(self.phiarr)):
            for jj in range(ii, len(self.phiarr)):
                self.cmat[ii, jj] = 0.0
                for pp in range(self.npixels):
                    if ii == jj:
                        self.cmat[ii, jj] += 0.5*self.Fp[ii, pp]**2 + \
                                            0.5*self.Fc[ii, pp]**2
                    else:
                        self.cmat[ii, jj] += 0.5*self.Fp[ii, pp]*self.Fp[jj, pp] + \
                                            0.5*self.Fc[ii, pp]*self.Fc[jj, pp]
                        self.cmat[jj, ii] = self.cmat[ii, jj]

        return self.cmat / self.npixels





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



"""
A general signal element of the pta model/likelihood.

For now, the Fmat fourier design matrices are assumed to be for identical
frequencies for all pulsars.

Note: deprecated
"""
class ptasignalOld(object):
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
    parindex = 0            # Index in parameters array
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

    # Quantities for spectral noise
    Tmax = None
    corrmat = None

    # Quantities for correlated signals
    Ffreqs = None
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
    parfile_content = None      # The actual content of the original par-file
    timfile_content = None      # The actual content of the original tim-file
    t2psr = None                # A libstempo object, if libstempo is imported

    raj = 0
    decj = 0
    toas = None
    toaerrs = None
    prefitresiduals = None
    residuals = None
    detresiduals = None     # Residuals after subtraction of deterministic sources
    gibbsresiduals = None     # Residuals used in Gibbs sampling  (QUESTION: why no parameter?)
    gibbscoefficients = None    # Coefficients used in Gibbs sampling  (QUESTION: why no parameter?)
    freqs = None
    #unitconversion = None
    Gmat = None
    Gcmat = None
    Mmat = None
    ptmpars = []
    ptmparerrs = []
    ptmdescription = []
    flags = None
    name = "J0000+0000"
    Tmax = None

    # The auxiliary quantities
    Fmat = None
    SFmat = None            # Fmatrix for the frequency lines
    FFmat = None            # Total of Fmat and SFmat
    Fdmmat = None
    Hmat = None             # The compression matrix
    Homat = None            # The orthogonal-to compression matrix
    Hcmat = None            # The co-compression matrix
    Hocmat = None           # The orthogonal co-compression matrix
    Umat = None
    Uimat = None
    avetoas = None          
    SFdmmat = None         # Fdmmatrix for the dm frequency lines
    #FFdmmat = None         # Total of SFdmmatrix and Fdmmat
    #Dmat = None
    Dvec = None
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
    Zmat = None         # For the Gibbs sampling, this is the Fmat/Emat
    Gr = None
    GGr = None
    GtF = None
    GtD = None
    GtU = None
    #GGtD = None
    AGr = None      # Replaces GGr in 2-component noise model
    AoGr = None     #   Same but for orthogonal basis (when compressing)
    AGF = None      # Replaces GGtF in 2-component noise model
    #AoGF = None     #   Same but for orthogonal basis (when compressing)
    AGD = None      # Replaces GGtD in 2-component noise model
    #AoGD = None     #   Same but for orthogonal basis (when compressing)
    AGE = None      # Replaces GGtE in 2-component noise model
    #AoGE = None     #   Same but for orthogonal basis (when compressing)
    AGU = None      # Replace GGtU in 2-component noise model
    #AoGU = None     #   Same .... you got it

    # Auxiliaries used in the likelihood
    twoComponentNoise = False       # Whether we use the 2-component noise model
    Nvec = None             # The total white noise (eq^2 + ef^2*err)
    Wvec = None             # The weights in 2-component noise
    Wovec = None            # The weights in 2-component orthogonal noise
    Nwvec = None            # Total noise in 2-component basis (eq^2 + ef^2*Wvec)
    Nwovec = None           # Total noise in 2-component orthogonal basis
    Jweight = None          # The weight of the jitter noise in compressed basis
    Jvec = None

    # To select the number of Frequency modes
    bfinc = None        # Number of modes of all internal matrices
    bfdminc = None      # Number of modes of all internal matrices (DM)
    bcurfinc = None     # Current number of modes in RJMCMC
    bcurfdminc = None   # Current number of modes in RJMCMC

    # Indices for when we are in mark11
    fourierind = None
    dmfourierind = None
    jitterfourierind = None

    def __init__(self):
        self.parfile_content = None
        self.timfile_content = None
        self.t2psr = None

        self.raj = 0
        self.decj = 0
        self.toas = None
        self.toaerrs = None
        self.prefitresiduals = None
        self.residuals = None
        self.detresiduals = None     # Residuals after subtraction of deterministic sources
        self.freqs = None
        #self.unitconversion = None
        self.Gmat = None
        self.Gcmat = None
        self.Mmat = None
        self.ptmpars = []
        self.ptmparerrs = []
        self.ptmdescription = []
        self.flags = None
        self.name = "J0000+0000"
        self.Tmax = None

        self.Fmat = None
        self.SFmat = None
        self.FFmat = None
        self.Fdmmat = None
        self.Hmat = None
        self.Homat = None
        self.Hcmat = None
        self.Hocmat = None
        self.Umat = None
        self.Uimat = None
        self.avetoas = None
        #self.Dmat = None
        self.Dvec = None
        self.DF = None
        self.Ffreqs = None
        self.SFfreqs = None
        self.Fdmfreqs = None
        self.Emat = None
        self.EEmat = None
        self.Gr = None
        self.GGr = None
        self.GtF = None
        self.GtD = None
        #self.GGtFF = None
        #self.GGtD = None

        self.bfinc = None
        self.bfdminc = None
        self.bprevfinc = None
        self.bprevfdminc = None

        # Indices for when we are in mark11
        self.fourierind = None
        self.dmfourierind = None
        self.jitterfourierind = None

    """
    Read the pulsar data (TOAs, residuals, design matrix, etc..) from an HDF5
    file

    @param h5df:        The DataFile object we are reading from
    @param psrname:     Name of the Pulsar to be read from the HDF5 file
    """
    def readFromH5(self, h5df, psrname):
        h5df.readPulsar(self, psrname)

    """
    Initialise the libstempo object for use in nonlinear timing model modelling.
    No parameters are required, all content must already be in memory
    """
    def initLibsTempoObject(self):
        # Check that the parfile_content and timfile_content are set
        if self.parfile_content is None or self.timfile_content is None:
            raise ValueError('No par/tim file present for pulsar {0}'.format(self.name))

        # For non-linear timing models, libstempo must be imported
        if t2 is None:
            raise ImportError("libstempo")

        # Write a temporary par-file and tim-file for libstempo to read. First
        # obtain 
        parfilename = tempfile.mktemp()
        timfilename = tempfile.mktemp()
        parfile = open(parfilename, 'w')
        timfile = open(timfilename, 'w')
        parfile.write(self.parfile_content)
        timfile.write(self.timfile_content)
        parfile.close()
        timfile.close()

        # Create the libstempo object
        self.t2psr = t2.tempopulsar(parfilename, timfilename)

        # Create the BATS?
        # tempresiduals = self.t2psr.residuals(updatebats=True, formresiduals=False)

        # Delete the temporary files
        os.remove(parfilename)
        os.remove(timfilename)



    """
    Constructs a new modified design matrix by adding some columns to it. Returns
    a list of new objects that represent the new timing model

    @param newpars:     Names of the parameters/columns that need to be added
                        For now, can only be [Offset, F0, F1, DM, DM0, DM1]
                        (or higher derivatives of F/DM)

    @param oldMat:      The old design matrix. If None, use current one

    @param oldGmat:     The old G-matrix. ''

    @param oldGcmat:    The old co-G-matrix ''

    @param oldptmpars:  The old timing model parameter values

    @param oldptmdescription:   The old timing model parameter labels

    @param noWarning:   If True, do not warn for duplicate parameters

    @return (list):     Return the elements: (newM, newG, newGc,
                        newptmpars, newptmdescription)
                        in order: the new design matrix, the new G-matrix, the
                        new co-Gmatrix (orthogonal complement), the new values
                        of the timing model parameters, the new descriptions of
                        the timing model parameters. Note that the timing model
                        parameters are not really 'new', just re-selected
    """
    def addToDesignMatrix(self, addpars, \
            oldMmat=None, oldGmat=None, oldGcmat=None, \
            oldptmpars=None, oldptmdescription=None, \
            noWarning=False):
        if oldMmat is None:
            oldMmat = self.Mmat
        if oldptmdescription is None:
            oldptmdescription = self.ptmdescription
        if oldptmpars is None:
            oldptmpars = self.ptmpars
        #if oldunitconversion is None:
        #    oldunitconversion = self.unitconversion
        if oldGmat is None:
            oldGmat = self.Gmat
        if oldGcmat is None:
            oldGcmat = self.Gcmat

        # First make sure that the parameters we are adding are not already in
        # the design matrix
        indok = np.array([1]*len(addpars), dtype=np.bool)
        addpars = np.array(addpars)
        #for ii, parlabel in enumerate(oldptmdescription):
        for ii, parlabel in enumerate(addpars):
            if parlabel in oldptmdescription:
                indok[ii] = False

        if sum(indok) != len(indok) and not noWarning:
            print "WARNING: cannot add parameters to the design matrix that are already present"
            print "         refusing to add:", map(str, addpars[indok == False])

        # Only add the parameters with indok == True
        if sum(indok) > 0:
            # We have some parameters to add
            addM = np.zeros((oldMmat.shape[0], np.sum(indok)))
            adddes = map(str, addpars[indok])
            addparvals = []
            addunitvals = []

            Dmatdiag = pic_DMk / (self.freqs**2)
            for ii, par in enumerate(addpars[indok]):
                addparvals.append(0.0)
                addunitvals.append(1.0)
                if par == 'DM':
                    addM[:, ii] = Dmatdiag.copy()
                elif par[:2] == 'DM':
                    power = int(par[2:])
                    addM[:, ii] = Dmatdiag * (self.toas ** power)
                elif par == 'Offset':
                    addM[:, ii] = 1.0
                elif par[0] == 'F':
                    try:
                        power = int(par[1:])
                        addM[:, ii] = (self.toas ** power)
                    except ValueError:
                        raise ValueError("ERROR: parameter {0} not implemented in 'addToDesignMatrix'".format(par))
                else:
                    raise ValueError("ERROR: parameter {0} not implemented in 'addToDesignMatrix'".format(par))

            newM = np.append(oldMmat, addM, axis=1)
            newptmdescription = np.append(oldptmdescription, adddes)
            newptmpars = np.append(oldptmpars, addparvals)
            #newunitconversion = np.append(oldunitconversion, addunitvals)

            # Construct the G-matrices
            U, s, Vh = sl.svd(newM)
            newG = U[:, (newM.shape[1]):].copy()
            newGc = U[:, :(newM.shape[1])].copy()
        else:
            newM = oldMmat.copy()
            newptmdescription = np.array(oldptmdescription)
            #newunitconversion = np.array(oldunitconversion)
            newptmpars = oldptmpars.copy()

            if oldGmat is not None:
                newG = oldGmat.copy()
                newGc = oldGcmat.copy()
            else:
                U, s, Vh = sl.svd(newM)
                newG = U[:, (newM.shape[1]):].copy()
                newGc = U[:, :(newM.shape[1])].copy()


        return newM, newG, newGc, newptmpars, map(str, newptmdescription)

    """
    Consgtructs a new modified design matrix by deleting some columns from it.
    Returns a list of new objects that represent the new timing model

    @param delpars:     Names of the parameters/columns that need to be deleted.
    
    @return (list):     Return the elements: (newM, newG, newGc,
                        newptmpars, newptmdescription)
                        in order: the new design matrix, the new G-matrix, the
                        new co-Gmatrix (orthogonal complement), the new values
                        of the timing model parameters, the new descriptions of
                        the timing model parameters. Note that the timing model
                        parameters are not really 'new', just re-selected
    """
    def delFromDesignMatrix(self, delpars, \
            oldMmat=None, oldGmat=None, oldGcmat=None, \
            oldptmpars=None, oldptmdescription=None):
        if oldMmat is None:
            oldMmat = self.Mmat
        if oldptmdescription is None:
            oldptmdescription = self.ptmdescription
        #if oldunitconversion is None:
        #    oldunitconversion = self.unitconversion
        if oldptmpars is None:
            oldptmpars = self.ptmpars
        if oldGmat is None:
            oldGmat = self.Gmat
        if oldGcmat is None:
            oldGcmat = self.Gcmat

        # First make sure that the parameters we are deleting are actually in
        # the design matrix
        inddel = np.array([1]*len(delpars), dtype=np.bool)
        indkeep = np.array([1]*oldMmat.shape[1], dtype=np.bool)
        delpars = np.array(delpars)
        #for ii, parlabel in enumerate(oldptmdescription):
        for ii, parlabel in enumerate(delpars):
            if not parlabel in oldptmdescription:
                inddel[ii] = False
                print "WARNING: {0} not in design matrix. Not deleting".format(parlabel)
            else:
                index = np.flatnonzero(np.array(oldptmdescription) == parlabel)
                indkeep[index] = False

        if np.sum(indkeep) != len(indkeep):
            # We have actually deleted some parameters
            newM = oldMmat[:, indkeep]
            newptmdescription = np.array(oldptmdescription)[indkeep]
            #newunitconversion = np.array(oldunitconversion)[indkeep]
            newptmpars = oldptmpars[indkeep]

            # Construct the G-matrices
            U, s, Vh = sl.svd(newM)
            newG = U[:, (newM.shape[1]):].copy()
            newGc = U[:, :(newM.shape[1])].copy()
        else:
            newM = oldMmat.copy()
            newptmdescription = np.array(oldptmdescription)
            #newunitconversion = oldunitconversion.copy()
            newptmpars = oldptmpars.copy()
            newG = oldGmat.copy()
            newGc = oldGcmat.copy()

        return newM, newG, newGc, newptmpars, map(str, newptmdescription)


    """
    Construct a modified design matrix, based on some options. Returns a list of
    new objects that represent the new timing model

    @param addDMQSD:    Whether we should make sure that the DM quadratics are
                        fit for. Should have 'DM', 'DM1', 'DM2'. If not present,
                        add them
    @param addQSD:      Same as addDMQSD, but now for pulsar spin frequency.
    @param removeJumps: Remove the jumps from the timing model.
    @param removeAll:   This removes all parameters from the timing model,
                        except for the DM parameters, and the QSD parameters

    @return (list):     Return the elements: (newM, newG, newGc,
                        newptmpars, newptmdescription)
                        in order: the new design matrix, the new G-matrix, the
                        new co-Gmatrix (orthogonal complement), the new values
                        of the timing model parameters, the new descriptions of
                        the timing model parameters. Note that the timing model
                        parameters are not really 'new', just re-selected

    TODO: Split this function in two parts. One that receives a list of
          names/identifiers of which parameters to include. The other that
          constructs the list, and calls that function.
    """
    def getModifiedDesignMatrix(self, addDMQSD=False, addQSD=False, \
            removeJumps=False, removeAll=False):
        
        (newM, newG, newGc, newptmpars, newptmdescription) = \
                (self.Mmat, self.Gmat, self.Gcmat, self.ptmpars, \
                self.ptmdescription)

        # DM and QSD parameter names
        dmaddes = ['DM', 'DM1', 'DM2']
        qsdaddes = ['Offset', 'F0', 'F1']

        # See which parameters need to be added
        addpar = []
        for parlabel in dmaddes:
            if addDMQSD and not parlabel in self.ptmpars:
                addpar += [parlabel]
        for parlabel in qsdaddes:
            if addQSD and not parlabel in self.ptmpars:
                addpar += [parlabel]

        # Add those parameters
        if len(addpar) > 0:
            (newM, newG, newGc, newptmpars, newptmdescription) = \
                    self.addToDesignMatrix(addpar, newM, newG, newGc, \
                    newptmpars, newptmdescription, \
                    noWarning=True)

        # See whether some parameters need to be deleted
        delpar = []
        for ii, parlabel in enumerate(self.ptmdescription):
            if (removeJumps or removeAll) and parlabel[:4].upper() == 'JUMP':
                delpar += [parlabel]
            elif removeAll and (not parlabel in dmaddes and not parlabel in qsdaddes):
                delpar += [parlabel]

        # Delete those parameters
        if len(delpar) > 0:
            (newM, newG, newGc, newptmpars, newptmdescription) = \
                    self.delFromDesignMatrix(delpar, newM, newG, newGc, \
                    newptmpars, newptmdescription)

        return newM, newG, newGc, newptmpars, newptmdescription




    # Modify the design matrix to include fitting for a quadratic in the DM
    # signal.
    # TODO: Check if the DM is fit for in the design matrix. Use ptmdescription
    #       for that. It should have a field with 'DM' in it.
    def addDMQuadratic(self):
        self.Mmat, self.Gmat, self.Gcmat, self.ptmpars, \
                self.ptmdescription = \
                self.getModifiedDesignMatrix(addDMQSD=True, removeJumps=False)


    """
    Estimate how many frequency modes are required for this pulsar. This
    function uses a simplified method, based on van Haasteren (2013). Given a
    red-noise spectrum (power-law here), a high-fidelity compression technique
    is used.

    @param noiseAmp:    the expected amplitude we want to be fully sensitive to
    @param noiseSi:     the spectral index of the signal we want to be fully
                        sensitive to
    @param Tmax:        time baseline, if not determined from this pulsar
    @param threshold:   the fidelity with which the signal has to be
                        reconstructed
    """
    def numfreqsFromSpectrum(self, noiseAmp, noiseSi, \
            Tmax=None, threshold=0.99, dm=False):
        ntoas = len(self.toas)
        nfreqs = int(ntoas/2)

        if Tmax is None:
            Tmax = np.max(self.toas) - np.min(self.toas)

        # Construct the Fourier modes, and the frequency coefficients (for
        # noiseAmp=1)
        (Fmat, Ffreqs) = fourierdesignmatrix(self.toas, 2*nfreqs, Tmax)
        freqpy = Ffreqs * pic_spy
        pcdoubled = (pic_spy**3 / (12*np.pi*np.pi * Tmax)) * freqpy ** (-noiseSi)

        if dm:
            # Make Fmat into a DM variation Fmat
            Dvec = pic_DMk / (self.freqs**2)
            Fmat = (Dvec * Fmat.T).T

        # Check whether the Gmatrix exists
        if self.Gmat is None:
            U, s, Vh = sl.svd(self.Mmat)
            Gmat = U[:, self.Mmat.shape[1]:]
        else:
            Gmat = self.Gmat

        # Find the Cholesky decomposition of the projected radiometer-noise
        # covariance matrix
        GNG = np.dot(Gmat.T, (self.toaerrs**2 * Gmat.T).T)
        try:
            L = sl.cholesky(GNG).T
            cf = sl.cho_factor(L)
            Li = sl.cho_solve(cf, np.eye(GNG.shape[0]))
        except np.linalg.LinAlgError as err:
            raise ValueError("ERROR: GNG singular according to Cholesky")

        # Construct the transformed Phi-matrix, and perform SVD. That matrix
        # should have a few singular values (nfreqs not precisely determined)
        LGF = np.dot(Li, np.dot(Gmat.T, Fmat))
        Phiw = np.dot(LGF, (pcdoubled * LGF).T)
        U, s, Vh = sl.svd(Phiw)

        # From the eigenvalues in s, we can determine the number of frequencies
        fisherelements = s**2 / (1 + noiseAmp**2 * s)**2
        cumev = np.cumsum(fisherelements)
        totrms = np.sum(fisherelements)
        return int((np.flatnonzero( (cumev/totrms) >= threshold )[0] + 1)/2)


    """
    Figure out what the list of timing model parameters is that needs to be
    deleted from the design matrix in order to do nonlinear timing model
    parameter analysis, given 

    @param tmpars:  A list of suggested parameters to keep in the design
                    matrix. Only parameters not present in this list and present
                    in the design matrix will be returned.
    @param keep:    If True, return the parameters that we keep in the design
                    matrix. If False, return the parameters that we will delete
                    from the design matrix

    @return:        List of parameters to be deleted
    """
    def getNewTimingModelParameterList(self, keep=True, \
            tmpars = None):
        # Remove from the timing model parameter list of the design matrix,
        # all parameters not in the list 'tmpars'. The parameters not in
        # tmpars are numerically included
        if tmpars is None:
            tmpars = ['Offset', 'F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', \
                    'PX', 'DM', 'DM1', 'DM2']

        tmparkeep = []
        tmpardel = []
        for tmpar in self.ptmdescription:
            if tmpar in tmpars:
                # This parameter stays in the compression matrix (so is
                # marginalised over
                tmparkeep += [tmpar]
            elif tmpar == 'Offset':
                print "WARNING: Offset needs to be included in the design matrix. Including it anyway..."
                tmparkeep += [tmpar]
            else:
                tmpardel += [tmpar]

        if keep:
            returnpars = tmparkeep
        else:
            returnpars = tmpardel

        return returnpars



    """
    Construct the compression matrix and it's orthogonal complement. This is
    always done, even if in practice there is no compression. That is just the
    fidelity = 1 case.

        # U-compression:
        # W s V^{T} = G^{T} U U^{T} G    H = G Wl
        # F-compression
        # W s V^{T} = G^{T} F F^{T} G    H = G Wl

    @param compression: what kind of compression to use. Can be \
                        None/average/frequencies/avefrequencies
    @param nfmodes:     when using frequencies, use this number if not -1
    @param ndmodes:     when using dm frequencies, use this number if not -1
    @param likfunc:     which likelihood function is being used. Only useful when it
                        is mark4/not mark4. TODO: parameter can be removed?
    @param threshold:   To which fidelity will we compress the basis functions [1.0]
    @param tmpars:      When compressing to a list of timing model parameters,
                        this list of parameters is used.
    """
    # TODO: selection of timing-model parameters should apply to _all_ forms of
    # compression. Still possible to do frequencies and include timing model
    # parameters, as long as we include the complement function
    def constructCompressionMatrix(self, compression='None', \
            nfmodes=-1, ndmodes=-1, likfunc='mark4', threshold=1.0, \
            tmpars = None):
        if compression == 'average':
            # To be sure, just construct the averages again. But is already done
            # in 'createPulsarAuxiliaries'
            if likfunc[:5] != 'mark4':
                (self.avetoas, self.Umat) = dailyaveragequantities(self.toas)
                Wjit = np.sum(self.Umat, axis=0)
                self.Jweight = np.sum(Wjit * self.Umat, axis=1)

            #"""
            GU = np.dot(self.Gmat.T, self.Umat)
            GUUG = np.dot(GU, GU.T)
            #"""

            # Construct an orthogonal basis, and singular values
            #svech, Vmath = sl.eigh(GUUG)
            Vmat, svec, Vhsvd = sl.svd(GUUG)

            # Decide how many basis vectors we'll take. (Would be odd if this is
            # not the number of columns of self.U. How to test? For now, use
            # 99.9% of rms power
            cumrms = np.cumsum(svec)
            totrms = np.sum(svec)
            #print "svec:   ", svec
            #print "cumrms: ", cumrms
            #print "totrms: ", totrms
            inds = (cumrms/totrms) >= threshold
            if np.sum(inds) > 0:
                # We can compress
                l = np.flatnonzero( inds )[0] + 1
            else:
                # We cannot compress, keep all
                l = self.Umat.shape[1]

            print "Number of U basis vectors for " + \
                    self.name + ": " + str(self.Umat.shape) + \
                    " --> " + str(l)
            """
            ll = self.Umat.shape[1]

            print "Umat: ", self.Umat.shape
            print "CumRMS:    ", cumrms
            print "cumrms[l] / tot = ", cumrms[l] / totrms
            print "svec range:   ", svec[130:150]
            print "cumrms range: ", cumrms[130:150]
            print "designmatrix: ", self.Mmat.shape
            print "TMPars: ", self.ptmdescription
            plt.plot(np.arange(120,ll+10), np.log10(svec[120:ll+10]), 'k-')
            """

            # H is the compression matrix
            Bmat = Vmat[:, :l].copy()
            Bomat = Vmat[:, l:].copy()
            H = np.dot(self.Gmat, Bmat)
            Ho = np.dot(self.Gmat, Bomat)

            # Use another SVD to construct not only Hmat, but also Hcmat
            # We use this version of Hmat, and not H from above, in case of
            # linear dependences...
            #svec, Vmat = sl.eigh(H)
            Vmat, s, Vh = sl.svd(H)
            self.Hmat = Vmat[:, :l]
            self.Hcmat = Vmat[:, l:]

            # For compression-complements, construct Ho and Hoc
            if Ho.shape[1] > 0:
                Vmat, s, Vh = sl.svd(Ho)
                self.Homat = Vmat[:, :Ho.shape[1]]
                self.Hocmat = Vmat[:, Ho.shape[1]:]
            else:
                self.Homat = np.zeros((Vmat.shape[0], 0))
                self.Hocmat = np.eye(Vmat.shape[0])

        elif compression == 'frequencies' or compression == 'avefrequencies':
            Ftot = np.zeros((len(self.toas), 0))

            # Decide on the (dm)frequencies to include
            if nfmodes == -1:
                # Include all, and only all, frequency modes
                #Ftot = np.append(Ftot, self.Fmat, axis=1)

                # Produce an orthogonal basis for the frequencies
                l = self.Fmat.shape[1]
                Vmat, svec, Vhsvd = sl.svd(self.Fmat)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)
            elif nfmodes == 0:
                # Why would anyone do this?
                pass
            else:
                # Should we check whether nfmodes is not too large?
                #Ftot = np.append(Ftot, self.Fmat[:, :nfmodes], axis=1)

                # Produce an orthogonal basis for the frequencies
                l = nfmodes
                Vmat, svec, Vhsvd = sl.svd(self.Fmat)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)

            if ndmodes == -1:
                # Include all, and only all, frequency modes
                # Ftot = np.append(Ftot, self.DF, axis=1)

                # Produce an orthogonal basis for the frequencies
                l = self.DF.shape[1]
                Vmat, svec, Vhsvd = sl.svd(self.DF)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)
            elif ndmodes == 0:
                # Do not include DM in the compression
                pass
            else:
                # Should we check whether nfmodes is not too large?
                # Ftot = np.append(Ftot, self.DF[:, :ndmodes], axis=1)

                # Produce an orthogonal basis for the frequencies
                l = self.DF.shape[1]
                Vmat, svec, Vhsvd = sl.svd(self.DF)
                Ftot = np.append(Ftot, Vmat[:, :l].copy(), axis=1)

            if compression == 'avefrequencies':
                print "WARNING: this type of compression is only for testing purposes"

                # Calculate Umat and Ui
                (self.avetoas, self.Umat, Ui) = dailyaveragequantities(self.toas, calcInverse=True)
                UUi = np.dot(self.Umat, Ui)
                GF = np.dot(self.Gmat.T, np.dot(UUi, Ftot))

                Wjit = np.sum(self.Umat, axis=0)
                self.Jweight = np.sum(Wjit * self.Umat, axis=1)
            else:
                GF = np.dot(self.Gmat.T, Ftot)

            GFFG = np.dot(GF, GF.T)

            # Construct an orthogonal basis, and singular (eigen) values
            #svec, Vmat = sl.eigh(GFFG)
            Vmat, svec, Vhsvd = sl.svd(GFFG)

            # Decide how many basis vectors we'll take.
            cumrms = np.cumsum(svec)
            totrms = np.sum(svec)
            # print "Freqs: ", cumrms / totrms
            l = np.flatnonzero( (cumrms/totrms) >= threshold )[0] + 1
            # l = Ftot.shape[1]-8         # This line would cause the threshold to be ignored

            #print "Number of F basis vectors for " + \
            #        self.name + ": " + str(self.Fmat.shape) + \
            #        " --> " + str(l)

            # H is the compression matrix
            Bmat = Vmat[:, :l].copy()
            Bomat = Vmat[:, l:].copy()
            H = np.dot(self.Gmat, Bmat)
            Ho = np.dot(self.Gmat, Bomat)

            # Use another SVD to construct not only Hmat, but also Hcmat
            # We use this version of Hmat, and not H from above, in case of
            # linear dependences...
            #svec, Vmat = sl.eigh(H)
            Vmat, s, Vh = sl.svd(H)
            self.Hmat = Vmat[:, :l]
            self.Hcmat = Vmat[:, l:]

            # For compression-complements, construct Ho and Hoc
            Vmat, s, Vh = sl.svd(Ho)
            self.Homat = Vmat[:, :Ho.shape[1]]
            self.Hocmat = Vmat[:, Ho.shape[1]:]
        elif compression == 'qsd':
            # Only include (DM)QSD in the G-matrix. The other parameters can be
            # handled numerically with 'lineartimingmodel' signals
            (newM, newG, newGc, newptmpars, newptmdescription) = \
                    self.getModifiedDesignMatrix(removeAll=True)
            self.Hmat = newG
            self.Hcmat = newGc
            self.Homat = np.zeros((self.Hmat.shape[0], 0))      # There is no complement
            self.Hocmat = np.zeros((self.Hmat.shape[0], 0))
        elif compression == 'timingmodel':
            tmpardel = self.getNewTimingModelParameterList(keep=False, tmpars=tmpars)

            (newM, newG, newGc, newptmpars, newptmdescription) = \
                    self.delFromDesignMatrix(tmpardel)
            self.Hmat = newG
            self.Hcmat = newGc
            self.Homat = np.zeros((self.Hmat.shape[0], 0))      # There is no complement
            self.Hocmat = np.zeros((self.Hmat.shape[0], 0))
        elif compression == 'None' or compression is None:
            self.Hmat = self.Gmat
            self.Hcmat = self.Gcmat
            self.Homat = np.zeros((self.Hmat.shape[0], 0))      # There is no complement
            self.Hocmat = np.zeros((self.Hmat.shape[0], 0))
        elif compression == 'dont':     # Do not compress
            pass
        else:
            raise IOError, "Invalid compression argument"


    """
    For every pulsar, quite a few Auxiliary quantities (like GtF etc.) are
    necessary for the evaluation of various likelihood functions. This function
    calculates these quantities, and optionally writes them to the HDF5 file for
    quick use later.

    @param h5df:            The DataFile we will write things to
    @param Tmax:            The full duration of the experiment
    @param nfreqs:          The number of noise frequencies we require for this
                            pulsar
    @param ndmfreqs:        The number of DM frequencies we require for this pulsar
    @param twoComponent:    Whether or not we do the two-component noise
                            acceleration
    @param nSingleFreqs:    The number of single floating noise frequencies
    @param nSingleDMFreqs:  The number of single floating DM frequencies
    @param compression:     Whether we use compression (None/frequencies/average)
    @param likfunc:         Which likelihood function to do it for (all/markx/..)
    @param write:           Which data to write to the HDF5 file ('no' for no
                            writing, 'likfunc' for the current likfunc, 'all'
                            for all quantities
    @param tmsigpars:       If not none, this is a list of the timing model
                            parameters that we are treating numerically.
                            Therefore, do not include in the compression matrix

    """
    def createPulsarAuxiliaries(self, h5df, Tmax, nfreqs, ndmfreqs, \
            twoComponent=False, nSingleFreqs=0, nSingleDMFreqs=0, \
            compression='None', likfunc='mark3', write='likfunc', \
            tmsigpars=None, noGmatWrite=False):
        # For creating the auxiliaries it does not really matter: we are now
        # creating all quantities per default
        # TODO: set this parameter in another place?
        if twoComponent and likfunc!='mark11':
            self.twoComponentNoise = True

        # Before writing anything to file, we need to know right away how many
        # fixed and floating frequencies this model contains.
        nf = 0 ; ndmf = 0 ; nsf = nSingleFreqs ; nsdmf = nSingleDMFreqs
        if nfreqs is not None and nfreqs != 0:
            nf = nfreqs
        if ndmfreqs is not None and ndmfreqs != 0:
            ndmf = ndmfreqs

        self.Tmax = Tmax

        # Write these numbers to the HDF5 file
        if write != 'no':
            # Check whether the frequencies already exist in the HDF5-file. If
            # so, compare with what we have here. If they differ, then print out
            # a warning.
            # TODO: instead of a warning, and overwriting, something more
            #       conservative should be done
            modelFrequencies = np.array([nf, ndmf, nsf, nsdmf])
            try:
                file_modelFreqs = np.array(h5df.getData(self.name, 'pic_modelFrequencies'))
                if not np.all(modelFrequencies == file_modelFreqs):
                    print "WARNING: model frequencies already present in {0} differ from the current".format(h5df.filename)
                    print "         model. Overwriting..."
            except IOError:
                pass

            h5df.addData(self.name, 'pic_modelFrequencies', modelFrequencies)
            h5df.addData(self.name, 'pic_Tmax', [self.Tmax])

        # Create the Fourier design matrices for noise
        if nf > 0:
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nf, Tmax)
        else:
            self.Fmat = np.zeros((len(self.toas), 0))
            self.Ffreqs = np.zeros(0)

        # Create the Fourier design matrices for DM variations
        if ndmf > 0:
            (self.Fdmmat, self.Fdmfreqs) = fourierdesignmatrix(self.toas, 2*ndmf, Tmax)
            #self.Dmat = np.diag(pic_DMk / (self.freqs**2))
            self.Dvec = pic_DMk / (self.freqs**2)
            #self.DF = np.dot(self.Dmat, self.Fdmmat)
            self.DF = (self.Dvec * self.Fdmmat.T).T
        else:
            self.Fdmmat = np.zeros(0)
            self.Fdmfreqs = np.zeros(0)
            #self.Dmat = np.diag(pic_DMk / (self.freqs**2))
            self.Dvec = pic_DMk / (self.freqs**2)
            self.DF = np.zeros((len(self.freqs), 0))

        # Create the dailay averaged residuals
        (self.avetoas, self.Umat, self.Uimat) = \
                dailyaveragequantities(self.toas, calcInverse=True)
        Wjit = np.sum(self.Umat, axis=0)
        self.Jweight = np.sum(Wjit * self.Umat, axis=1)

        # Write these quantities to disk
        if write != 'no':
            h5df.addData(self.name, 'pic_Fmat', self.Fmat)
            h5df.addData(self.name, 'pic_Ffreqs', self.Ffreqs)
            h5df.addData(self.name, 'pic_Fdmmat', self.Fdmmat)
            h5df.addData(self.name, 'pic_Fdmfreqs', self.Fdmfreqs)
            #h5df.addData(self.name, 'pic_Dmat', self.Dmat)
            h5df.addData(self.name, 'pic_Dvec', self.Dvec)
            h5df.addData(self.name, 'pic_DF', self.DF)

            h5df.addData(self.name, 'pic_avetoas', self.avetoas)
            h5df.addData(self.name, 'pic_Umat', self.Umat)
            h5df.addData(self.name, 'pic_Uimat', self.Uimat)
            h5df.addData(self.name, 'pic_Jweight', self.Jweight)

        # Next we'll need the G-matrices, and the compression matrices.
        if compression != 'dont':
            U, s, Vh = sl.svd(self.Mmat)
            self.Gmat = U[:, self.Mmat.shape[1]:]
            self.Gcmat = U[:, :self.Mmat.shape[1]]

            # Construct the compression matrix
            if tmsigpars is None:
                tmpars = None
            else:
                tmpars = []
                for par in self.ptmdescription:
                    if not par in tmsigpars:
                        tmpars += [par]
            self.constructCompressionMatrix(compression, nfmodes=2*nf,
                    ndmodes=2*ndmf, threshold=1.0, tmpars=tmpars)
            if write != 'no':
                h5df.addData(self.name, 'pic_Hcmat', self.Hcmat)

                if not noGmatWrite:
                    h5df.addData(self.name, 'pic_Gmat', self.Gmat)
                    h5df.addData(self.name, 'pic_Gcmat', self.Gcmat)
                    h5df.addData(self.name, 'pic_Hmat', self.Hmat)
                    h5df.addData(self.name, 'pic_Homat', self.Homat)
                    h5df.addData(self.name, 'pic_Hocmat', self.Hocmat)



        # Now, write such quantities on a per-likelihood basis
        if likfunc == 'mark1' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            self.GtD = np.dot(self.Hmat.T, self.DF)
            #(self.avetoas, self.Umat) = dailyaveragequantities(self.toas)
            self.GtU = np.dot(self.Hmat.T, self.Umat)

            # For two-component noise
            # Diagonalise GtEfG (HtEfH)
            if self.twoComponentNoise:
                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)
                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    #HotF = np.dot(self.Homat.T, self.Fmat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGF = np.dot(self.Aomat.T, HotF)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGF = np.zeros((0, self.GtF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_GtD', self.GtD)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGF', self.AGF)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGF', self.AoGF)

        if likfunc == 'mark2' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)

            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)
                self.AGr = np.dot(self.Amat.T, self.Gr)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)

        if likfunc == 'mark3' or likfunc == 'mark3fa' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    HotF = np.dot(self.Homat.T, self.Fmat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGF = np.dot(self.Aomat.T, HotF)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGF = np.zeros((0, self.GtF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGF', self.AGF)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGF', self.AoGF)

        if likfunc == 'mark4' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            GtU = np.dot(self.Hmat.T, self.Umat)

            self.UtF = np.dot(self.Uimat, self.Fmat)
            self.UtD = np.dot(self.Uimat, self.DF)

            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGU = np.dot(self.Amat.T, GtU)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    HotU = np.dot(self.Homat.T, self.Umat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGU = np.dot(self.Aomat.T, HotU)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGU = np.zeros((0, GtU.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_UtF', self.UtF)
                h5df.addData(self.name, 'pic_UtD', self.UtD)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGU', self.AGU)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGU', self.AoGU)


        if likfunc == 'mark4ln' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            GtU = np.dot(self.Hmat.T, self.Umat)

            self.UtF = np.dot(self.Uimat, self.Fmat)

            # Initialise the single frequency with a frequency of 10 / yr
            self.frequencyLinesAdded = nSingleFreqs
            deltaf = 2.3 / pic_spy      # Just some random number
            sfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleFreqs)
            self.SFmat = singleFreqFourierModes(self.toas, np.log10(sfreqs))
            self.FFmat = np.append(self.Fmat, self.SFmat, axis=1)
            self.SFfreqs = np.log10(np.array([sfreqs, sfreqs]).T.flatten())

            self.UtFF = np.dot(self.Uimat, self.FFmat)

            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGU = np.dot(self.Amat.T, GtU)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    HotU = np.dot(self.Homat.T, self.Umat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGU = np.dot(self.Aomat.T, HotU)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGU = np.zeros((0, GtU.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_UtF', self.UtF)
                h5df.addData(self.name, 'pic_UtD', self.UtD)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat)
                h5df.addData(self.name, 'pic_FFmat', self.FFmat)
                h5df.addData(self.name, 'pic_SFfreqs', self.SFfreqs)
                h5df.addData(self.name, 'pic_UtFF', self.UtFF)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGU', self.AGU)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGU', self.AoGU)

        if likfunc == 'mark6' or likfunc == 'mark6fa' or write == 'all':
            # Red noise
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Hmat, self.GtF)


            # DM + Red noise stuff (mark6 needs this)
            self.Emat = np.append(self.Fmat, self.DF, axis=1)

            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                # DM
                GtD = np.dot(self.Hmat.T, self.DF)
                #self.GGtD = np.dot(self.Hmat, GtD)
                GtE = np.dot(self.Hmat.T, self.Emat)
                #self.GGtE = np.dot(self.Hmat, GtE)

                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)
                self.AGD = np.dot(self.Amat.T, GtD)
                self.AGE = np.dot(self.Amat.T, GtE)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    #HotF = np.dot(self.Homat.T, self.Fmat)
                    #HotD = np.dot(self.Homat.T, self.DF)
                    #HotE = np.dot(self.Homat.T, self.Emat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGF = np.dot(self.Aomat.T, HotF)
                    #self.AoGD = np.dot(self.Aomat.T, HotD)
                    #self.AoGE = np.dot(self.Aomat.T, HotE)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGF = np.zeros((0, self.GtF.shape[1]))
                    #self.AoGD = np.zeros((0, GtD.shape[1]))
                    #self.AoGE = np.zeros((0, GtE.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                #h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat)
                #h5df.addData(self.name, 'pic_GGtE', self.GGtE)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGF', self.AGF)
                    h5df.addData(self.name, 'pic_AGD', self.AGD)
                    h5df.addData(self.name, 'pic_AGE', self.AGE)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                    #h5df.addData(self.name, 'pic_AoGD', self.AoGD)
                    #h5df.addData(self.name, 'pic_AoGE', self.AoGE)

        if likfunc == 'mark7' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    #HotF = np.dot(self.Homat.T, self.Fmat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGF = np.dot(self.Aomat.T, HotF)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGF = np.zeros((0, self.GtF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGF', self.AGF)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGF', self.AoGF)

        if likfunc == 'mark8' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # For the DM stuff
            # DM + Red noise stuff
            self.Emat = np.append(self.Fmat, self.DF, axis=1)

            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtD = np.dot(self.Hmat.T, self.DF)
                GtE = np.dot(self.Hmat.T, self.Emat)
                #self.GGtD = np.dot(self.Hmat, GtD)
                #self.GGtE = np.dot(self.Hmat, GtE)

                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)
                self.AGD = np.dot(self.Amat.T, GtD)
                self.AGE = np.dot(self.Amat.T, GtE)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    #HotF = np.dot(self.Homat.T, self.Fmat)
                    #HotD = np.dot(self.Homat.T, self.DF)
                    #HotE = np.dot(self.Homat.T, self.Emat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGF = np.dot(self.Aomat.T, HotF)
                    #self.AoGD = np.dot(self.Aomat.T, HotD)
                    #self.AoGE = np.dot(self.Aomat.T, HotE)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGF = np.zeros((0, self.GtF.shape[1]))
                    #self.AoGD = np.zeros((0, GtD.shape[1]))
                    #self.AoGE = np.zeros((0, GtE.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                #h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat)
                #h5df.addData(self.name, 'pic_GGtE', self.GGtE)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGF', self.AGF)
                    h5df.addData(self.name, 'pic_AGD', self.AGD)
                    h5df.addData(self.name, 'pic_AGE', self.AGE)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                    #h5df.addData(self.name, 'pic_AoGD', self.AoGD)
                    #h5df.addData(self.name, 'pic_AoGE', self.AoGE)

        if likfunc == 'mark9' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # Initialise the single frequency with a frequency of 10 / yr
            self.frequencyLinesAdded = nSingleFreqs
            deltaf = 2.3 / pic_spy      # Just some random number
            sfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleFreqs)
            self.SFmat = singleFreqFourierModes(self.toas, np.log10(sfreqs))
            self.FFmat = np.append(self.Fmat, self.SFmat, axis=1)
            self.SFfreqs = np.log10(np.array([sfreqs, sfreqs]).T.flatten())
            GtFF = np.dot(self.Hmat.T, self.FFmat)

            # For two-component noise model
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)
                self.AGFF = np.dot(self.Amat.T, GtFF)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    #HotF = np.dot(self.Homat.T, self.Fmat)
                    #HotFF = np.dot(self.Homat.T, self.FFmat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGF = np.dot(self.Aomat.T, HotF)
                    #self.AoGFF = np.dot(self.Aomat.T, HotFF)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGF = np.zeros((0, self.GtF.shape[1]))
                    #self.AoGFF = np.zeros((0, GtFF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat)
                h5df.addData(self.name, 'pic_FFmat', self.FFmat)
                h5df.addData(self.name, 'pic_SFfreqs', self.SFfreqs)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGF', self.AGF)
                    h5df.addData(self.name, 'pic_AGFF', self.AGFF)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                    #h5df.addData(self.name, 'pic_AoGFF', self.AoGFF)

        if likfunc == 'mark10' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # For the DM stuff
            # DM + Red noise stuff (mark6 needs this)
            self.Emat = np.append(self.Fmat, self.DF, axis=1)

            # Initialise the single frequency with a frequency of 10 / yr
            self.frequencyLinesAdded = nSingleFreqs
            self.dmfrequencyLinesAdded = nSingleDMFreqs
            deltaf = 2.3 / pic_spy      # Just some random number
            sfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleFreqs)
            sdmfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleDMFreqs)
            self.SFmat = singleFreqFourierModes(self.toas, np.log10(sfreqs))
            self.SFdmmat = singleFreqFourierModes(self.toas, np.log10(sdmfreqs))
            self.FFmat = np.append(self.Fmat, self.SFmat, axis=1)
            self.SFfreqs = np.log10(np.array([sfreqs, sfreqs]).T.flatten())
            self.SFdmfreqs = np.log10(np.array([sdmfreqs, sdmfreqs]).T.flatten())
            #self.DSF = np.dot(self.Dmat, self.SFdmmat)
            self.DSF = (self.Dvec * self.SFdmmat.T).T
            self.DFF = np.append(self.DF, self.DSF, axis=1)

            GtFF = np.dot(self.Hmat.T, self.FFmat)

            self.EEmat = np.append(self.FFmat, self.DFF, axis=1)
            GtEE = np.dot(self.Hmat.T, self.EEmat)
            self.GGtEE = np.dot(self.Hmat, GtEE)
            
            # For two-component noise
            # Diagonalise GtEfG
            if self.twoComponentNoise:
                GtD = np.dot(self.Hmat.T, self.DF)
                GtE = np.dot(self.Hmat.T, self.Emat)
                #self.GGtD = np.dot(self.Hmat, GtD)
                #self.GGtE = np.dot(self.Hmat, GtE)

                GtNeG = np.dot(self.Hmat.T, ((self.toaerrs**2) * self.Hmat.T).T)
                self.Wvec, self.Amat = sl.eigh(GtNeG)

                self.AGr = np.dot(self.Amat.T, self.Gr)
                self.AGF = np.dot(self.Amat.T, self.GtF)

                self.AGFF = np.dot(self.Amat.T, GtFF)
                self.AGD = np.dot(self.Amat.T, GtD)

                self.AGE = np.dot(self.Amat.T, GtE)
                self.AGEE = np.dot(self.Amat.T, GtEE)

                # Diagonalise HotEfHo
                if self.Homat.shape[1] > 0:
                    HotNeHo = np.dot(self.Homat.T, ((self.toaerrs**2) * self.Homat.T).T)
                    self.Wovec, self.Aomat = sl.eigh(HotNeHo)

                    Hor = np.dot(self.Homat.T, self.residuals)
                    #HotF = np.dot(self.Homat.T, self.Fmat)
                    #HotFF = np.dot(self.Homat.T, self.FFmat)
                    #HotD = np.dot(self.Homat.T, self.DF)
                    #HotE = np.dot(self.Homat.T, self.Emat)
                    #HotEE = np.dot(self.Homat.T, self.EEmat)
                    self.AoGr = np.dot(self.Aomat.T, Hor)
                    #self.AoGF = np.dot(self.Aomat.T, HotF)
                    #self.AoGFF = np.dot(self.Aomat.T, HotFF)
                    #self.AoGD = np.dot(self.Aomat.T, HotD)
                    #self.AoGE = np.dot(self.Aomat.T, HotE)
                    #self.AoGEE = np.dot(self.Aomat.T, HotEE)
                else:
                    self.Wovec = np.zeros(0)
                    self.Aomat = np.zeros((self.Amat.shape[0], 0))
                    self.AoGr = np.zeros((0, self.Gr.shape[0]))
                    #self.AoGF = np.zeros((0, self.GtF.shape[1]))
                    #self.AoGFF = np.zeros((0, GtFF.shape[1]))
                    #self.AoGD = np.zeros((0, GtD.shape[1]))
                    #self.AoGE = np.zeros((0, GtE.shape[1]))
                    #self.AoGEE = np.zeros((0, GtEE.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                #h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat)
                #h5df.addData(self.name, 'pic_GGtE', self.GGtE)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat)
                h5df.addData(self.name, 'pic_SFdmmat', self.SFdmmat)
                h5df.addData(self.name, 'pic_FFmat', self.FFmat)
                h5df.addData(self.name, 'pic_SFfreqs', self.SFfreqs)
                h5df.addData(self.name, 'pic_DSF', self.DSF)
                h5df.addData(self.name, 'pic_DFF', self.DFF)
                h5df.addData(self.name, 'pic_EEmat', self.EEmat)
                h5df.addData(self.name, 'pic_GGtEE', self.GGtEE)

                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_AGF', self.AGF)
                    h5df.addData(self.name, 'pic_AGFF', self.AGFF)
                    h5df.addData(self.name, 'pic_AGD', self.AGD)
                    h5df.addData(self.name, 'pic_AGE', self.AGE)
                    h5df.addData(self.name, 'pic_AGEE', self.AGEE)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                    #h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                    #h5df.addData(self.name, 'pic_AoGFF', self.AoGFF)
                    #h5df.addData(self.name, 'pic_AoGD', self.AoGD)
                    #h5df.addData(self.name, 'pic_AoGE', self.AoGE)
                    #h5df.addData(self.name, 'pic_AoGEE', self.AoGEE)

        if likfunc == 'mark11' or write == 'all':
            # No need to write anything just yet?
            pass

        if likfunc == 'gibbs1' or write == 'all':
            # DM + Red noise stuff (mark6 needs this)
            self.Zmat = np.append(self.Mmat, self.Fmat, axis=1)

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Zmat', self.Zmat)



    """
    For every pulsar, quite a few Auxiliary quantities (like GtF etc.) are
    necessary for the evaluation of various likelihood functions. This function
    tries to read these quantities from the HDF5 file, so that we do not have to
    do many unnecessary calculations during runtime.

    @param h5df:            The DataFile we will write things to
    @param Tmax:            The full duration of the experiment
    @param nfreqs:          The number of noise frequencies we require for this
                            pulsar
    @param ndmfreqs:        The number of DM frequencies we require for this pulsar
    @param twoComponent:    Whether or not we do the two-component noise
                            acceleration
    @param nSingleFreqs:    The number of single floating noise frequencies
    @param nSingleDMFreqs:  The number of single floating DM frequencies
    @param compression:     Whether we use compression (None/frequencies/average)
    @param likfunc:         Which likelihood function to do it for (all/markx/..)
    @param memsave:         Whether to save memory

    """
    def readPulsarAuxiliaries(self, h5df, Tmax, nfreqs, ndmfreqs, \
            twoComponent=False, nSingleFreqs=0, nSingleDMFreqs=0, \
            compression='None', likfunc='mark3', \
            evalCompressionComplement=True, memsave=True, noGmat=False):
        # TODO: set this parameter in another place?
        if twoComponent:
            self.twoComponentNoise = True

        # Before reading anything to file, we need to know right away how many
        # fixed and floating frequencies this model contains. This will be
        # checked with the content of the HDF5 file
        nf = 0 ; ndmf = 0 ; nsf = nSingleFreqs ; nsdmf = nSingleDMFreqs
        if nfreqs is not None and nfreqs != 0:
            nf = nfreqs
        if ndmfreqs is not None and ndmfreqs != 0:
            ndmf = ndmfreqs

        # Read in the file frequencies and Tmax
        file_Tmax = h5df.getData(self.name, 'pic_Tmax')
        file_freqs = h5df.getData(self.name, 'pic_modelFrequencies')

        if file_Tmax != Tmax or not np.all(np.array(file_freqs) == \
                np.array([nf, ndmf, nsf, nsdmf])):
            print "file_Tmax, Tmax = {0}, {1}".format(file_Tmax, Tmax)
            print "nf, ndmf, nsf, nsdmf = {0}, {1}, {2}, {3}".format( \
                    nf, ndmf, nsf, nsdmf)
            raise ValueError("File frequencies are not compatible with model frequencies")
        else:
            self.Tmax = Tmax
        # Ok, this model seems good to go. Let's start

        # G/H compression matrices
        self.Gmat = np.array(h5df.getData(self.name, 'pic_Gmat', dontread=memsave))
        self.Gcmat = np.array(h5df.getData(self.name, 'pic_Gcmat', dontread=memsave))
        self.Homat = np.array(h5df.getData(self.name, 'pic_Homat', dontread=memsave))
        self.Hocmat = np.array(h5df.getData(self.name, 'pic_Hocmat',
            dontread=(not evalCompressionComplement)))

        self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat'))

        self.Gr = np.array(h5df.getData(self.name, 'pic_Gr', dontread=memsave))
        self.GGr = np.array(h5df.getData(self.name, 'pic_GGr', dontread=memsave))
        self.Wvec = np.array(h5df.getData(self.name, 'pic_Wvec',
            dontread=(not self.twoComponentNoise)))
        self.Wovec = np.array(h5df.getData(self.name, 'pic_Wovec',
            dontread=(not self.twoComponentNoise)))
        self.Amat = np.array(h5df.getData(self.name, 'pic_Amat',
            dontread=(memsave and not self.twoComponentNoise)))
        self.Aomat = np.array(h5df.getData(self.name, 'pic_Aomat',
            dontread=(memsave and not self.twoComponentNoise)))
        self.AoGr = np.array(h5df.getData(self.name, 'pic_AoGr',
            dontread=(not self.twoComponentNoise)))
        self.Ffreqs = np.array(h5df.getData(self.name, 'pic_Ffreqs'))

        if ndmf > 0:
            self.Fdmfreqs = np.array(h5df.getData(self.name, 'pic_Fdmfreqs'))
        else:
            self.Fdmfreqs = np.zeros(0)

        # If compression is not done, but Hmat represents a compression matrix,
        # we need to re-evaluate the lot. Raise an error
        if not noGmat:
            if (compression == 'None' or compression is None) and \
                    h5df.getShape(self.name, 'pic_Gmat')[1] != \
                    h5df.getShape(self.name, 'pic_Hmat')[1]:
                raise ValueError("Compressed file detected. Re-calculating all quantities.")
            elif (compression != 'None' and compression != None) and \
                    h5df.getShape(self.name, 'pic_Gmat')[1] == \
                    h5df.getShape(self.name, 'pic_Hmat')[1]:
                raise ValueError("Uncompressed file detected. Re-calculating all quantities.")

        if likfunc == 'mark1':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', dontread=noGmat))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat',
            #    dontread=memsave))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF'))
            self.GtD = np.array(h5df.getData(self.name, 'pic_GtD'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF',
                #dontread=(memsave and not self.twoComponentNoise)))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            #self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat', dontread=memsave))
            self.Dvec = np.array(h5df.getData(self.name, 'pic_Dvec', dontread=memsave))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', dontread=memsave))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark2':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark3' or likfunc == 'mark3fa':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(not self.twoComponentNoise)))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark4':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.UtF = np.array(h5df.getData(self.name, 'pic_UtF'))
            self.UtD = np.array(h5df.getData(self.name, 'pic_UtD', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGU = np.array(h5df.getData(self.name, 'pic_AGU',
                dontread=(not self.twoComponentNoise)))
            #self.AoGU = np.array(h5df.getData(self.name, 'pic_AoGU',
            #    dontread=(memsave and not self.twoComponentNoise)))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))
            self.Umat = np.array(h5df.getData(self.name, 'pic_Umat'))
            self.Jweight = np.array(h5df.getData(self.name, 'pic_Jweight'))
            self.Uimat = np.array(h5df.getData(self.name, 'pic_Uimat'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))

        if likfunc == 'mark4ln':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.UtF = np.array(h5df.getData(self.name, 'pic_UtF', dontread=memsave))
            self.UtD = np.array(h5df.getData(self.name, 'pic_UtD'))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', dontread=memsave))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', dontread=memsave))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.UtFF = np.array(h5df.getData(self.name, 'pic_UtFF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGU = np.array(h5df.getData(self.name, 'pic_AGU',
                dontread=(not self.twoComponentNoise)))
            #self.AoGU = np.array(h5df.getData(self.name, 'pic_AoGU',
            #    dontread=(memsave and not self.twoComponentNoise)))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))
            self.Umat = np.array(h5df.getData(self.name, 'pic_Umat'))
            self.Jweight = np.array(h5df.getData(self.name, 'pic_Jweight'))
            self.Uimat = np.array(h5df.getData(self.name, 'pic_Uimat'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))

        if likfunc == 'mark6' or likfunc == 'mark6fa':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            #self.GGtD = np.array(h5df.getData(self.name, 'pic_GGtD', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat'))
            #self.GGtE = np.array(h5df.getData(self.name, 'pic_GGtE', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE',
                dontread=(not self.twoComponentNoise)))
            #self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGD = np.array(h5df.getData(self.name, 'pic_AoGD',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGE = np.array(h5df.getData(self.name, 'pic_AoGE',
            #    dontread=(memsave and not self.twoComponentNoise)))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            #self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat', dontread=memsave))
            self.Dvec = np.array(h5df.getData(self.name, 'pic_Dvec', dontread=memsave))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', dontread=memsave))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark7':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF',
            #    dontread=(memsave and not self.twoComponentNoise)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark8':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            #self.GGtD = np.array(h5df.getData(self.name, 'pic_GGtD', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat', dontread=memsave))
            #self.GGtE = np.array(h5df.getData(self.name, 'pic_GGtE', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE',
                dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGD = np.array(h5df.getData(self.name, 'pic_AoGD',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGE = np.array(h5df.getData(self.name, 'pic_AoGE',
            #    dontread=(memsave and not self.twoComponentNoise)))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            #self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat'))
            self.Dvec = np.array(h5df.getData(self.name, 'pic_Dvec'))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark9':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', dontread=memsave))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', dontread=memsave))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGFF = np.array(h5df.getData(self.name, 'pic_AGFF',
                dontread=(not self.twoComponentNoise)))
            #self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGFF = np.array(h5df.getData(self.name, 'pic_AoGFF',
            #    dontread=(memsave and not self.twoComponentNoise)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark10':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave))
            #self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
            #        dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            #self.GGtD = np.array(h5df.getData(self.name, 'pic_GGtD', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat'))
            #self.GGtE = np.array(h5df.getData(self.name, 'pic_GGtE', dontread=memsave))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', dontread=memsave))
            self.SFdmmat = np.array(h5df.getData(self.name, 'pic_SFdmmat', dontread=memsave))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', dontread=memsave))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.DSF = np.array(h5df.getData(self.name, 'pic_DSF', dontread=memsave))
            self.DFF = np.array(h5df.getData(self.name, 'pic_DFF'))
            self.EEmat = np.array(h5df.getData(self.name, 'pic_EEmat'))
            self.GGtEE = np.array(h5df.getData(self.name, 'pic_GGtEE'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGFF = np.array(h5df.getData(self.name, 'pic_AGFF',
                dontread=(not self.twoComponentNoise)))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE',
                dontread=(not self.twoComponentNoise)))
            self.AGEE = np.array(h5df.getData(self.name, 'pic_AGEE',
                dontread=(not self.twoComponentNoise)))
            #self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGFF = np.array(h5df.getData(self.name, 'pic_AoGFF',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGD = np.array(h5df.getData(self.name, 'pic_AoGD',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGE = np.array(h5df.getData(self.name, 'pic_AoGE',
            #    dontread=(memsave and not self.twoComponentNoise)))
            #self.AoGEE = np.array(h5df.getData(self.name, 'pic_AoGEE',
            #    dontread=(memsave and not self.twoComponentNoise)))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            #self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat'))
            self.Dvec = np.array(h5df.getData(self.name, 'pic_Dvec'))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'gibbs1':
            self.Zmat = np.array(h5df.getData(self.name, 'pic_Zmat'))



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

                if not likfunc in ['mark1', 'mark2', 'mark3', 'mark3fa', 'mark4', 'mark7', 'mark9']:
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

                if not likfunc in ['mark1', 'mark2', 'mark3', 'mark3fa', 'mark4', 'mark7', 'mark9']:
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
    # The DataFile object
    h5df = None

    # The ptaPulsar objects
    ptapsrs = []

    # The model/signals description
    ptasignals = []

    dimensions = 0
    pmin = None
    pmax = None
    pstart = None
    pwidth = None
    pardes = None
    haveStochSources = False
    haveDetSources = False

    # What likelihood function to use
    likfunc = 'mark3'

    # Whether we evaluate the complement of the compressed likelihood
    compression = 'None'
    evallikcomp = False

    # Whether we use the option of forcing the frequency lines to be ordered in
    # the prior
    orderFrequencyLines = False

    # Additional informative quantities (reset after RJMCMC jump)
    npf = None      # Number of frequencies per pulsar (red noise/signal)
    npff = None     # Number of frequencies per pulsar (single-freq components)
    npfdm = None    # Number of frequencies per pulsar (DM)
    npe = None      # Number of frequencies per pulsar (rn + DM)
    npobs = None    # Number of observations per pulsar
    npgs = None     # Number of non-projected observations per pulsar (columns Hmat)
    npgos = None    # Number of orthogonal non-projected observations per pulsar (columns Homat)
    npu = None      # Number of avetoas per pulsar
    npm = None      # Number of columns of design matrix (full design matrix)
    Tmax = None     # One Tmax to rule them all...

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
    rGZ = None          #                              gibbs
    FGGNGGF = None      #        mark3, mark?
    EGGNGGE = None      #                      mark6
    ZGGNGGZ = None      #                              gibbs
    NGGF = None         #        mark3, mark?  mark6

    # Whether we have already called the likelihood in one call, so we can skip
    # some things in comploglikelihood
    skipUpdateToggle = False


    """
    Constructor. Read data/model if filenames are given

    @param h5filename:      HDF5 filename with pulsar data
    @param jsonfilename:    JSON file with model
    @param pulsars:         Which pulsars to read ('all' = all, otherwise provide a
                            list: ['J0030+0451', 'J0437-4715', ...])
                            WARNING: duplicates are _not_ checked for.
    """
    def __init__(self, h5filename=None, jsonfilename=None, pulsars='all', \
                 auxFromFile=True, noGmat=False, verbose=False, noCreate=False):
        self.clear()

        if h5filename is not None:
            self.initFromFile(h5filename, pulsars=pulsars)

            if jsonfilename is not None:
                self.initModelFromFile(jsonfilename, auxFromFile=auxFromFile, \
                                       noGmat=noGmat, verbose=verbose, \
                                       noCreate=noCreate)

    """
    Clear all the structures present in the object
    """
    # TODO: Do we need to delete all with 'del'?
    def clear(self):
        self.h5df = None
        self.ptapsrs = []
        self.ptasignals = []

        self.dimensions = 0
        self.pmin = None
        self.pmax = None
        self.pstart = None
        self.pwidth = None
        self.likfunc = 'mark3'
        self.orderFrequencyLines = False
        self.haveStochSources = False
        self.haveDetSources = False
        self.skipUpdateToggle = False


    """
    Initialise this likelihood object from an HDF5 file

    @param filename:    Name of the HDF5 file we will be reading
    @param pulsars:     Which pulsars to read ('all' = all, otherwise provide a
                        list: ['J0030+0451', 'J0437-4715', ...])
                        WARNING: duplicates are _not_ checked for.
    @param append:      If set to True, do not delete earlier read-in pulsars
    """
    def initFromFile(self, filename, pulsars='all', append=False):
        # Retrieve the pulsar list
        self.h5df = DataFile(filename)
        psrnames = self.h5df.getPulsarList()

        # Determine which pulsars we are reading in
        readpsrs = []
        if pulsars=='all':
            readpsrs = psrnames
        else:
            # Check if all provided pulsars are indeed in the HDF5 file
            if np.all(np.array([pulsars[ii] in psrnames for ii in range(len(pulsars))]) == True):
                readpsrs = pulsars
            elif pulsars in destpsrnames:
                pulsars = [pulsars]
                readpsrs = pulsars
            else:
                raise ValueError("ERROR: Not all provided pulsars in HDF5 file")

        # Free earlier pulsars if we are not appending
        if not append:
            self.ptapsrs = []

        # Initialise all pulsars
        for psrname in readpsrs:
            newpsr = ptaPulsar()
            newpsr.readFromH5(self.h5df, psrname)
            self.ptapsrs.append(newpsr)


    """
    Note: This function is not (yet) ready for use
    """
    def addSignalFourierCoeffOldOld(self, psrind, index, Tmax, isDM=False):
        if isDM:
            newsignal = dict({
                'stype':'dmfouriercoeff',
                'npars':len(self.ptapsrs[0].Fdmfreqs),
                'ntotpars':len(self.ptapsrs[0].Fdmfreqs),
                'bvary':np.array([1]*newsignal.ntotpars, dtype=np.bool),
                'corr':'single',
                'Tmax':Tmax,
                'parindex':index
                })
        else:
            newsignal = dict({
                'stype':'dmfouriercoeff',
                'npars':len(self.ptapsrs[0].Ffreqs),
                'ntotpars':len(self.ptapsrs[0].Ffreqs),
                'bvary':np.array([1]*newsignal.ntotpars, dtype=np.bool),
                'corr':'single',
                'Tmax':Tmax,
                'parindex':index
                })

        # Since this parameter space is so large, calculate the
        # best first-estimate values of these quantities
        # We assume that many auxiliaries have been set already (is done
        # in initModel, so should be ok)
        # TODO: check whether this works, and make smarter
        npars = newsignal['npars']
        psr = self.ptapsrs[newsignal['pulsarind']]

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



    """
    Add a signal to the internal description data structures, based on a signal
    dictionary

    @param signal:  The signal dictionary we will add to the list
    @param index:   The index of the first par in the global par list
    @param Tmax:    The total time-baseline we use for this signal
    """
    def addSignal(self, signal, index=0, Tmax=None):
        # Assert that the necessary keys are present
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # Determine the time baseline of the array of pulsars
        if not 'Tmax' in signal and Tmax is None:
            Tstart = np.min(self.ptapsrs[0].toas)
            Tfinish = np.max(self.ptapsrs[0].toas)
            for m2psr in self.ptapsrs:
                Tstart = np.min([np.min(m2psr.toas), Tstart])
                Tfinish = np.max([np.max(m2psr.toas), Tfinish])
            Tmax = Tfinish - Tstart
        elif 'Tmax' in signal:
            Tmax = signal['Tmax']
            

        # Adjust some basic details about the signal
        signal['Tmax'] = Tmax
        signal['parindex'] = index

        # Convert a couple of values
        signal['bvary'] = np.array(signal['bvary'], dtype=np.bool)
        signal['npars'] = np.sum(signal['bvary'])
        signal['ntotpars'] = len(signal['bvary'])
        signal['pmin'] = np.array(signal['pmin'])
        signal['pmax'] = np.array(signal['pmax'])
        signal['pwidth'] = np.array(signal['pwidth'])
        signal['pstart'] = np.array(signal['pstart'])


        # Add the signal
        if signal['stype']=='efac':
            # Efac
            self.addSignalEfac(signal)
        elif signal['stype'] in ['equad', 'jitter']:
            # Equad or Jitter
            self.addSignalEquad(signal)
        elif signal['stype'] in ['powerlaw', 'spectrum', 'spectralModel']:
            # Any time-correlated signal
            self.addSignalTimeCorrelated(signal)
            self.haveStochSources = True
        elif signal['stype'] in ['dmpowerlaw', 'dmspectrum']:
            # A DM variation signal
            self.addSignalDMV(signal)
            self.haveStochSources = True
        elif signal['stype'] == 'frequencyline':
            # Single free-floating frequency line
            psrSingleFreqs = self.getNumberOfSignals(stype='frequencyline', \
                    corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveStochSources = True
        elif signal['stype'] == 'dmfrequencyline':
            # Single free-floating frequency line
            psrSingleFreqs = self.getNumberOfSignals(stype='dmfrequencyline', \
                    corr='single')
            signal['npsrfreqindex'] = psrSingleFreqs[signal['pulsarind']]
            self.addSignalFrequencyLine(signal)
            self.haveStochSources = True
        elif signal['stype'] == 'bwm':
            # A burst with memory
            self.addSignalBWM(signal)
            self.haveDetSources = True
        elif signal['stype'] == 'lineartimingmodel':
            # A Tempo2 linear timing model, except for (DM)QSD parameters
            self.addSignalTimingModel(signal)
            self.haveDetSources = True
        elif signal['stype'] == 'nonlineartimingmodel':
            # A Tempo2 timing model, except for (DM)QSD parameters
            # Note: libstempo must be installed
            self.addSignalTimingModel(signal, linear=False)
            self.haveDetSources = True
        elif signal['stype'] == 'fouriermode':
            self.addSignalFourierMode(signal)
            self.ptapsrs[signal['pulsarind']].fourierind = index
        elif signal['stype'] == 'dmfouriermode':
            self.addSignalFourierMode(signal)
            self.ptapsrs[signal['pulsarind']].dmfourierind = index
        elif signal['stype'] == 'jitterfouriermode':
            self.addSignalFourierMode(signal)
            self.ptapsrs[signal['pulsarind']].jitterfourierind = index
        else:
            # Some other unknown signal
            self.ptasignals.append(signal)



    """
    Add an EFAC signal

    Required keys in signal
    @param psrind:      Index of the pulsar this efac applies to
    @param index:       Index of first parameter in total parameters array
    @param flagname:    Name of the flag this efac applies to (field-name)
    @param flagvalue:   Value of the flag this efac applies to (e.g. CPSR2)
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters

    # TODO: make prior flat in log?
    """
    def addSignalEfac(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in efac signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        signal['Nvec'] = self.ptapsrs[signal['pulsarind']].toaerrs**2

        if signal['flagname'] != 'pulsarname':
            # This efac only applies to some TOAs, not all of 'm
            ind = np.array(self.ptapsrs[signal['pulsarind']].flags) != signal['flagvalue']
            signal['Nvec'][ind] = 0.0

        self.ptasignals.append(signal.copy())

    """
    Add an EQUAD or jitter signal

    Required keys in signal
    @param stype:       Either 'jitter' or 'equad'
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param flagname:    Name of the flag this efac applies to (field-name)
    @param flagvalue:   Value of the flag this efac applies to (e.g. CPSR2)
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """
    def addSignalEquad(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'flagname', 'flagvalue', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in equad signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        psr = self.ptapsrs[signal['pulsarind']]
        signal['Nvec'] = np.ones(len(psr.toaerrs))

        if signal['stype'] == 'jitter':
            signal['Jvec'] = np.ones(len(psr.avetoas))

        if signal['flagname'] != 'pulsarname':
            # This equad only applies to some TOAs, not all of 'm
            ind = np.array(psr.flags) != signal['flagvalue']
            signal['Nvec'][ind] = 0.0

            if signal['stype'] == 'jitter':
                signal['Jvec']= selection_to_dselection(signal['Nvec'],
                        psr.Umat)

        self.ptasignals.append(signal.copy())


    """
    Add a single frequency line signal

    Required keys in signal
    @param stype:       Either 'frequencyline' or 'dmfrequencyline'
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param freqindex:   If there are several of these sources, which is this?
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """
    def addSignalFrequencyLine(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in frequency line signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        self.ptasignals.append(signal.copy())


    """
    Add some time-correlated signal

    Required keys in signal
    @param stype:       Either 'spectrum', 'powerlaw', or 'spectralModel'
    @param corr:        Either 'single', 'uniform', 'dipole', 'gr', ...
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param Tmax         Time baseline of the entire experiment
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    @param lAniGWB:     In case of an anisotropic GWB, this sets the order of
                        anisotropy (default=2, also for all other signals)
    """
    def addSignalTimeCorrelated(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex', 'Tmax']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))
        if 'stype' == 'anisotropicgwb':
            if not 'lAniGWB' in signal:
                raise ValueError("ERROR: Missing lAniGWB key in signal")
        elif 'stype' == 'pixelgwb':
            if not 'npixels' in signal:
                raise ValueError("ERROR: Missing npixels key in signal")

        if signal['corr'] == 'gr':
            # Correlated with the Hellings \& Downs matrix
            signal['corrmat'] = hdcorrmat(self.ptapsrs)
        elif signal['corr'] == 'uniform':
            # Uniformly correlated (Clock signal)
            signal['corrmat'] = np.ones((len(self.ptapsrs), len(self.ptapsrs)))
        elif signal['corr'] == 'dipole':
            # Dipole correlations (SS Ephemeris)
            signal['corrmat'] = dipolecorrmat(self.ptapsrs)
        elif signal['corr'] == 'anisotropicgwb':
            # Anisotropic GWB correlations
            signal['aniCorr'] = aniCorrelations(self.ptapsrs, signal['lAniGWB'])
        elif signal['corr'] == 'pixelgwb':
            # Pixel based anisotropic GWB correlations
            signal['aniCorr'] = pixelCorrelations(self.ptapsrs, signal['npixels'])

        if signal['corr'] != 'single':
            # Also fill the Ffreqs array, since we are dealing with correlations
            numfreqs = np.array([len(self.ptapsrs[ii].Ffreqs) \
                    for ii in range(len(self.ptapsrs))])
            ind = np.argmax(numfreqs)
            signal['Ffreqs'] = self.ptapsrs[ind].Ffreqs.copy()

        self.ptasignals.append(signal.copy())

    """
    Add some DM variation signal

    Required keys in signal
    @param stype:       Either 'spectrum', 'powerlaw', or 'spectralModel'
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param Tmax         Time baseline of the entire experiment
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """
    def addSignalDMV(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex', 'Tmax']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in DMV signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        self.ptasignals.append(signal.copy())


    """
    Add a burst with memory signal

    Required keys in signal
    @param stype:       Basically always 'bwm'
    @param psrind:      Index of the pulsar this signal applies to (Earth-burst: -1)
    @param index:       Index of first parameter in total parameters array
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """
    def addSignalBWM(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in BWM signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        self.ptasignals.append(signal.copy())

    """
    Add a signal that represents a numerical tempo2 timing model

    Required keys in signal
    @param stype:       Basically always 'lineartimingmodel' (TODO: include nonlinear)
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    @param parid:       The identifiers (as used in par-file) that identify
                        which parameters are included
    """
    def addSignalTimingModel(self, signal, linear=True):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', 'parid', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in TimingModel signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # Assert that this signal applies to a pulsar
        if signal['pulsarind'] < 0 or signal['pulsarind'] >= len(self.ptapsrs):
            raise ValueError("ERROR: timingmodel signal applied to non-pulsar ({0})".format(signal['pulsarind']))

        # Check that the parameters included here are also present in the design
        # matrix
        for ii, parid in enumerate(signal['parid']):
            if not parid in self.ptapsrs[signal['pulsarind']].ptmdescription:
                raise ValueError("ERROR: timingmodel signal contains non-valid parameter id")

        # If this is a non-linear signal, make sure to initialise the libstempo
        # object
        if linear == False:
            self.ptapsrs[signal['pulsarind']].initLibsTempoObject()

        self.ptasignals.append(signal.copy())


    """
    Add a signal that represents a the Fourier coefficients

    Required keys in signal
    @param stype:       Basically always 'lineartimingmodel' (TODO: include nonlinear)
    @param psrind:      Index of the pulsar this signal applies to
    @param index:       Index of first parameter in total parameters array
    @param bvary:       List of indicators, specifying whether parameters can vary
    @param pmin:        Minimum bound of prior domain
    @param pmax:        Maximum bound of prior domain
    @param pwidth:      Typical width of the parameters (e.g. initial stepsize)
    @param pstart:      Typical start position for the parameters
    """
    def addSignalFourierMode(self, signal):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex', 'Tmax']
        if not all(k in signal for k in keys):
            raise ValueError("ERROR: Not all signal keys are present in TimingModel signal. Keys: {0}. Required: {1}".format(signal.keys(), keys))

        # Assert that this signal applies to a pulsar
        if signal['pulsarind'] < 0 or signal['pulsarind'] >= len(self.ptapsrs):
            raise ValueError("ERROR: Fourier coefficient signal applied to non-pulsar ({0})".format(signal['pulsarind']))

        self.ptasignals.append(signal.copy())

    
    """
    Find the number of signals per pulsar matching some criteria, given a list
    of signal dictionaries. Main use is, for instance, to find the number of
    free frequency lines per pulsar given the signal model dictionary.

    @param signals: Dictionary of all signals
    @param stype:   The signal type that must be matched
    @param corr:    Signal correlation that must be matched
    """
    def getNumberOfSignalsFromDict(self, signals, stype='powerlaw', corr='single'):
        psrSignals = np.zeros(len(self.ptapsrs), dtype=np.int)

        for ii, signal in enumerate(signals):
            if signal['stype'] == stype and signal['corr'] == corr:
                if signal['pulsarind'] == -1:
                    psrSignals[:] += 1
                else:
                    psrSignals[signal['pulsarind']] += 1

        return psrSignals

    """
    Find the number of signals per pulsar matching some criteria in the current
    signal list.

    @param stype:   The signal type that must be matched
    @param corr:    Signal correlation that must be matched
    """
    def getNumberOfSignals(self, stype='powerlaw', corr='single'):
        return self.getNumberOfSignalsFromDict(self.ptasignals, stype, corr)

    """
    Find the signal numbers of a certain type and correlation

    @param signals: Dictionary of all signals
    @param stype:   The signal type that must be matched
    @param corr:    Signal correlation that must be matched
    @param psrind:  Pulsar index that must be matched (-2 means all)

    @return:        Index array with signals that qualify
    """
    def getSignalNumbersFromDict(self, signals, stype='powerlaw', \
            corr='single', psrind=-2):
        signalNumbers = []

        for ii, signal in enumerate(signals):
            if signal['stype'] == stype and signal['corr'] == corr:
                if psrind == -2:
                    signalNumbers.append(ii)
                elif signal['pulsarind'] == psrind:
                    signalNumbers.append(ii)

        return np.array(signalNumbers, dtype=np.int)


    """
    Check the read-in signal dictionary. Reject improperly defined models

    TODO: Actually implement the checks
    """
    def checkSignalDictionary(self, signals):
        return True


    """
    Allocate memory for the ptaLikelihood attribute matrices that we'll need in
    the likelihood function.  This function does not perform any calculations,
    although it does initialise the 'counter' integer arrays like npf and npgs.
    """
    # TODO: see if we can implement the RJMCMC for the Fourier modes
    # TODO: these quantities should depend on the model, not the likelihood function
    def allocateLikAuxiliaries(self):
        # First figure out how large we have to make the arrays
        npsrs = len(self.ptapsrs)
        self.npf = np.zeros(npsrs, dtype=np.int)
        self.npu = np.zeros(npsrs, dtype=np.int)
        self.npff = np.zeros(npsrs, dtype=np.int)
        self.npfdm = np.zeros(npsrs, dtype=np.int)
        self.npffdm = np.zeros(npsrs, dtype=np.int)
        self.npobs = np.zeros(npsrs, dtype=np.int)
        self.npgs = np.zeros(npsrs, dtype=np.int)
        self.npgos = np.zeros(npsrs, dtype=np.int)
        self.npm = np.zeros(npsrs, dtype=np.int)
        for ii, psr in enumerate(self.ptapsrs):
            if not self.likfunc in ['mark2']:
                self.npf[ii] = len(self.ptapsrs[ii].Ffreqs)
                self.npff[ii] = self.npf[ii]

            if self.likfunc in ['mark4ln', 'mark9', 'mark10']:
                self.npff[ii] += len(self.ptapsrs[ii].SFfreqs)

            #if self.likfunc in ['mark1', 'mark3', 'mark4', 'mark4ln', 'mark11']:
            self.npu[ii] = len(self.ptapsrs[ii].avetoas)

            if self.likfunc in ['mark1', 'mark4', 'mark4ln', 'mark6', \
                    'mark6fa', 'mark8', 'mark10', 'gibbs1']:
                self.npfdm[ii] = len(psr.Fdmfreqs)
                self.npffdm[ii] = len(psr.Fdmfreqs)

            if self.likfunc in ['mark10']:
                self.npffdm[ii] += len(psr.SFdmfreqs)

            self.npobs[ii] = len(psr.toas)
            psr.Nvec = np.zeros(len(psr.toas))
            psr.Jvec = np.zeros(len(psr.avetoas))

            if self.likfunc in ['mark1', 'mark2', 'mark3', 'mark3fa', 'mark4', \
                    'mark4ln', 'mark6', 'mark6fa', 'mark7', 'mark8', 'mark9', \
                    'mark10', 'gibbs1']:
                self.npgs[ii] = len(psr.toas) - psr.Hcmat.shape[1] # (Hc = orth(M) )
                self.npgos[ii] = len(psr.toas) - self.npgs[ii]
                psr.Nwvec = np.zeros(self.npgs[ii])
                psr.Nwovec = np.zeros(self.npgos[ii])

            if self.likfunc[:5] in ['gibbs']:
                self.npm[ii] = psr.Mmat.shape[1]

        self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        self.Phivec = np.zeros(np.sum(self.npf))
        self.Thetavec = np.zeros(np.sum(self.npfdm))
        self.Muvec = np.zeros(np.sum(self.npu))

        if self.likfunc == 'mark1':
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)

            self.Gr = np.zeros(np.sum(self.npgs))
            self.GCG = np.zeros((np.sum(self.npgs), np.sum(self.npgs)))
        elif self.likfunc == 'mark2':
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
        elif self.likfunc == 'mark3' or self.likfunc == 'mark7' \
                or self.likfunc == 'mark3fa':
            self.Sigma = np.zeros((np.sum(self.npf), np.sum(self.npf)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGF = np.zeros(np.sum(self.npf))
            self.FGGNGGF = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        elif self.likfunc == 'mark4':
            self.Sigma = np.zeros((np.sum(self.npu), np.sum(self.npu)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGU = np.zeros(np.sum(self.npu))
            self.UGGNGGU = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        elif self.likfunc == 'mark4ln':
            self.Sigma = np.zeros((np.sum(self.npu), np.sum(self.npu)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGU = np.zeros(np.sum(self.npu))
            self.UGGNGGU = np.zeros((np.sum(self.npu), np.sum(self.npu)))
        elif self.likfunc == 'mark6' or self.likfunc == 'mark8' \
                or self.likfunc == 'mark6fa':
            self.Sigma = np.zeros((np.sum(self.npf)+np.sum(self.npfdm), \
                    np.sum(self.npf)+np.sum(self.npfdm)))
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGE = np.zeros(np.sum(self.npf)+np.sum(self.npfdm))
            self.EGGNGGE = np.zeros((np.sum(self.npf)+np.sum(self.npfdm), np.sum(self.npf)+np.sum(self.npfdm)))
        elif self.likfunc == 'mark9':
            self.Sigma = np.zeros((np.sum(self.npff), np.sum(self.npff)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGF = np.zeros(np.sum(self.npff))
            self.FGGNGGF = np.zeros((np.sum(self.npff), np.sum(self.npff)))
        elif self.likfunc == 'mark10':
            self.Sigma = np.zeros((np.sum(self.npff)+np.sum(self.npffdm), \
                    np.sum(self.npff)+np.sum(self.npffdm)))
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
            self.rGE = np.zeros(np.sum(self.npff)+np.sum(self.npffdm))
            self.EGGNGGE = np.zeros((np.sum(self.npff)+np.sum(self.npffdm), \
                    np.sum(self.npff)+np.sum(self.npffdm)))
        elif self.likfunc == 'mark11':
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
        elif self.likfunc == 'gibbs1':
            zlen = np.sum(self.npm) + np.sum(self.npf)
            self.Sigma = np.zeros((zlen, zlen))
            self.GNGldet = np.zeros(npsrs)
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.rGZ = np.zeros(zlen)

            self.rGr = np.zeros(npsrs)
            self.ZGGNGGZ = np.zeros((zlen, zlen))


    """
    Based on somewhat simpler quantities, this function makes a full model
    dictionary. Standard use would be to save that dictionary in a json file, so
    that it does not need to be manually created by the user

    @param nfreqmodes:      blah
    @param dmnfreqmodes:    blah
    """
    def makeModelDict(self,  nfreqs=20, ndmfreqs=None, \
            Tmax=None, \
            incRedNoise=False, noiseModel='powerlaw', fc=None, \
            incDM=False, dmModel='powerlaw', \
            incClock=False, clockModel='powerlaw', \
            incGWB=False, gwbModel='powerlaw', \
            incDipole=False, dipoleModel='powerlaw', \
            incAniGWB=False, anigwbModel='powerlaw', lAniGWB=1, \
            incPixelGWB=False, pixelgwbModel='powerlaw', npixels=4, \
            incBWM=False, \
            incTimingModel=False, nonLinear=False, \
            varyEfac=True, incEquad=False, \
            separateCEquads=False, separateEquads=False, \
            separateEfacs=False, \
            incCEquad=False, \
            incJitter=False, \
            incSingleFreqNoise=False, \
                                        # True
            singlePulsarMultipleFreqNoise=None, \
                                        # [True, ..., False]
            multiplePulsarMultipleFreqNoise=None, \
                                        # [0, 3, 2, ..., 4]
            dmFrequencyLines=None, \
                                        # [0, 3, 2, ..., 4]
            orderFrequencyLines=False, \
            compression = 'frequencies', \
            evalCompressionComplement = True, \
            likfunc='mark1'):
        # We have to determine the number of frequencies we'll need
        numNoiseFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)
        numDMFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)
        numSingleFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)
        numSingleDMFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)

        # Figure out what the frequencies per pulsar are
        for pindex, m2psr in enumerate(self.ptapsrs):
            if incDM:
                if ndmfreqs is None or ndmfreqs=="None":
                    ndmfreqs = nfreqs
            else:
                ndmfreqs = 0

            if incSingleFreqNoise:
                numSingleFreqs[pindex] = 1
            elif singlePulsarMultipleFreqNoise is not None:
                if singlePulsarMultipleFreqNoise[pindex]:
                    numSingleFreqs[pindex] = 1
            elif multiplePulsarMultipleFreqNoise is not None:
                numSingleFreqs[pindex] = multiplePulsarMultipleFreqNoise[pindex]

            if dmFrequencyLines is not None:
                numSingleDMFreqs[pindex] = dmFrequencyLines[pindex]

            numNoiseFreqs[pindex] = nfreqs
            numDMFreqs[pindex] = ndmfreqs

        signals = []

        for ii, m2psr in enumerate(self.ptapsrs):
            if separateEfacs:
                uflagvals = list(set(m2psr.flags))  # Unique flags
                for flagval in uflagvals:
                    newsignal = OrderedDict({
                        "stype":"efac",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"efacequad",
                        "flagvalue":flagval,
                        "bvary":[varyEfac],
                        "pmin":[0.001],
                        "pmax":[50.0],
                        "pwidth":[0.1],
                        "pstart":[1.0]
                        })
                    signals.append(newsignal)
            else:
                newsignal = OrderedDict({
                    "stype":"efac",
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":[varyEfac],
                    "pmin":[0.001],
                    "pmax":[50.0],
                    "pwidth":[0.1],
                    "pstart":[1.0]
                    })
                signals.append(newsignal)

            if incEquad:
                if separateEquads:
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype":"equad",
                            "corr":"single",
                            "pulsarind":ii,
                            "flagname":"efacequad",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.1],
                            "pstart":[-8.0]
                            })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype":"equad",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":[True],
                        "pmin":[-10.0],
                        "pmax":[-4.0],
                        "pwidth":[0.1],
                        "pstart":[-8.0]
                        })
                    signals.append(newsignal)

            if incCEquad or incJitter:
                if separateCEquads:
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype":"jitter",
                            "corr":"single",
                            "pulsarind":ii,
                            "flagname":"efacequad",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.1],
                            "pstart":[-8.0]
                            })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype":"jitter",
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":[True],
                        "pmin":[-10.0],
                        "pmax":[-4.0],
                        "pwidth":[0.1],
                        "pstart":[-8.0]
                        })
                    signals.append(newsignal)

            if incRedNoise:
                if noiseModel=='spectrum':
                    nfreqs = numNoiseFreqs[ii]
                    bvary = [True]*nfreqs
                    pmin = [-18.0]*nfreqs
                    pmax = [-7.0]*nfreqs
                    pstart = [-10.0]*nfreqs
                    pwidth = [0.1]*nfreqs
                elif noiseModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-20.0, 0.02, 1.0e-11]
                    pmax = [-10.0, 6.98, 3.0e-9]
                    pstart = [-14.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                elif noiseModel=='spectralModel':
                    bvary = [True, True, True]
                    pmin = [-28.0, 0.0, -4.0]
                    pmax = [-14.0, 12.0, 2.0]
                    pstart = [-22.0, 2.0, -1.0]
                    pwidth = [-0.2, 0.1, 0.1]
                else:
                    raise ValueError("ERROR: option {0} not known".
                            format(noiseModel))

                newsignal = OrderedDict({
                    "stype":noiseModel,
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart
                    })
                signals.append(newsignal)

            if incDM:
                if dmModel=='dmspectrum':
                    nfreqs = numDMFreqs[ii]
                    bvary = [True]*nfreqs
                    pmin = [-14.0]*nfreqs
                    pmax = [-3.0]*nfreqs
                    pstart = [-7.0]*nfreqs
                    pwidth = [0.1]*nfreqs
                    #dmModel = 'dmspectrum'
                elif dmModel=='dmpowerlaw':
                    bvary = [True, True, False]
                    pmin = [-14.0, 0.02, 1.0e-11]
                    pmax = [-6.5, 6.98, 3.0e-9]
                    pstart = [-13.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    #dmModel = 'dmpowerlaw'
                else:
                    raise ValueError("ERROR: option {0} not known".
                            format(dmModel))

                newsignal = OrderedDict({
                    "stype":dmModel,
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart
                    })
                signals.append(newsignal)

            for jj in range(numSingleFreqs[ii]):
                newsignal = OrderedDict({
                    "stype":'frequencyline',
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":[True, True],
                    "pmin":[-9.0, -18.0],
                    "pmax":[-5.0, -9.0],
                    "pwidth":[-0.1, -0.1],
                    "pstart":[-7.0, -10.0]
                    })
                signals.append(newsignal)

            for jj in range(numSingleDMFreqs[ii]):
                newsignal = OrderedDict({
                    "stype":'dmfrequencyline',
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":[True, True],
                    "pmin":[-9.0, -18.0],
                    "pmax":[-5.0, -9.0],
                    "pwidth":[-0.1, -0.1],
                    "pstart":[-7.0, -10.0]
                    })
                signals.append(newsignal)

            if incTimingModel or likfunc == 'mark11':
                if nonLinear:
                    # Get the parameter errors from libstempo. Initialise the
                    # libstempo object
                    m2psr.initLibsTempoObject()

                    errs = []
                    est = []
                    for t2par in m2psr.t2psr.pars:
                        errs += [m2psr.t2psr[t2par].err]
                        est += [m2psr.t2psr[t2par].val]
                    tmperrs = np.array([0.0] + errs)
                    tmpest = np.array([0.0] + est)
                else:
                    # TODO: Get these values from the HDF5 file. However, for
                    # compatibility, now not yet

                    # Just do the timing-model fit ourselves here, in order to set
                    # the prior.
                    w = 1.0 / m2psr.toaerrs**2
                    Sigi = np.dot(m2psr.Mmat.T, (w * m2psr.Mmat.T).T)
                    try:
                        cf = sl.cho_factor(Sigi)
                        Sigma = sl.cho_solve(cf, np.eye(Sigi.shape[0]))
                    except np.linalg.LinAlgError:
                        U, s, Vh = sl.svd(Sigi)
                        if not np.all(s > 0):
                            raise ValueError("Sigi singular according to SVD")
                        Sigma = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
                    tmperrs = np.sqrt(np.diag(Sigma))
                    #tmperrs = m2psr.ptmparerrs
                    tmpest = m2psr.ptmpars

                # Figure out which parameters we'll keep in the design matrix
                jumps = []
                for tmpar in m2psr.ptmdescription:
                    if tmpar[:4] == 'JUMP':
                        jumps += [tmpar]

                #newptmdescription = m2psr.getNewTimingModelParameterList(keep=True, \
                #        tmpars=['Offset', 'F0', 'F1', 'RAJ', 'DECJ', 'PMRA', \
                #        'PMDEC', 'PX', 'DM', 'DM1', 'DM2'] + jumps)
                newptmdescription = m2psr.getNewTimingModelParameterList(keep=True, \
                        tmpars=['Offset', 'F0', 'F1', 'DM1', 'DM2'] + jumps)
                # Create a modified design matrix (one that we will analytically
                # marginalise over).
                #(newM, newG, newGc, newptmpars, newptmdescription) = \
                #        m2psr.getModifiedDesignMatrix(removeAll=True)

                # Actually, if we are in likfunc == 'mark11', we need _all_
                # TM parameters in the model
                if likfunc == 'mark11':
                    newptmdescription = []

                # Select the numerical parameters. These are the ones not
                # present in the quantities that getModifiedDesignMatrix
                # returned
                parids=[]
                bvary = []
                pmin = []
                pmax = []
                pwidth = []
                pstart = []
                for jj, parid in enumerate(m2psr.ptmdescription):
                    if not parid in newptmdescription:
                        parids += [parid]
                        bvary += [True]
                        if parid == 'SINI':
                            pmin += [-1.0]
                            pmax += [1.0]
                            pwidth += [0.1]
                        elif parid == 'ECC':
                            pmin += [0.0]
                            pmax += [1.0]
                            pwidth += [0.1]
                        else:
                            # tmpest[jj] = 0.0        # DELETE THISS!!!!!
                            pmin += [-500.0 * tmperrs[jj] + tmpest[jj]]
                            pmax += [500.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [(pmax[-1]-pmin[-1])/50.0]
                        pstart += [tmpest[jj]]

                if nonLinear:
                    stype = 'nonlineartimingmodel'
                else:
                    stype = 'lineartimingmodel'

                newsignal = OrderedDict({
                    "stype":stype,
                    "corr":"single",
                    "pulsarind":ii,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "parid":parids
                    })
                signals.append(newsignal)

            if likfunc == 'mark11':
                nmodes = 2*numNoiseFreqs[ii]
                bvary = [True]*nmodes
                pmin = [-1.0e-3]*nmodes
                pmax = [1.0e-3]*nmodes
                pstart = [0.0]*nmodes
                pwidth = [1.0e-8]*nmodes

                newsignal = OrderedDict({
                    "stype":'fouriermode',
                    "corr":"single",
                    "pulsarind":ii,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart
                    })
                signals.append(newsignal)

                if incDM:
                    nmodes = 2*numDMFreqs[ii]
                    bvary = [True]*nmodes
                    pmin = [-1.0]*nmodes
                    pmax = [1.0]*nmodes
                    pstart = [0.0]*nmodes
                    pwidth = [1.0e-5]*nmodes

                    newsignal = OrderedDict({
                        "stype":'dmfouriermode',
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":bvary,
                        "pmin":pmin,
                        "pmax":pmax,
                        "pwidth":pwidth,
                        "pstart":pstart
                        })
                    signals.append(newsignal)

                if incJitter or incCEquad:
                    (avetoas, Umat) = dailyaveragequantities(m2psr.toas)
                    nmodes = len(avetoas)
                    bvary = [True]*nmodes
                    pmin = [-1.0e3]*nmodes
                    pmax = [1.0e3]*nmodes
                    pstart = [0.0]*nmodes
                    pwidth = [1.0e-8]*nmodes

                    newsignal = OrderedDict({
                        "stype":'jitterfouriermode',
                        "corr":"single",
                        "pulsarind":ii,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":bvary,
                        "pmin":pmin,
                        "pmax":pmax,
                        "pwidth":pwidth,
                        "pstart":pstart
                        })
                    signals.append(newsignal)

                    del avetoas
                    del Umat

        if incGWB:
            if gwbModel=='spectrum':
                nfreqs = np.max(numNoiseFreqs)
                bvary = [True]*nfreqs
                pmin = [-18.0]*nfreqs
                pmax = [-7.0]*nfreqs
                pstart = [-10.0]*nfreqs
                pwidth = [0.1]*nfreqs
            elif gwbModel=='powerlaw':
                bvary = [True, True, False]
                pmin = [-17.0, 1.02, 1.0e-11]
                pmax = [-10.0, 6.98, 3.0e-9]
                pstart = [-15.0, 2.01, 1.0e-10]
                pwidth = [0.1, 0.1, 5.0e-11]
            else:
                raise ValueError("ERROR: option {0} not known".
                        format(gwbModel))

            newsignal = OrderedDict({
                "stype":gwbModel,
                "corr":"gr",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart
                })
            signals.append(newsignal)

        if incClock:
            if clockModel=='spectrum':
                nfreqs = np.max(numNoiseFreqs)
                bvary = [True]*nfreqs
                pmin = [-18.0]*nfreqs
                pmax = [-7.0]*nfreqs
                pstart = [-10.0]*nfreqs
                pwidth = [0.1]*nfreqs
            elif clockModel=='powerlaw':
                bvary = [True, True, False]
                pmin = [-17.0, 1.02, 1.0e-11]
                pmax = [-10.0, 6.98, 3.0e-9]
                pstart = [-15.0, 2.01, 1.0e-10]
                pwidth = [0.1, 0.1, 5.0e-11]
            else:
                raise ValueError("ERROR: option {0} not known".
                        format(clockModel))

            newsignal = OrderedDict({
                "stype":clockModel,
                "corr":"uniform",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart
                })
            signals.append(newsignal)

        if incDipole:
            if dipoleModel=='spectrum':
                nfreqs = np.max(numNoiseFreqs)
                bvary = [True]*nfreqs
                pmin = [-18.0]*nfreqs
                pmax = [-7.0]*nfreqs
                pstart = [-10.0]*nfreqs
                pwidth = [0.1]*nfreqs
            elif dipoleModel=='powerlaw':
                bvary = [True, True, False]
                pmin = [-17.0, 1.02, 1.0e-11]
                pmax = [-10.0, 6.98, 3.0e-9]
                pstart = [-15.0, 2.01, 1.0e-10]
                pwidth = [0.1, 0.1, 5.0e-11]
            else:
                raise ValueError("ERROR: option {0} not known".
                        format(dipoleModel))

            newsignal = OrderedDict({
                "stype":dipoleModel,
                "corr":"dipole",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart
                })
            signals.append(newsignal)

        if incAniGWB:
            nclm = (lAniGWB+1)**2-1
            clmvary = [True]*nclm
            clmmin = [-5.0]*nclm
            clmmax = [5.0]*nclm
            clmstart = [0.0]*nclm
            clmwidth = [0.2]*nclm
            if anigwbModel=='spectrum':
                nfreqs = np.max(numNoiseFreqs)
                bvary = [True]*nfreqs + clmvary
                pmin = [-18.0]*nfreqs + clmmin
                pmax = [-7.0]*nfreqs + clmmax
                pstart = [-10.0]*nfreqs + clmstart
                pwidth = [0.1]*nfreqs + clmwidth
            elif anigwbModel=='powerlaw':
                bvary = [True, True, False] + clmvary
                pmin = [-17.0, 1.02, 1.0e-11] + clmmin
                pmax = [-10.0, 6.98, 3.0e-9] + clmmax
                pstart = [-15.0, 2.01, 1.0e-10] + clmstart
                pwidth = [0.1, 0.1, 5.0e-11] + clmwidth
            else:
                raise ValueError("ERROR: option {0} not known".
                        format(anigwbModel))

            newsignal = OrderedDict({
                "stype":anigwbModel,
                "corr":"anisotropicgwb",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart,
                "lAniGWB":lAniGWB
                })
            signals.append(newsignal)

        if incPixelGWB:
            if pixelgwbModel=='spectrum':
                nfreqs = np.max(numNoiseFreqs)
                bvary = [True]*(nfreqs + 2*npixels)
                pmin = [-18.0]*nfreqs + [0, -0.5*np.pi] * npixels
                pmax = [-7.0]*nfreqs + [2*np.pi, 0.5*np.pi] * npixels
                pstart = [-10.0]*nfreqs + [0.0, 0.0] * npixels
                pwidth = [0.1]*nfreqs + [0.1, 0.1] * npixels
            elif anigwbModel=='powerlaw':
                bvary = [True, True, False] + [True, True] * npixels
                pmin = [-17.0, 1.02, 1.0e-11] + [0.0, -0.5*np.pi] * npixels
                pmax = [-10.0, 6.98, 3.0e-9] + [2*np.pi, 0.5*np.pi] * npixels
                pstart = [-15.0, 2.01, 1.0e-10] + [0.0, 0.0] * npixels
                pwidth = [0.1, 0.1, 5.0e-11] + [0.1, 0.1] * npixels
            else:
                raise ValueError("ERROR: option {0} not known".
                        format(anigwbModel))

            newsignal = OrderedDict({
                "stype":anigwbModel,
                "corr":"pixelgwb",
                "pulsarind":-1,
                "bvary":bvary,
                "pmin":pmin,
                "pmax":pmax,
                "pwidth":pwidth,
                "pstart":pstart,
                "npixels":npixels
                })
            signals.append(newsignal)

        if incBWM:
            toamax = self.ptapsrs[0].toas[0]
            toamin = self.ptapsrs[0].toas[0]
            for psr in self.ptapsrs:
                if toamax < np.max(psr.toas):
                    toamax = np.max(psr.toas)
                if toamin > np.min(psr.toas):
                    toamin = np.min(psr.toas)
            newsignal = OrderedDict({
                "stype":'bwm',
                "corr":"gr",
                "pulsarind":-1,
                "bvary":[True, True, True, True, True],
                "pmin":[toamin, -18.0, 0.0, 0.0, 0.0],
                "pmax":[toamax, -10.0, 2*np.pi, np.pi, np.pi],
                "pwidth":[30*24*3600.0, 0.1, 0.1, 0.1, 0.1],
                "pstart":[0.5*(toamax-toamin), -15.0, 3.0, 1.0, 1.0]
                })
            signals.append(newsignal)

        # The list of signals
        modeldict = OrderedDict({
            "file version":2014.01,
            "author":"piccard-makeModel",
            "numpulsars":len(self.ptapsrs),
            "pulsarnames":[self.ptapsrs[ii].name for ii in range(len(self.ptapsrs))],
            "numNoiseFreqs":list(numNoiseFreqs),
            "numDMFreqs":list(numDMFreqs),
            "compression":compression,
            "orderFrequencyLines":orderFrequencyLines,
            "evalCompressionComplement":evalCompressionComplement,
            "likfunc":likfunc,
            "Tmax":Tmax,
            "signals":signals
            })

        return modeldict


    """
    Create the model dictionary, based on the currently used model
    """
    def getModelDict(self):
        # We have to determine the number of frequencies we'll need
        npsrs = len(self.ptapsrs)
        numNoiseFreqs = [int(len(self.ptapsrs[ii].Ffreqs)/2) for ii in range(npsrs)]
        numDMFreqs = [int(len(self.ptapsrs[ii].Fdmfreqs)/2) for ii in range(npsrs)]

        signals = []

        for ii, m2signal in enumerate(self.ptasignals):
            signals.append(m2signal.copy())

            # Delete a few redundant quantities
            if 'Nvec' in signals[-1]:
                del signals[-1]['Nvec']
            if 'Jvec' in signals[-1]:
                del signals[-1]['Jvec']
            if 'corrmat' in signals[-1]:
                del signals[-1]['corrmat']
            if 'aniCorr' in signals[-1]:
                del signals[-1]['aniCorr']
            if 'Ffreqs' in signals[-1]:
                del signals[-1]['Ffreqs']

            # Numpy arrays are not "JSON serializable"
            signals[-1]['bvary'] = map(bool, signals[-1]['bvary'])
            signals[-1]['pmin'] = map(float, signals[-1]['pmin'])
            signals[-1]['pmax'] = map(float, signals[-1]['pmax'])
            signals[-1]['pstart'] = map(float, signals[-1]['pstart'])
            signals[-1]['pwidth'] = map(float, signals[-1]['pwidth'])
            #if 'unitconversion' in signals[-1]:
            #    signals[-1]['unitconversion'] = map(float, signals[-1]['unitconversion'])

        modeldict = OrderedDict({
            "file version":2014.03,
            "author":"piccard-makeModel",
            "numpulsars":len(self.ptapsrs),
            "pulsarnames":[self.ptapsrs[ii].name for ii in range(len(self.ptapsrs))],
            "numNoiseFreqs":list(numNoiseFreqs),
            "numDMFreqs":list(numDMFreqs),
            "compression":self.compression,
            "orderFrequencyLines":self.orderFrequencyLines,
            "evalCompressionComplement":self.evallikcomp,
            "likfunc":self.likfunc,
            "Tmax":self.Tmax,
            "signals":signals
            })

        return modeldict


    """
    Initialise a model from a json file

    @param filename:    Filename of the json file with the model
    """
    def initModelFromFile(self, filename, auxFromFile=True, noGmat=False,
            verbose=False, noCreate=False):
        with open(filename) as data_file:
            model = OrderedDict(json.load(data_file))
        self.initModel(model, fromFile=auxFromFile, noGmatWrite=noGmat,
                verbose=verbose, noCreate=noCreate)

    """
    Write the model to a json file

    @param filename:    Filename of the json file with the model
    """
    def writeModelToFile(self, filename):
        model = self.getModelDict()

        with open(filename, 'w') as outfile:
            json.dump(model, outfile, sort_keys=False, indent=4, separators=(',', ': '))

    """
    Based on single-pulsar model dictionaries, construct a full PTA model
    dictionary. This function will only check the number of pulsars in the json
    files, no additional checks are done. Any GW/correlated signal are bluntly
    combined, so make sure no duplicates exist.

    The likelihood and other settings are copied from the first model in the
    list

    @param modeldictlist:   List of model dictionaries that will be combined
    """
    def combineModelDicts(self, *modeldictlist):
        # Check the consistency of the dictionaries with the current model
        ndictpsrs = 0
        for model in modeldictlist:
            ndictpsrs += model['numpulsars']

        # If the combined number of pulsars in the dictionaries does not match
        # the internal number of pulsars: generate an error
        if ndictpsrs != len(self.ptapsrs):
            raise IOError, "Number of pulsars does not match sum of dictionary models"

        # Check that all pulsars are present
        haveDictPulsar = np.array([0]*ndictpsrs, dtype=np.bool)
        psrnames = [self.ptapsrs[ii].name for ii in range(len(self.ptapsrs))]
        for model in modeldictlist:
            for dictpsrname in model['pulsarnames']:
                if not dictpsrname in psrnames:
                    raise IOError, "Pulsar {0} not present in internal pulsar list {1}".format(dictpsrname, psrnames)
                haveDictPulsar[psrnames.index(dictpsrname)] = True

        # Check that _all_ pulsars are present
        if not np.sum(haveDictPulsar) == len(self.ptapsrs):
            raise IOError, "Pulsars not present in provided models: {0}".format(\
                    psrnames[haveDictPulsar == False])

        # Ok, it seems we are good. Let's combine the models
        nmd = modeldictlist[0].copy()
        nmd['author'] = 'piccard-combineModelDicts'
        nmd['numpulsars'] = ndictpsrs
        nmd['pulsarnames'] = psrnames
        nmd['numNoiseFreqs'] = [0]*len(psrnames)
        nmd['numDMFreqs'] = [0]*len(psrnames)
        nmd['signals'] = []
        for ii, md in enumerate(modeldictlist):
            mdpulsars = md['pulsarnames']
            mdsignals = md['signals']
            mdpsrind = [0] * len(mdpulsars)

            # Find the indices of these pulsars in the internal psr list
            for pp, mdp in enumerate(mdpulsars):
                pind = psrnames.index(mdp)
                mdpsrind[pp] = pind

            # Now that we have the index translation, set the frequency data
            for pp, mdp in enumerate(mdpulsars):
                nmd['numNoiseFreqs'][mdpsrind[pp]] = md['numNoiseFreqs'][pp]
                nmd['numDMFreqs'][mdpsrind[pp]] = md['numDMFreqs'][pp]

            # Add all the signals
            for ss, mds in enumerate(mdsignals):
                # Copy the signal
                newsignal = mds.copy()

                # Translate the pulsar index
                if newsignal['pulsarind'] != -1:
                    newsignal['pulsarind'] = mdpsrind[newsignal['pulsarind']]

                nmd['signals'].append(newsignal)

        return nmd

    def setDictStartPars(self, startpars):
        """
        Given an array of varying start-parameters, set the start-parameters in
        the model dictionary
        """
        for ss, signal in enumerate(self.ptasignals):
            parindex = signal['parindex']
            pp = 0
            for ii in range(signal['ntotpars']):
                if signal['bvary'][ii]:
                    signal['pstart'][ii] = startpars[parindex+pp]
                    pp = pp + 1


    """
    Initialise the model.
    @param numNoiseFreqs:       Dictionary with the full model
    @param fromFile:            Try to read the necessary Auxiliaries quantities
                                from the HDF5 file
    @param verbose:             Give some extra information about progress
    """
    def initModel(self, fullmodel, fromFile=True, verbose=False, \
                  noGmatWrite=False, noCreate=False, \
                  addDMQSD=False):
        numNoiseFreqs = fullmodel['numNoiseFreqs']
        numDMFreqs = fullmodel['numDMFreqs']
        compression = fullmodel['compression']
        evalCompressionComplement = fullmodel['evalCompressionComplement']
        orderFrequencyLines = fullmodel['orderFrequencyLines']
        likfunc = fullmodel['likfunc']
        signals = fullmodel['signals']

        if 'Tmax' in fullmodel:
            if fullmodel['Tmax'] is not None:
                self.Tmax = fullmodel['Tmax']
            else:
                self.Tmax = None
        else:
            self.Tmax = None

        if len(self.ptapsrs) < 1:
            raise IOError, "No pulsars loaded"

        if fullmodel['numpulsars'] != len(self.ptapsrs):
            raise IOError, "Model does not have the right number of pulsars"

        if not self.checkSignalDictionary(signals):
            raise IOError, "Signal dictionary not properly defined"

        # Details about the likelihood function
        self.likfunc = likfunc
        self.orderFrequencyLines = orderFrequencyLines

        # Determine the time baseline of the array of pulsars
        Tstart = np.min(self.ptapsrs[0].toas)
        Tfinish = np.max(self.ptapsrs[0].toas)
        for m2psr in self.ptapsrs:
            Tstart = np.min([np.min(m2psr.toas), Tstart])
            Tfinish = np.max([np.max(m2psr.toas), Tfinish])

        if self.Tmax is None:
            self.Tmax = Tfinish - Tstart

        # If the compressionComplement is defined, overwrite the default
        if evalCompressionComplement != 'None':
            self.evallikcomp = evalCompressionComplement
            self.compression = compression
        elif compression == 'None':
            self.evallikcomp = False
        else:
            self.evallikcomp = True
            self.compression = compression

        # Find out how many single-frequency modes there are
        numSingleFreqs = self.getNumberOfSignalsFromDict(signals, \
                stype='frequencyline', corr='single')
        numSingleDMFreqs = self.getNumberOfSignalsFromDict(signals, \
                stype='dmfrequencyline', corr='single')

        # Find out how many efac signals there are, and translate that to a
        # separateEfacs boolean array (for two-component noise analysis)
        numEfacs = self.getNumberOfSignalsFromDict(signals, \
                stype='efac', corr='single')
        numJits = self.getNumberOfSignalsFromDict(signals, \
                stype='jitter', corr='single')
        numEquads = self.getNumberOfSignalsFromDict(signals, \
                stype='equad', corr='single')
        separateEfacs = (numEfacs + numEquads + numJits) > 1

        # Modify design matrices, and create pulsar Auxiliary quantities
        for pindex, m2psr in enumerate(self.ptapsrs):
            # If we model DM variations, we will need to include QSD
            # marginalisation for DM. Modify design matrix accordingly
            #if dmModel[pindex] != 'None':
            if numDMFreqs[pindex] > 0 and addDMQSD:
                m2psr.addDMQuadratic()
                print "WARNING: DMQSD not saved to HDF5 file (auxiliaries are...)"
                print "         DM1 and DM2 better added to par-file"

            tmsigpars = None
            if compression == 'timingmodel':
                # If we compress on the timing model, obtain the timing model
                # parameters from the relevant signal
                linsigind = self.getSignalNumbersFromDict(signals,
                        stype='lineartimingmodel', psrind=pindex)
                nlsigind = self.getSignalNumbersFromDict(signals,
                        stype='nonlineartimingmodel', psrind=pindex)
                if len(linsigind) + len(nlsigind) < 1:
                    raise ValueError("Model for pulsar {0} should contain at least one timing model signal for compression 'timingmodel'".format(m2psr.name))

                tmsigpars = []    # All the timing model parameters of this pulsar
                for ss in np.append(linsigind, nlsigind):
                    tmsigpars += signals[ss]['parid']

            # We'll try to read the necessary quantities from the HDF5 file
            try:
                if not fromFile:
                    raise StandardError('Requested to re-create the Auxiliaries')
                # Read Auxiliaries
                if verbose:
                    print "Reading Auxiliaries for {0}".format(m2psr.name)
                m2psr.readPulsarAuxiliaries(self.h5df, self.Tmax, \
                        numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], not separateEfacs[pindex], \
                        nSingleFreqs=numSingleFreqs[pindex], \
                        nSingleDMFreqs=numSingleDMFreqs[pindex], \
                        likfunc=likfunc, compression=compression, \
                        evalCompressionComplement=evalCompressionComplement, \
                        memsave=True, noGmat=noGmatWrite)
            except (StandardError, ValueError, IOError, RuntimeError) as err:
                # Create the Auxiliaries ourselves

                # For every pulsar, construct the auxiliary quantities like the Fourier
                # design matrix etc
                if verbose:
                    print str(err)
                    print "Creating Auxiliaries for {0}".format(m2psr.name)

                # If we are not allowed to overwrite file stuff, raise an
                # exception
                if noCreate:
                    raise RuntimeError("Not allowed to create auxiliaries")

                m2psr.createPulsarAuxiliaries(self.h5df, self.Tmax, numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], not separateEfacs[pindex], \
                                nSingleFreqs=numSingleFreqs[pindex], \
                                nSingleDMFreqs=numSingleDMFreqs[pindex], \
                                likfunc=likfunc, compression=compression, \
                                write='likfunc', tmsigpars=tmsigpars, \
                                noGmatWrite=noGmatWrite)

            # When selecting Fourier modes, like in mark7/mark8, the binclude vector
            # indicates whether or not a frequency is included in the likelihood. By
            # default they are all 'on'
            if self.likfunc == 'mark7' or self.likfunc == 'mark8':
                m2psr.setLimitedModeAuxiliaries([1]*numNoiseFreqs[pindex], \
                        [1]*numDMFreqs[pindex], likfunc=self.likfunc)

        # Initialise the ptasignal objects
        self.ptasignals = []
        index = 0
        for ii, signal in enumerate(signals):
            self.addSignal(signal, index, self.Tmax)
            index += self.ptasignals[-1]['npars']

        self.allocateLikAuxiliaries()
        self.initPrior()
        self.pardes = self.getModelParameterList()


    """
    Get a list of all the model parameters, the parameter indices, and the
    descriptions

    TODO: insert these descriptions in the signal dictionaries
    """
    def getModelParameterList(self):
        pardes = []

        for ii, sig in enumerate(self.ptasignals):
            pindex = 0
            for jj in range(sig['ntotpars']):
                if sig['bvary'][jj]:
                    # This parameter is in the mcmc
                    index = sig['parindex'] + pindex
                    pindex += 1
                else:
                    index = -1

                psrindex = sig['pulsarind']
                if sig['stype'] == 'efac':
                    flagname = sig['flagname']
                    flagvalue = 'efac'+sig['flagvalue']
                elif sig['stype'] == 'equad':
                    flagname = sig['flagname']
                    flagvalue = 'equad'+sig['flagvalue']
                elif sig['stype'] == 'jitter':
                    flagname = sig['flagname']
                    flagvalue = 'jitter'+sig['flagvalue']
                elif sig['stype'] == 'spectrum':
                    flagname = 'frequency'

                    # If there are more parameters than frequencies, this is an
                    # anisotropic background
                    if jj >= len(self.ptapsrs[psrindex].Ffreqs)/2 and sig['corr'] == 'pixelgwb':
                        ind = (jj - len(self.ptapsrs[psrindex].Ffreqs)/2) % 2
                        flagvalue = ['GWB-RA', 'GWB-DEC'][ind]
                    elif jj >= len(self.ptapsrs[psrindex].Ffreqs)/2:
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
                elif sig['stype'] == 'dmspectrum':
                    flagname = 'dmfrequency'
                    flagvalue = str(self.ptapsrs[psrindex].Fdmfreqs[2*jj])
                elif sig['stype'] == 'powerlaw':
                    flagname = 'powerlaw'

                    if jj < 3 or sig['corr'] == 'pixelgwb':
                        if sig['corr'] == 'gr':
                            flagvalue = ['GWB-Amplitude', 'GWB-spectral-index', 'low-frequency-cutoff'][jj]
                        elif sig['corr'] == 'uniform':
                            flagvalue = ['CLK-Amplitude', 'CLK-spectral-index', 'low-frequency-cutoff'][jj]
                        elif sig['corr'] == 'dipole':
                            flagvalue = ['DIP-Amplitude', 'DIP-spectral-index', 'low-frequency-cutoff'][jj]
                        elif sig['corr'] == 'anisotropicgwb':
                            flagvalue = ['GWB-Amplitude', 'GWB-spectral-index', 'low-frequency-cutoff'][jj]
                        elif sig['corr'] == 'pixelgwb':
                            if jj > 4:
                                ind = 3 + ((jj-3) % 2)
                            else:
                                ind = jj
                            flagvalue = ['GWB-Amplitude', 'GWB-spectral-index', 'low-frequency-cutoff', \
                                         'GWB-RA', 'GWB-DEC'][ind]
                        else:
                            flagvalue = ['RN-Amplitude', 'RN-spectral-index', 'low-frequency-cutoff'][jj]
                    else:
                        # Index counting same as above
                        clmind = jj - 3 + 1
                        lani = int(np.sqrt(clmind))
                        mani = clmind - lani*(lani+1)
                        flagvalue = 'C_(' + str(lani) + ',' + str(mani) + ')'
                elif sig['stype'] == 'dmpowerlaw':
                    flagname = 'dmpowerlaw'
                    flagvalue = ['DM-Amplitude', 'DM-spectral-index', 'low-frequency-cutoff'][jj]
                elif sig['stype'] == 'spectralModel':
                    flagname = 'spectralModel'
                    flagvalue = ['SM-Amplitude', 'SM-spectral-index', 'SM-corner-frequency'][jj]
                elif sig['stype'] == 'frequencyline':
                    flagname = 'frequencyline'
                    flagvalue = ['Line-Freq', 'Line-Ampl'][jj]
                elif sig['stype'] == 'bwm':
                    flagname = 'BurstWithMemory'
                    flagvalue = ['burst-arrival', 'amplitude', 'raj', 'decj', 'polarisation'][jj]
                elif sig['stype'] == 'lineartimingmodel' or \
                        sig['stype'] == 'nonlineartimingmodel':
                    flagname = sig['stype']
                    flagvalue = sig['parid'][jj]
                elif sig['stype'] == 'fouriermode':
                    if jj % 2 == 0:
                        flagname = 'Fourier-cos'
                    else:
                        flagname = 'Fourier-sin'
                    flagvalue = str(self.ptapsrs[psrindex].Ffreqs[jj])
                elif sig['stype'] == 'dmfouriermode':
                    if jj % 2 == 0:
                        flagname = 'DM-Fourier-cos'
                    else:
                        flagname = 'DM-Fourier-sin'
                    flagvalue = str(self.ptapsrs[psrindex].Fdmfreqs[jj])
                elif sig['stype'] == 'jitterfouriermode':
                    flagname = 'Jitter-mode'
                    flagvalue = str(pic_T0 + self.ptapsrs[psrindex].avetoas[jj] / pic_spy)
                else:
                    flagname = 'none'
                    flagvalue = 'none'

                pulsarname = self.ptapsrs[psrindex].name

                pardes.append(\
                        {'index': index, 'pulsar': psrindex, 'sigindex': ii, \
                            'sigtype': sig['stype'], 'correlation': sig['corr'], \
                            'name': flagname, 'id': flagvalue, 'pulsarname': \
                            pulsarname})

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
            fil.write("{0:d} \t{1:d} \t{2:s} \t{3:s} \t{4:s} \t{5:s} \t{6:s}\n".format(\
                    self.pardes[ii]['index'],
                    self.pardes[ii]['pulsar'],
                    self.pardes[ii]['sigtype'],
                    self.pardes[ii]['correlation'],
                    self.pardes[ii]['name'],
                    self.pardes[ii]['id'],
                    self.pardes[ii]['pulsarname']))

        fil.close()


    def saveResiduals(self, outputdir):
        """
        Save the residuals of all pulsars to text files

        @param outputdir:   Directory where the txt files will be saved
        """

        for pp, psr in enumerate(self.ptapsrs):
            filename = outputdir + '/residuals-' + psr.name + '.txt'

            fil = open(filename, "w")
            for ii in range(len(psr.toas)):
                fil.write("{0} \t{1} \t{2} \t{3}\n".format(\
                        pic_T0 + psr.toas[ii]/pic_spd, \
                        psr.residuals[ii], \
                        psr.toaerrs[ii], \
                        psr.flags[ii]))
        fil.close()

    """
    Re-calculate the number of varying parameters per signal, and the number of
    dimensions in total.
    """
    def setDimensions(self):
        self.dimensions = 0
        for m2signal in self.ptasignals:
            m2signal['npars'] = np.sum(m2signal['bvary'])
            self.dimensions += m2signal['npars']


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
            for ii in range(m2signal['ntotpars']):
                if m2signal['bvary'][ii]:
                    self.pmin[index] = m2signal['pmin'][ii]
                    self.pmax[index] = m2signal['pmax'][ii]
                    self.pwidth[index] = m2signal['pwidth'][ii]
                    self.pstart[index] = m2signal['pstart'][ii]
                    index += 1

    """
    Return a list of all efac parameter numbers, their names, and the pulsar
    they belong to
    """
    def getEfacNumbers(self, stype='efac'):
        parind = []
        psrind = []
        names = []

        for ii, m2signal in enumerate(self.ptasignals):
            if m2signal['stype'] == stype and m2signal['bvary'][0]:
                parind.append(m2signal['parindex'])
                psrind.append(m2signal['pulsarind'])
                names.append(m2signal['flagvalue'])

        return (parind, psrind, names)

    """
    Return a list of all spectrum signals: signal name, start-par, stop-par, and
    the actual frequencies

    TODO: parameters can be non-varying. Take that into account as well
    """
    def getSpectraNumbers(self):
        signame = []
        signameshort = []
        parmin = []
        parmax = []
        freqs = []
        for ii, m2signal in enumerate(self.ptasignals):
            if m2signal['stype'] == 'spectrum' or m2signal['stype'] == 'dmspectrum':
                if m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'single':
                    signame.append('Red noise ' + self.ptapsrs[m2signal['pulsarind']].name)
                    signameshort.append('rnspectrum-' + self.ptapsrs[m2signal['pulsarind']].name)
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'gr':
                    signame.append('GWB spectrum')
                    signameshort.append('gwbspectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'uniform':
                    signame.append('Clock spectrum')
                    signameshort.append('clockspectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'dipole':
                    signame.append('Dipole spectrum')
                    signameshort.append('dipolespectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'anisotropicgwb':
                    signame.append('Anisotropy spectrum')
                    signameshort.append('anisotropyspectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))
                elif m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'pixelgwb':
                    signame.append('Pixel-based gwb spectrum')
                    signameshort.append('pixelgwbspectrum')
                elif m2signal['stype'] == 'dmspectrum':
                    signame.append('DM variation ' + self.ptapsrs[m2signal['pulsarind']].name)
                    signameshort.append('dmspectrum-' + self.ptapsrs[m2signal['pulsarind']].name)
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[m2signal['pulsarind']].Fdmfreqs)))))
                else:
                    signame.append('Spectrum')
                    signameshort.append('spectrum')
                    freqs.append(np.sort(np.array(list(set(self.ptapsrs[0].Ffreqs)))))

                parmin.append(m2signal['parindex'])
                parmax.append(m2signal['parindex']+m2signal['npars'])

        return (signame, signameshort, parmin, parmax, freqs)


    """
    Loop over all signals, and fill the diagonal pulsar noise covariance matrix
    (based on efac/equad)
    For two-component noise model, fill the total weights vector

    @param parameters:  The total parameters vector
    @param selection:   Boolean array, indicating which parameters to include
    @param incJitter:   Whether or not to include Jitter in the noise vectort

    """
    def setPsrNoise(self, parameters, selection=None):
        # For every pulsar, set the noise vector to zero
        for m2psr in self.ptapsrs:
            if m2psr.twoComponentNoise:
                m2psr.Nwvec[:] = 0
                m2psr.Nwovec[:] = 0
            #else:
            m2psr.Nvec[:] = 0
            m2psr.Jvec[:] = 0

        if selection is None:
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # Loop over all white noise signals, and fill the pulsar Nvec
        for ss, m2signal in enumerate(self.ptasignals):
            #m2signal = self.ptasignals[ss]
            psr = self.ptapsrs[m2signal['pulsarind']]
            if selection[ss]:
                if m2signal['stype'] == 'efac':
                    if m2signal['npars'] == 1:
                        pefac = parameters[m2signal['parindex']]
                    else:
                        pefac = m2signal['pstart'][0]

                    if psr.twoComponentNoise:
                        psr.Nwvec += psr.Wvec * pefac**2

                        if len(psr.Wovec) > 0:
                            psr.Nwovec += psr.Wovec * pefac**2

                    psr.Nvec += m2signal['Nvec'] * pefac**2

                elif m2signal['stype'] == 'equad':
                    if m2signal['npars'] == 1:
                        pequadsqr = 10**(2*parameters[m2signal['parindex']])
                    else:
                        pequadsqr = 10**(2*m2signal['pstart'][0])

                    if psr.twoComponentNoise:
                        psr.Nwvec += pequadsqr
                        psr.Nwovec += pequadsqr

                    psr.Nvec += m2signal['Nvec'] * pequadsqr
                elif m2signal['stype'] == 'jitter':
                    if m2signal['npars'] == 1:
                        pequadsqr = 10**(2*parameters[m2signal['parindex']])
                    else:
                        pequadsqr = 10**(2*m2signal['pstart'][0])

                    psr.Jvec += m2signal['Jvec'] * pequadsqr


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
    def constructPhiAndTheta(self, parameters, selection=None, phimat=True):
        self.Phi[:] = 0         # Start with a fresh matrix
        self.Phivec[:] = 0      # ''
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
                sparameters = m2signal['pstart'].copy()
                sparameters[m2signal['bvary']] = \
                        parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']]
                if m2signal['stype'] == 'spectrum':
                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npff[:m2signal['pulsarind']])
                        # nfreq = int(self.npf[m2signal['pulsarind']]/2)
                        nfreq = m2signal['npars']

                        # Pcdoubled is an array where every element of the parameters
                        # of this m2signal is repeated once (e.g. [1, 1, 3, 3, 2, 2, 5, 5, ...]

                        pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                        # Fill the phi matrix
                        di = np.diag_indices(2*nfreq)
                        if phimat:
                            self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += 10**pcdoubled
                        else:
                            self.Phivec[findex:findex+2*nfreq] = 10**pcdoubled
                    elif m2signal['corr'] in ['gr', 'uniform', 'dipole', \
                                              'anisotropicgwb', 'pixelgwb']:
                        nfreq = m2signal['npars']

                        if m2signal['corr'] in ['gr', 'uniform', 'dipole']:
                            pcdoubled = np.array([sparameters, sparameters]).T.flatten()
                            corrmat = m2signal['corrmat']
                        elif m2signal['corr'] == 'anisotropicgwb':
                            nclm = m2signal['aniCorr'].clmlength()
                            # These indices do not seem right at all!
                            pcdoubled = np.array([\
                                    sparameters[:-nclm],\
                                    sparameters[:-nclm]]).T.flatten()
                            clm = sparameters[-nclm:]
                            corrmat = m2signal['aniCorr'].corrmat(clm)
                        elif m2signal['corr'] == 'pixelgwb':
                            npixels = m2signal['aniCorr'].npixels
                            pixpars = sparameters[-2*npixels:]
                            corrmat = m2signal['aniCorr'].corrmat(pixpars)

                        indexa = 0
                        indexb = 0
                        for aa in range(npsrs):
                            for bb in range(npsrs):
                                # Some pulsars may have fewer frequencies than
                                # others (right?). So only use overlapping ones
                                nof = np.min([self.npf[aa], self.npf[bb], 2*nfreq])
                                di = np.diag_indices(nof)

                                if phimat:
                                    self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += 10**pcdoubled[:nof] * corrmat[aa, bb]
                                indexb += self.npff[bb]
                            indexb = 0
                            indexa += self.npff[aa]
                elif m2signal['stype'] == 'dmspectrum':
                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npffdm[:m2signal['pulsarind']])
                        nfreq = int(self.npfdm[m2signal['pulsarind']]/2)

                        pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                        # Fill the Theta matrix
                        self.Thetavec[findex:findex+2*nfreq] += 10**pcdoubled
                elif m2signal['stype'] == 'powerlaw':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npff[:m2signal['pulsarind']])
                        nfreq = int(self.npf[m2signal['pulsarind']]/2)
                        freqpy = self.ptapsrs[m2signal['pulsarind']].Ffreqs * pic_spy
                        pcdoubled = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * m2signal['Tmax'])) * freqpy ** (-Si)

                        # Fill the phi matrix
                        di = np.diag_indices(2*nfreq)

                        if phimat:
                            self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                        else:
                            self.Phivec[findex:findex+2*nfreq] = pcdoubled
                    elif m2signal['corr'] in ['gr', 'uniform', 'dipole', \
                                              'anisotropicgwb', 'pixelgwb']:
                        freqpy = m2signal['Ffreqs'] * pic_spy
                        pcdoubled = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * m2signal['Tmax'])) * freqpy ** (-Si)
                        nfreq = len(freqpy)

                        if m2signal['corr'] in ['gr', 'uniform', 'dipole']:
                            corrmat = m2signal['corrmat']
                        elif m2signal['corr'] == 'anisotropicgwb':
                            nclm = m2signal['aniCorr'].clmlength()
                            clm = sparameters[-nclm:]
                            corrmat = m2signal['aniCorr'].corrmat(clm)
                        elif m2signal['corr'] == 'pixelgwb':
                            pixpars = sparameters[3:]
                            corrmat = m2signal['aniCorr'].corrmat(pixpars)

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

                                if phimat:
                                    self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * corrmat[aa, bb]
                                indexb += self.npff[bb]
                            indexb = 0
                            indexa += self.npff[aa]
                elif m2signal['stype'] == 'spectralModel':
                    Amp = 10**sparameters[0]
                    alpha = sparameters[1]
                    fc = 10**sparameters[2] / pic_spy

                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npff[:m2signal['pulsarind']])
                        nfreq = int(self.npf[m2signal['pulsarind']]/2)
                        freqpy = self.ptapsrs[m2signal['pulsarind']].Ffreqs
                        pcdoubled = (Amp * pic_spy**3 / m2signal['Tmax']) * ((1 + (freqpy/fc)**2)**(-0.5*alpha))

                        #pcdoubled = (Amp * pic_spy**3 / (m2signal['Tmax'])) * freqpy ** (-Si)

                        # Fill the phi matrix
                        di = np.diag_indices(2*nfreq)

                        if phimat:
                            self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                        else:
                            self.Phivec[findex:findex+2*nfreq] = pcdoubled
                    elif m2signal['corr'] in ['gr', 'uniform', 'dipole', \
                                              'anisotropicgwb', 'pixelgwb']:
                        freqpy = self.ptapsrs[0].Ffreqs * pic_spy
                        pcdoubled = (Amp * pic_spy**3 / m2signal['Tmax']) / \
                                ((1 + (freqpy/fc)**2)**(-0.5*alpha))
                        nfreq = len(freqpy)

                        if m2signal['corr'] in ['gr', 'uniform', 'dipole']:
                            corrmat = m2signal['corrmat']
                        elif m2signal['corr'] == 'anisotropicgwb':
                            nclm = m2signal['aniCorr'].clmlength()
                            # These indices don't seem right. Check the -nclm part
                            clm = sparameters[-nclm:m2signal['parindex']+m2signal['npars']]
                            corrmat = m2signal['aniCorr'].corrmat(clm)
                        elif m2signal['corr'] == 'pixelgwb':
                            pixpars = sparameters[3:]
                            corrmat = m2signal['aniCorr'].corrmat(pixpars)

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

                                if phimat:
                                    self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * corrmat[aa, bb]
                                indexb += self.npff[bb]
                            indexb = 0
                            indexa += self.npff[aa]
                elif m2signal['stype'] == 'dmpowerlaw':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npffdm[:m2signal['pulsarind']])
                        nfreq = int(self.npfdm[m2signal['pulsarind']]/2)
                        freqpy = self.ptapsrs[m2signal['pulsarind']].Fdmfreqs * pic_spy
                        # TODO: change the units of the DM signal
                        pcdoubled = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * m2signal['Tmax'])) * freqpy ** (-Si)

                        # Fill the Theta matrix
                        self.Thetavec[findex:findex+2*nfreq] += pcdoubled

                elif m2signal['stype'] == 'frequencyline':
                    # For a frequency line, the FFmatrix is assumed to be set elsewhere
                    findex = np.sum(self.npff[:m2signal['pulsarind']]) + \
                            self.npf[m2signal['pulsarind']] + 2*m2signal['npsrfreqindex']

                    pcdoubled = np.array([sparameters[1], sparameters[1]])
                    di = np.diag_indices(2)

                    if phimat:
                        self.Phi[findex:findex+2, findex:findex+2][di] += 10**pcdoubled
                    else:
                        self.Phivec[findex:findex+2] = 10**pcdoubled
                elif m2signal['stype'] == 'dmfrequencyline':
                    # For a DM frequency line, the DFF is assumed to be set elsewhere
                    findex = np.sum(self.npffdm[:m2signal['pulsarind']]) + \
                            self.npfdm[m2signal['pulsarind']] + 2*m2signal['npsrdmfreqindex']

                    pcdoubled = np.array([sparameters[1], sparameters[1]])
                    self.Thetavec[findex:findex+2] += 10**pcdoubled

    """
    Update the deterministic signals for the new values of the parameters. This
    updated signal is used in the likelihood functions
    """
    def updateDetSources(self, parameters, selection=None):
        npsrs = len(self.ptapsrs)

        if selection is None:
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # Set all the detresiduals equal to residuals
        for pp, psr in enumerate(self.ptapsrs):
            psr.detresiduals = psr.residuals.copy()

        # In the case we have numerical timing model (linear/nonlinear)
        for ss, m2signal in enumerate(self.ptasignals):
            if selection[ss]:
                # Create a parameters array for this particular signal
                sparameters = m2signal['pstart'].copy()
                sparameters[m2signal['bvary']] = \
                        parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']]
                if m2signal['stype'] == 'lineartimingmodel':
                    # This one only applies to one pulsar at a time
                    ind = []
                    pp = m2signal['pulsarind']
                    newdes = m2signal['parid']
                    psr = self.ptapsrs[pp]

                    if len(newdes) == psr.Mmat.shape[1]:
                        # This is a speed hack, necessary for mark11
                        Mmat = psr.Mmat
                    else:
                        # Create slicing vector (select parameters actually in signal)
                        for jj, parid in enumerate(psr.ptmdescription):
                            if parid in newdes:
                                ind += [True]
                            else:
                                ind += [False]
                        ind = np.array(ind, dtype=np.bool)
                        Mmat = psr.Mmat[:,ind]

                    # residuals = M * pars
                    psr.detresiduals -= \
                            np.dot(Mmat, \
                            (sparameters-m2signal['pstart'])) # / m2signal['unitconversion'])

                elif m2signal['stype'] == 'nonlineartimingmodel':
                    # The t2psr libstempo object has to be set. Assume it is.
                    pp = m2signal['pulsarind']
                    psr = self.ptapsrs[pp]

                    # For each varying parameter, update the libstempo object
                    # parameter with the new value
                    pindex = 0
                    offset = 0
                    for jj in range(m2signal['ntotpars']):
                        if m2signal['bvary'][jj]:
                            # If this parameter varies, update the parameter
                            if m2signal['parid'][jj] == 'Offset':
                                offset = sparameters[pindex]
                            else:
                                psr.t2psr[m2signal['parid'][jj]].val = \
                                        sparameters[pindex]
                            pindex += 1

                    # Generate the new residuals
                    psr.detresiduals = np.array(psr.t2psr.residuals(updatebats=True), dtype=np.double) + offset

        # Loop over all signals, and construct the deterministic signals
        for ss in range(len(self.ptasignals)):
            m2signal = self.ptasignals[ss]
            if selection[ss]:
                # Create a parameters array for this particular signal
                sparameters = m2signal['pstart'].copy()
                sparameters[m2signal['bvary']] = \
                        parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']]
                if m2signal['stype'] == 'bwm':
                    for pp in range(len(self.ptapsrs)):
                        if m2signal['pulsarind'] == pp or m2signal['pulsarind'] == -1:
                            bwmsig = bwmsignal(sparameters, \
                                    self.ptapsrs[pp].raj, self.ptapsrs[pp].decj, \
                                    self.ptapsrs[pp].toas)

                            self.ptapsrs[pp].detresiduals -= bwmsig
                elif m2signal['stype'] == 'fouriermode':
                    self.ptapsrs[pp].detresiduals -= \
                            np.dot(self.ptapsrs[pp].Fmat, sparameters)
                elif m2signal['stype'] == 'dmfouriermode':
                    self.ptapsrs[pp].detresiduals -= \
                            np.dot(self.ptapsrs[pp].DF, sparameters)
                elif m2signal['stype'] == 'jitterfouriermode':
                    self.ptapsrs[pp].detresiduals -= \
                            np.dot(self.ptapsrs[pp].Umat, sparameters)


        # If necessary, transform these residuals to two-component basis
        for pp, psr in enumerate(self.ptapsrs):
            if psr.twoComponentNoise:
                Gr = np.dot(psr.Hmat.T, psr.detresiduals)
                psr.AGr = np.dot(psr.Amat.T, Gr)



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
            if m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'single':
                # Red noise, see if we need to include it
                findex = int(np.sum(self.npf[:m2signal['pulsarind']])/2)
                nfreq = int(self.npf[m2signal['pulsarind']]/2)

                # Select the common frequency modes
                inc = np.logical_and(bfind[findex:findex+nfreq], bcurfind[findex:findex+nfreq])
                # Select the _new_ frequency modes
                newind = np.logical_and(bfind[findex:findex+nfreq], inc == False)

                if np.sum(newind) > 0:
                    lp -= np.sum(np.log(m2signal['pmax'][newind] - m2signal['pmin'][newind]))
            elif m2signal['stype'] == 'dmspectrum' and m2signal['corr'] == 'single':
                fdmindex = int(np.sum(self.npfdm[:m2signal['pulsarind']])/2)
                nfreqdm = int(self.npfdm[m2signal['pulsarind']]/2)

                # Select the common frequency modes
                inc = np.logical_and(bfdmind[findex:findex+nfreq], bcurfdmind[findex:findex+nfreq])
                # Select the _new_ frequency modes
                newind = np.logical_and(bfind[findex:findex+nfreq], inc == False)

                if np.sum(newind) > 0:
                    lp -= np.sum(np.log(m2signal['pmax'][newind] - m2signal['pmin'][newind]))

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
                if m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'single':
                    indexfull = m2signal['parindex']+npropmod1-1
                    index = npropmod1-1
                    newparameters[indexfull] = m2signal['pmin'][index] + \
                            np.random.rand() * (m2signal['pmax'][index] - m2signal['pmin'][index])

        if npropmod2 > curmod2:
            for m2signal in self.ptasignals:
                if m2signal['stype'] == 'dmspectrum':
                    indexfull = m2signal['parindex']+npropmod2-1
                    index = npropmod2-1
                    newparameters[indexfull] = m2signal['pmin'][index] + \
                            np.random.rand() * (m2signal['pmax'][index] - m2signal['pmin'][index])

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
            if m2signal['stype'] == 'frequencyline':
                self.ptapsrs[m2signal['pulsarind']].SFfreqs[2*m2signal['npsrfreqindex']:2*m2signal['npsrfreqindex']+2] = parameters[m2signal['parindex']]

        for pindex in range(len(self.ptapsrs)):
            m2psr = self.ptapsrs[pindex]
            if m2psr.frequencyLinesAdded > 0:
                if self.likfunc == 'mark4ln':
                    m2psr.SFmat = singleFreqFourierModes(m2psr.toas, 10**m2psr.SFfreqs[::2])
                    m2psr.FFmat = np.append(m2psr.Fmat, m2psr.SFmat, axis=1)

                    m2psr.UtFF = np.dot(m2psr.Uimat, m2psr.FFmat)
                else:
                    m2psr.SFmat = singleFreqFourierModes(m2psr.toas, 10**m2psr.SFfreqs[::2])
                    m2psr.FFmat = np.append(m2psr.Fmat, m2psr.SFmat, axis=1)
                    #GtSF = np.dot(m2psr.Hmat.T, m2psr.SFmat)
                    #GGtSF = np.dot(m2psr.Hmat, GtSF)
                    #m2psr.GGtFF = np.append(m2psr.GGtF, GGtSF, axis=1)
                    #m2psr.GGtFF = np.dot(m2psr.Hmat, GtFF)

                    """
                    if m2psr.twoComponentNoise:
                        #GtSF = np.dot(m2psr.Hmat.T, m2psr.SFmat)
                        AGSF = np.dot(m2psr.AG, m2psr.SFmat)
                        m2psr.AGFF = np.append(m2psr.AGF, AGSF, axis=1)

                        #GtFF = np.dot(m2psr.Hmat.T, m2psr.FFmat)
                        #GtFF = np.append(m2psr.GtF, GtSF, axis=1)

                        #m2psr.AGFF = np.dot(m2psr.Amat.T, GtFF)
                    """

            if m2psr.dmfrequencyLinesAdded > 0:
                m2psr.SFdmmat = singleFreqFourierModes(m2psr.toas, 10**m2psr.SFdmfreqs[::2])
                #m2psr.DSF = np.dot(m2psr.Dmat, m2psr.SFdmmat)
                m2psr.DSF = (m2psr.Dvec * m2psr.SFdmmat.T).T
                m2psr.DFF = np.append(m2psr.DF, m2psr.DSF, axis=1)

                m2psr.EEmat = np.append(m2psr.FFmat, m2psr.DF, axis=1)
                GtEE = np.dot(m2psr.Hmat.T, m2psr.EEmat)
                m2psr.GGtEE = np.dot(m2psr.Hmat, GtEE)

                if m2psr.twoComponentNoise:
                    m2psr.AGFF = np.dot(m2psr.Amat.T, GtFF)

                    if self.likfunc in ['mark6', 'mark6fa', 'mark8', 'mark10']:
                        m2psr.AGEE = np.dot(m2psr.Amat.T, GtEE)

    """
    Complement loglikelihood. This is not really the full log-likelihood by itself.
    It is the part of the log-likelihood that is complementary to the compressed
    log-likelihood. This way, we can still do data compression and evidence
    calculation simultaneously

    This function is basically equal to mark2loglikelihood, but now for the
    complement
    """
    def comploglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # MARK A

        self.setPsrNoise(parameters)

        # If this is already evaluated in the likelihood, do not do it here
        if not self.skipUpdateToggle:
            # Put stuff here that can be skipped if the likelihood function is
            # called before this one
            # setPsrNoise is not here anymore, since jitter can differs
            # TODO: make a toggle that decides whether jitter is included

            if self.haveDetSources:
                self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
            if self.npgos[ii] > 0:
                if self.ptapsrs[ii].twoComponentNoise:
                    self.rGr[ii] = np.sum(self.ptapsrs[ii].AoGr ** 2 / self.ptapsrs[ii].Nwovec)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwovec))
                else:
                    if np.sum(psr.Jvec) > 0:
                        Nir = np.zeros(len(psr.detresiduals))
                        NiGc = np.zeros(psr.Hocmat.shape)
                        Jldet = 0

                        for cc, col in enumerate(psr.Umat.T):
                            u = (col == 1.0)
                            l = np.sum(u)
                            ji = 1.0 / psr.Jvec[cc]
                            ni = 1.0 / psr.Nvec[u]
                            beta = 1.0 / (np.sum(ni) + ji)
                            Ni = np.diag(ni) - beta * np.outer(ni, ni)

                            Nir[u] = np.dot(Ni, psr.detresiduals[u])
                            NiGc[u, :] = np.dot(Ni, psr.Hocmat[u, :])

                            Jldet += np.sum(np.log(psr.Nvec[u])) + \
                                    np.log(psr.Jvec[cc]) - \
                                    np.log(beta)
                    else:
                        Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                        NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hocmat.T).T
                        Jldet = np.sum(np.log(psr.Nvec))
                    GcNiGc = np.dot(self.ptapsrs[ii].Hocmat.T, NiGc)
                    GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)

                    try:
                        cf = sl.cho_factor(GcNiGc)
                        self.GNGldet[ii] = Jldet + \
                                2*np.sum(np.log(np.diag(cf[0])))
                        GcNiGcr = sl.cho_solve(cf, GcNir)
                    except np.linalg.LinAlgError:
                        print "comploglikelihood: GcNiGc singular"

                    self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                            - np.dot(GcNir, GcNiGcr)
            else:
                self.rGr[ii] = 0
                self.GNGldet[ii] = 0

        #print np.sum(self.npgos), -0.5*np.sum(self.rGr), -0.5*np.sum(self.GNGldet)

        # Now we are ready to return the log-likelihood
        return -0.5*np.sum(self.npgos)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet)



    """
    mark1 loglikelihood of the pta model/likelihood implementation

    This is the full likelihood, without any Woodbury expansions.
    """
    def mark1loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        self.setPsrNoise(parameters)

        self.constructPhiAndTheta(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK A

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        GtFtot = []
        GtDtot = []
        GtUtot = []
        uvec = np.array([])
        self.GCG[:] = 0
        for ii, psr in enumerate(self.ptapsrs):
            gindex = np.sum(self.npgs[:ii])
            ng = self.npgs[ii]

            # Two-component noise or not does not matter
            self.GCG[gindex:gindex+ng, gindex:gindex+ng] = \
                    np.dot(self.ptapsrs[ii].Hmat.T, (self.ptapsrs[ii].Nvec * self.ptapsrs[ii].Hmat.T).T)

            # Create the total GtF and GtD lists for addition of Red(DM) noise
            GtFtot.append(psr.GtF)
            GtDtot.append(psr.GtD)
            GtUtot.append(psr.GtU)
            uvec = np.append(uvec, psr.Jvec)

            self.Gr[gindex:gindex+ng] = np.dot(self.ptapsrs[ii].Hmat.T, self.ptapsrs[ii].detresiduals)

        # MARK B

        # Create the total GCG matrix
        # TODO: Remove the block_diag code in this file
        #GtF = sl.block_diag(*GtFtot)
        GtF = block_diag(*GtFtot)
        # MARK C

        self.GCG += blockmul(self.Phi, GtF.T, self.npf, self.npgs)

        # Mark D

        #GtD = sl.block_diag(*GtDtot)
        # TODO: Remove the block_diag code in this file
        GtD = block_diag(*GtDtot)

        # MARK E

        # Do not directly multiply. Use block multiplication.
        # TODO: For even more speed, these two could be combined
        self.GCG += blockmul(np.diag(self.Thetavec), GtD.T, self.npfdm, self.npgs)

        GtU = block_diag(*GtUtot)
        self.GCG += blockmul(np.diag(uvec), GtU.T, self.npu, self.npgs)

        # MARK F

        # Do the inversion
        try:
            cf = sl.cho_factor(self.GCG)
            GNGldet = 2*np.sum(np.log(np.diag(cf[0])))
            xi2 = np.dot(self.Gr, sl.cho_solve(cf, self.Gr))
        except np.linalg.LinAlgError as err:
            U, s, Vh = sl.svd(self.GCG)
            if not np.all(s > 0):
                raise ValueError("ERROR: GCG singular according to SVD")
            GNGldet = np.sum(np.log(s))
            xi2 = np.dot(self.Gr, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.Gr))))
            # print "  -- msg: ", err.message

        # MARK G

        return -0.5 * np.sum(self.npgs)*np.log(2*np.pi) \
                -0.5 * xi2 - 0.5 * GNGldet



    """
    mark2 loglikelihood of the pta model/likelihood implementation

    This likelihood can only deal with efac/equad signals
    """
    def mark2loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # MARK A

        self.setPsrNoise(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
            else:
                if np.sum(psr.Jvec) > 0:
                    Nir = np.zeros(len(psr.detresiduals))
                    NiGc = np.zeros(psr.Hcmat.shape)
                    Jldet = 0

                    for cc, col in enumerate(psr.Umat.T):
                        u = (col == 1.0)
                        l = np.sum(u)
                        mat = np.ones((l, l)) * psr.Jvec[cc]
                        mat[range(l), range(l)] += psr.Nvec[u]
                        cf = sl.cho_factor(mat)

                        Nir[u] = sl.cho_solve(cf, psr.detresiduals[u])
                        NiGc[u, :] = sl.cho_solve(cf, psr.Hcmat[u, :])

                        Jldet += np.sum(np.log(psr.Nvec[u])) + \
                                np.log(psr.Jvec[cc]) - \
                                np.log(beta)
                else:
                    Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                    NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                    Jldet = np.sum(np.log(psr.Nvec))

                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = Jldet + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
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
    setup_mark3 = "import numpy as np, piccard as pic, matplotlib.pyplot as plt
    ; m3lik = pic.ptaLikelihood() ; m3lik.initFromFile('mdc1-open1.h5') ;
    m3lik.initModelOld(15, modelIndependentGWB=False, modelIndependentNoise=False, modelIndependentDM=False, modelIndependentAniGWB=False, varyEfac=False, incRedNoise=True, incEquad=False, separateEfacs=False, incGWB=True, incDM=False, incAniGWB=False, lAniGWB=2, likfunc='mark3') ; m3lik.initPrior()"
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

        # MARK ??
        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)

            if psr.twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                NGGF = ((1.0/psr.Nwvec) * psr.AGF.T).T

                self.rGr[ii] = np.sum(psr.AGr ** 2 / psr.Nwvec)
                self.rGF[findex:findex+2*nfreq] = np.dot(psr.AGr, NGGF)
                self.GNGldet[ii] = np.sum(np.log(psr.Nwvec))
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = \
                        np.dot(psr.AGF.T, NGGF)
            else:
                if np.sum(psr.Jvec) > 0:
                    Nir = np.zeros(len(psr.detresiduals))
                    NiGc = np.zeros(psr.Hcmat.shape)
                    NiF = np.zeros(psr.Fmat.shape)
                    Jldet = 0

                    for cc, col in enumerate(psr.Umat.T):
                        u = (col == 1.0)
                        ji = 1.0 / psr.Jvec[cc]
                        ni = 1.0 / psr.Nvec[u]
                        beta = 1.0 / (np.sum(ni) + ji)
                        Ni = np.diag(ni) - beta * np.outer(ni, ni)

                        Nir[u] = np.dot(Ni, psr.detresiduals[u])
                        NiGc[u, :] = np.dot(Ni, psr.Hcmat[u, :])
                        NiF[u, :] = np.dot(Ni, psr.Fmat[u, :])

                        Jldet += np.sum(np.log(psr.Nvec[u])) + \
                                np.log(psr.Jvec[cc]) - \
                                np.log(beta)
                elif np.sum(psr.Jvec) > 0 and False:
                    # This is a slower alternative for the above
                    Nir = np.zeros(len(psr.detresiduals))
                    NiGc = np.zeros(psr.Hcmat.shape)
                    NiF = np.zeros(psr.Fmat.shape)
                    Jldet = 0

                    for cc, col in enumerate(psr.Umat.T):
                        u = (col == 1.0)
                        l = np.sum(u)
                        mat = np.diag(psr.Nvec[u]) + np.ones(l) * psr.Jvec[cc]
                        cf = sl.cho_factor(mat)
                        Ni = sl.cho_solve(cf, np.eye(l))
                        Nir[u] = np.dot(Ni, psr.detresiduals[u])
                        NiGc[u, :] = np.dot(Ni, psr.Hcmat[u, :])
                        NiF[u, :] = np.dot(Ni, psr.Fmat[u, :])
                        Jldet += 2*np.sum(np.log(np.diag(cf[0])))
                else:
                    Nir = psr.detresiduals / psr.Nvec
                    NiGc = ((1.0/psr.Nvec) * psr.Hcmat.T).T
                    NiF = ((1.0/psr.Nvec) * psr.Fmat.T).T
                    Jldet = np.sum(np.log(psr.Nvec))

                GcNiGc = np.dot(psr.Hcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, psr.detresiduals)
                GcNiF = np.dot(NiGc.T, psr.Fmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = Jldet + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(psr.detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(psr.detresiduals, NiF) \
                        - np.dot(GcNir, GcNiGcF)
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = \
                        np.dot(NiF.T, psr.Fmat) - np.dot(GcNiF.T, GcNiGcF)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            PhiLD = np.sum(np.log(np.diag(self.Phi)))
            Phiinv = np.diag(1.0 / np.diag(self.Phi))
        else:
            try:
                #cf = sl.cho_factor(self.Phi + 1.0e-20*np.eye(self.Phi.shape[0]))
                cf = sl.cho_factor(self.Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(self.Phi.shape[0]))
            except np.linalg.LinAlgError:
                try:
                    U, s, Vh = sl.svd(self.Phi)
                    if not np.all(s > 0):
                        raise ValueError("ERROR: Phi singular according to SVD")
                    PhiLD = np.sum(np.log(s))
                    Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
                except np.linalg.LinAlgError:
                    return -np.inf

                #print "Fallback to SVD for Phi"

        # MARK E

        # Construct and decompose Sigma
        self.Sigma = self.FGGNGGF + Phiinv
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGF, sl.cho_solve(cf, self.rGF))
        except np.linalg.LinAlgError:
            try:
                U, s, Vh = sl.svd(self.Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                SigmaLD = np.sum(np.log(s))
                rGSigmaGr = np.dot(self.rGF, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF))))
            except np.linalg.LinAlgError:
                return -np.inf

        # Mark F
        #print np.sum(self.npgs), -0.5*np.sum(self.rGr), -0.5*np.sum(self.GNGldet)

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

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
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
                if np.sum(psr.Jvec) > 0:
                    Nir = np.zeros(len(psr.detresiduals))
                    NiGc = np.zeros(psr.Hcmat.shape)
                    NiF = np.zeros(psr.Fmat.shape)
                    Jldet = 0

                    for cc, col in enumerate(psr.Umat.T):
                        u = (col == 1.0)
                        l = np.sum(u)
                        ji = 1.0 / psr.Jvec[cc]
                        ni = 1.0 / psr.Nvec[u]
                        beta = 1.0 / (np.sum(ni) + ji)
                        Ni = np.diag(ni) - beta * np.outer(ni, ni)

                        Nir[u] = np.dot(Ni, psr.detresiduals[u])
                        NiGc[u, :] = np.dot(Ni, psr.Hcmat[u, :])
                        NiF[u, :] = np.dot(Ni, psr.Fmat[u, :])

                        Jldet += np.sum(np.log(psr.Nvec[u])) + \
                                np.log(psr.Jvec[cc]) - \
                                np.log(beta)
                else:
                    Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                    NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                    NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Fmat.T).T
                    Jldet = np.sum(np.log(psr.Nvec))

                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].Fmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = Jldet + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].detresiduals, NiF) \
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

    TODO: does not do deterministic sources yet
    """
    def mark4loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # MARK A

        # We do noise explicitly
        # Jitter is done here in the likelihood. Do not add to diagonal noise
        self.setPsrNoise(parameters)

        # MARK B

        self.constructPhiAndTheta(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)
            uindex = np.sum(self.npu[:ii])
            nus = self.npu[ii]

            if psr.twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGU))
                NGGU = ((1.0/psr.Nwvec) * psr.AGU.T).T

                self.rGr[ii] = np.sum(psr.AGr ** 2 / psr.Nwvec)
                self.rGU[uindex:uindex+nus] = np.dot(psr.AGr, NGGU)
                self.GNGldet[ii] = np.sum(np.log(psr.Nwvec))
                self.UGGNGGU[uindex:uindex+nus, uindex:uindex+nus] = np.dot(psr.AGU.T, NGGU)
            else:
                Nir = psr.detresiduals / psr.Nvec
                NiGc = ((1.0/psr.Nvec) * psr.Hcmat.T).T
                GcNiGc = np.dot(psr.Hcmat.T, NiGc)
                NiU = ((1.0/psr.Nvec) * psr.Umat.T).T
                GcNir = np.dot(NiGc.T, psr.detresiduals)
                GcNiU = np.dot(NiGc.T, psr.Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(psr.Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(psr.detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGU[uindex:uindex+nus] = np.dot(psr.detresiduals, NiU) \
                        - np.dot(GcNir, GcNiGcU)
                self.UGGNGGU[uindex:uindex+nus, uindex:uindex+nus] = \
                        np.dot(NiU.T, psr.Umat) - np.dot(GcNiU.T, GcNiGcU)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            # Quick and dirty:
            #PhiU = ( * self.ptapsrs[ii].AGU.T).T

            # ThetaLD = np.sum(np.log(self.Thetavec))


            UPhiU = np.dot(self.ptapsrs[0].UtF, np.dot(self.Phi, self.ptapsrs[0].UtF.T))
            Phi = UPhiU + np.diag(self.ptapsrs[0].Jvec)

            if len(self.Thetavec) > 0:
                #UThetaU = np.dot(self.ptapsrs[0].UtD, (self.Thetavec * self.ptapsrs[0].UtD).T)
                UThetaU = np.dot(self.ptapsrs[0].UtD, np.dot(np.diag(self.Thetavec), self.ptapsrs[0].UtD.T))
                Phi += UThetaU

            #PhiLD = np.sum(np.log(np.diag(Phi)))
            #Phiinv = np.diag(1.0 / np.diag(Phi))
            try:
                cf = sl.cho_factor(Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                try:
                    U, s, Vh = sl.svd(Phi)
                    if not np.all(s > 0):
                        raise ValueError("ERROR: Phi singular according to SVD")
                    PhiLD = np.sum(np.log(s))
                    Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
                except np.linalg.LinAlgError:
                    return -np.inf

                #print "Fallback to SVD for Phi", parameters
        else:
            # Do it for multiple pulsars
            UPhiU = np.zeros(self.UGGNGGU.shape)
            for ii, psri in enumerate(self.ptapsrs):
                uindexi = np.sum(self.npu[:ii])
                nusi = self.npu[ii]
                findexi = np.sum(self.npf[:ii])
                nfreqi = int(self.npf[ii])
                findexdmi = np.sum(self.npfdm[:ii])
                nfreqdmi = int(self.npfdm[ii])
                for jj in range(ii, len(self.ptapsrs)):
                    uindexj = np.sum(self.npu[:jj])
                    nusj = self.npu[jj]
                    psrj = self.ptapsrs[jj]
                    findexj = np.sum(self.npf[:jj])
                    nfreqj = int(self.npf[jj])

                    UPhiU[uindexi:uindexi+nusi, uindexj:uindexj+nusj] = \
                            np.dot(psri.UtF, np.dot( \
                            self.Phi[findexi:findexi+nfreqi, findexj:findexj+nfreqj], \
                                        psrj.UtF.T))

                    if ii == jj:
                        di = np.diag_indices(min(nusi, nusj))
                        UPhiU[uindexi:uindexi+nusi, uindexj:uindexj+nusj][di] += \
                            self.ptapsrs[ii].Jvec
                    else:
                        UPhiU[uindexj:uindexj+nusj, uindexi:uindexi+nusi] = \
                            UPhiU[uindexi:uindexi+nusi, uindexj:uindexj+nusj].T

                    if len(self.Thetavec) > 0 and ii == jj:
                        UTU = np.dot(psri.UtD, (self.Thetavec[findexdmi:findexdmi+nfreqdmi] * psrj.UtD).T)
                        UPhiU[uindexi:uindexi+nusi, uindexj:uindexj+nusj] += UTU

            try:
                cf = sl.cho_factor(UPhiU)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(UPhiU.shape[0]))
            except np.linalg.LinAlgError:
                try:
                    U, s, Vh = sl.svd(UPhiU)
                    if not np.all(s > 0):
                        raise ValueError("ERROR: UPhiU singular according to SVD")
                    PhiLD = np.sum(np.log(s))
                    Phiinv = np.dot(Vh.T, ((1.0/s) * U).T)
                except np.linalg.LinAlgError:
                    return -np.inf


        # MARK E

        # Construct and decompose Sigma
        self.Sigma = self.UGGNGGU + Phiinv
        try:
            cf = sl.cho_factor(self.Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(self.rGU, sl.cho_solve(cf, self.rGU))
        except np.linalg.LinAlgError:
            try:
                U, s, Vh = sl.svd(self.Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                SigmaLD = np.sum(np.log(s))
                rGSigmaGr = np.dot(self.rGU, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGU))))
            except np.linalg.LinAlgError:
                return -np.inf

                #print "Fallback to SVD for Sigma", parameters
        # Mark F

        # Now we are ready to return the log-likelihood
        # print np.sum(self.npgs), -0.5*np.sum(self.rGr), -0.5*np.sum(self.GNGldet)
             
        #print self.Phi

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

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
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
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                NiU = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Umat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiU = np.dot(NiGc.T, self.ptapsrs[ii].Umat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcU = sl.cho_solve(cf, GcNiU)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGU[uindex:uindex+nus] = np.dot(self.ptapsrs[ii].detresiduals, NiU) \
                        - np.dot(GcNir, GcNiGcU)
                self.UGGNGGU[uindex:uindex+nus, uindex:uindex+nus] = \
                        np.dot(NiU.T, self.ptapsrs[ii].Umat) - np.dot(GcNiU.T, GcNiGcU)



        # MARK D
        
        # Now that all arrays are filled, we can proceed to do some linear
        # algebra. First we'll invert Phi. For a single pulsar, this will be
        # diagonal
        if npsrs == 1:
            # Quick and dirty:
            #PhiU = ( * self.ptapsrs[ii].AGU.T).T

            UPhiU = np.dot(self.ptapsrs[0].UtFF, np.dot(self.Phi, self.ptapsrs[0].UtFF.T))
            Phi = UPhiU + np.diag(self.ptapsrs[0].Jvec)

            #PhiLD = np.sum(np.log(np.diag(Phi)))
            #Phiinv = np.diag(1.0 / np.diag(Phi))
            try:
                cf = sl.cho_factor(Phi)
                PhiLD = 2*np.sum(np.log(np.diag(cf[0])))
                Phiinv = sl.cho_solve(cf, np.identity(Phi.shape[0]))
            except np.linalg.LinAlgError:
                try:
                    U, s, Vh = sl.svd(Phi)
                    if not np.all(s > 0):
                        raise ValueError("ERROR: Phi singular according to SVD")
                    PhiLD = np.sum(np.log(s))
                    Phiinv = np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T))
                except np.linalg.LinAlgError:
                    return -np.inf

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

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
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
                if np.sum(psr.Jvec) > 0:
                    Nir = np.zeros(len(psr.detresiduals))
                    NiGc = np.zeros(psr.Hcmat.shape)
                    NiE = np.zeros(psr.Emat.shape)
                    Jldet = 0

                    for cc, col in enumerate(psr.Umat.T):
                        u = (col == 1.0)
                        l = np.sum(u)
                        ji = 1.0 / psr.Jvec[cc]
                        ni = 1.0 / psr.Nvec[u]
                        beta = 1.0 / (np.sum(ni) + ji)
                        Ni = np.diag(ni) - beta * np.outer(ni, ni)

                        Nir[u] = np.dot(Ni, psr.detresiduals[u])
                        NiGc[u, :] = np.dot(Ni, psr.Hcmat[u, :])
                        NiE[u, :] = np.dot(Ni, psr.Emat[u, :])

                        Jldet += np.sum(np.log(psr.Nvec[u])) + \
                                np.log(psr.Jvec[cc]) - \
                                np.log(beta)
                else:
                    Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                    NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                    NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Emat.T).T
                    Jldet = np.sum(np.log(psr.Nvec))

                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].Emat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = Jldet + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].detresiduals, NiE) \
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
        self.Sigma = self.EGGNGGE.copy()
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
            for jj in range(ii+1, npsrs):
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

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
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
                if np.sum(psr.Jvec) > 0:
                    Nir = np.zeros(len(psr.detresiduals))
                    NiGc = np.zeros(psr.Hcmat.shape)
                    NiE = np.zeros(psr.Emat.shape)
                    Jldet = 0

                    for cc, col in enumerate(psr.Umat.T):
                        u = (col == 1.0)
                        l = np.sum(u)
                        ji = 1.0 / psr.Jvec[cc]
                        ni = 1.0 / psr.Nvec[u]
                        beta = 1.0 / (np.sum(ni) + ji)
                        Ni = np.diag(ni) - beta * np.outer(ni, ni)

                        Nir[u] = np.dot(Ni, psr.detresiduals[u])
                        NiGc[u, :] = np.dot(Ni, psr.Hcmat[u, :])
                        NiE[u, :] = np.dot(Ni, psr.Emat[u, :])

                        Jldet += np.sum(np.log(psr.Nvec[u])) + \
                                np.log(psr.Jvec[cc]) - \
                                np.log(beta)
                else:
                    Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                    NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                    NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Emat.T).T
                    Jldet = np.sum(np.log(psr.Nvec))

                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].Emat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = Jldet + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].detresiduals, NiE) \
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

    NOTE: Since JSON update this needs some tweaks
    """
    def mark7loglikelihood(self, parameters, psrbfinc=None, psrbfdminc=None, \
            psrnfinc=None, psrnfdminc=None):
        npsrs = len(self.ptapsrs)

        # The red signals
        self.constructPhiAndTheta(parameters)

        # The white noise
        self.setPsrNoise(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

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
        for ii, psr in enumerate(self.ptapsrs):
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
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].lFmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].lFmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].detresiduals, NiF) \
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
    setup_mark8 = "import numpy as np, piccard as pic, matplotlib.pyplot as plt
    ; m3lik = pic.ptaLikelihood() ; m3lik.initFromFile('J0437.h5') ;
    m3lik.initModelOld(30, modelIndependentGWB=False, modelIndependentNoise=True, modelIndependentDM=True, modelIndependentAniGWB=False, varyEfac=True, incRedNoise=True, incEquad=True, separateEfacs=True, incGWB=False, incDM=True, incAniGWB=False, lAniGWB=2, likfunc='mark6') ; m3lik.initPrior()"
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

    NOTE: Since JSON update this needs some tweaks
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

        if self.haveDetSources:
            self.updateDetSources(parameters)

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
        for ii, psr in enumerate(self.ptapsrs):
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
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].lEmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].lEmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].detresiduals, NiE) \
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
        print "WARNING: mark8loglikelihood not yet fixed for more than one pulsar"

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

    NOTE: Since JSON update this needs some tweaks
    """
    def mark9loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # First re-construct the frequency matrices here...
        self.updateSpectralLines(parameters)

        # MARK A

        self.setPsrNoise(parameters)

        # MARK B

        self.constructPhiAndTheta(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
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
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].FFmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].FFmat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcF = sl.cho_solve(cf, GcNiF)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGF[findex:findex+2*nfreq] = np.dot(self.ptapsrs[ii].detresiduals, NiF) \
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

    NOTE: Since JSON update this needs some tweaks
    """
    def mark10loglikelihood(self, parameters):
        npsrs = len(self.ptapsrs)

        # The red signals
        self.constructPhiAndTheta(parameters)

        # The white noise
        self.setPsrNoise(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii, psr in enumerate(self.ptapsrs):
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
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                NiE = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Emat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiE = np.dot(NiGc.T, self.ptapsrs[ii].Emat)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                    GcNiGcE = sl.cho_solve(cf, GcNiE)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                        - np.dot(GcNir, GcNiGcr)
                self.rGE[findex+fdmindex:findex+fdmindex+2*nfreq+2*nfreqdm] = np.dot(self.ptapsrs[ii].detresiduals, NiE) \
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

        print "WARNGIN: mark10loglikelihood not yet fixed for more than one pulsar"

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


    """
    mark11 loglikelihood of the pta model/likelihood implementation

    This likelihood evaluates all the Gaussian processes individually. All the
    noise matrices are therefore diagonal
    """
    def mark11loglikelihood(self, parameters):
        # The red signals
        self.constructPhiAndTheta(parameters, phimat=False)

        # The white noise
        self.setPsrNoise(parameters)

        # The deterministic sources
        self.updateDetSources(parameters)

        for ii, psr in enumerate(self.ptapsrs):
            self.rGr[ii] = np.sum(psr.detresiduals**2 / psr.Nvec)
            self.GNGldet[ii] = np.sum(np.log(psr.Nvec))

            if psr.fourierind is not None:
                findex = np.sum(self.npf[:ii])
                nfreq = self.npf[ii]
                ind = psr.fourierind

                self.rGr[ii] += np.sum(parameters[ind:ind+nfreq]**2 / \
                    self.Phivec[findex:findex+nfreq])

                self.GNGldet[ii] += np.sum(np.log(self.Phivec))

            if psr.dmfourierind is not None:
                fdmindex = np.sum(self.npfdm[:ii])
                nfreqdm = self.npfdm[ii]
                ind = psr.fourierind

                self.rGr[ii] += np.sum(parameters[ind:ind+nfreqdm]**2 / \
                    self.Thetavec[fdmindex:fdmindex+nfreqdm])

                self.GNGldet[ii] += np.sum(np.log(self.Thetavec))

            if psr.jitterfourierind is not None:
                uindex = np.sum(self.npu[:ii])
                npus = self.npu[ii]
                ind = psr.jitterfourierind
                self.Muvec[uindex:uindex+npus] = psr.Jvec

                self.rGr[ii] += np.sum(parameters[ind:ind+npus]**2 / \
                    self.Muvec[uindex:uindex+npus])

                self.GNGldet[ii] += np.sum(np.log(self.Muvec))

        return -0.5*np.sum(self.npobs)*np.log(2*np.pi) \
                -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet)



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
            elif self.likfunc == 'mark10':
                ll = self.mark10loglikelihood(parameters)
            elif self.likfunc == 'mark11':
                ll = self.mark11loglikelihood(parameters)

            if self.evallikcomp:
                self.skipUpdateToggle = True
                ll += self.comploglikelihood(parameters)
                self.skipUpdateToggle = False
        else:
            ll = -1e99

        return ll

    # TODO: the prior for the amplitude parameters is not yet normalised
    def mark4logprior(self, parameters):
        lp = 0.0

        # Loop over all signals
        for m2signal in self.ptasignals:
            if m2signal['stype'] == 'powerlaw' and m2signal['corr'] == 'anisotropicgwb':
                nclm = m2signal['aniCorr'].clmlength()
                # lp += parameters[m2signal['parindex']]

                sparameters = m2signal['pstart'].copy()
                nvaryclm = np.sum(m2signal['bvary'][3:])
                nskip = np.sum(m2signal['bvary'][:3])
                sparameters[3:][m2signal['bvary'][3:]] = \
                        parameters[m2signal['parindex']+nskip:m2signal['parindex']+nskip+nvaryclm]

                clm = sparameters[m2signal['ntotpars']-nclm:m2signal['ntotpars']]
                if m2signal['aniCorr'].priorIndicator(clm) == False:
                    lp -= 1e99
            elif m2signal['stype'] == 'powerlaw' and m2signal['corr'] != 'single':
                lp += parameters[m2signal['parindex']]
            elif m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'anisotropicgwb':
                nclm = m2signal['aniCorr'].clmlength()
                sparameters = m2signal['pstart'].copy()
                nfreqs = m2signal['ntotpars'] - nclm
                nvaryclm = np.sum(m2signal['bvary'][nfreqs:])
                nskip = np.sum(m2signal['bvary'][:nfreqs])
                sparameters[nfreqs:][m2signal['bvary'][nfreqs:]] = \
                        parameters[m2signal['parindex']+nskip:m2signal['parindex']+nskip+nvaryclm]

                clm = sparameters[m2signal['parindex']+m2signal['ntotpars']-nclm:m2signal['parindex']+m2signal['ntotpars']]

                if m2signal['aniCorr'].priorIndicator(clm) == False:
                    lp -= 1e99
            elif m2signal['stype'] == 'spectrum' and m2signal['corr'] != 'single':
                lp += np.sum(parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']])

            # Divide by the prior range
            if np.sum(m2signal['bvary']) > 0:
                lp -= np.sum(np.log(m2signal['pmax'][m2signal['bvary']]-m2signal['pmin'][m2signal['bvary']]))
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
            if m2signal['stype'] == 'spectrum' and m2signal['corr'] == 'single':
                # Red noise, see if we need to include it
                findex = int(np.sum(self.npf[:m2signal['pulsarind']])/2)
                nfreq = int(self.npf[m2signal['pulsarind']]/2)
                inc = np.logical_and(bfind[findex:findex+nfreq], bcurfind[findex:findex+nfreq])

                if np.sum(inc) > 0:
                    lp -= np.sum(np.log(m2signal['pmax'][inc] - m2signal['pmin'][inc]))
                    #lp -= np.sum(inc) * 1.0
            elif m2signal['stype'] == 'dmspectrum' and m2signal['corr'] == 'single':
                fdmindex = int(np.sum(self.npfdm[:m2signal['pulsarind']])/2)
                nfreqdm = int(self.npfdm[m2signal['pulsarind']]/2)
                inc = np.logical_and(bfdmind[findex:findex+nfreq], bcurfdmind[findex:findex+nfreq])

                if np.sum(inc) > 0:
                    lp -= np.sum(np.log(m2signal['pmax'][inc] - m2signal['pmin'][inc]))
                    #lp -= np.sum(inc) * 1.0
            else:
                if np.sum(m2signal['bvary']) > 0:
                    lp -= np.sum(np.log(m2signal['pmax'][m2signal['bvary']]-m2signal['pmin'][m2signal['bvary']]))

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
                    if m2signal['stype'] == 'frequencyline':
                        self.ptapsrs[m2signal['pulsarind']].SFfreqs[2*m2signal['npsrfreqindex']:2*m2signal['npsrfreqindex']+2] = parameters[m2signal['parindex']]

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
            elif self.likfunc == 'mark10':  # Mark9 ''
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

    def loglikelihoodhc(self, cube, ndim, nparams):
        acube = np.zeros(ndim)

        for ii in range(ndim):
            acube[ii] = cube[ii]

        return self.loglikelihood(acube)


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

    NOTE: the G-matrix is ignored when generating data (so generating pre-fit
    data)

    TODO: repeated calls of this function increasingly introduce errors.
          Something is not initialised back to zero every call. Investigate
    """
    def gensig(self, parameters=None, filename=None, timedomain=False):
        if parameters == None:
            parameters = self.pstart.copy()

        npsrs = len(self.ptapsrs)

        self.setPsrNoise(parameters)

        if self.haveStochSources:
            self.constructPhiAndTheta(parameters)

        # Allocate some auxiliary matrices
        Cov = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
        totFmat = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totDFmat = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totDmat = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
        totG = np.zeros((np.sum(self.npobs), np.sum(self.npgs)))
        totU = np.zeros((np.sum(self.npobs), np.sum(self.npu)))
        tottoas = np.zeros(np.sum(self.npobs))
        tottoaerrs = np.zeros(np.sum(self.npobs))

        # Fill the auxiliary matrices
        for ii, psr in enumerate(self.ptapsrs):
            nindex = np.sum(self.npobs[:ii])
            findex = np.sum(self.npf[:ii])
            fdmindex = np.sum(self.npfdm[:ii])
            uindex = np.sum(self.npu[:ii])
            gindex = np.sum(self.npgs[:ii])
            npobs = self.npobs[ii]
            nppf = self.npf[ii]
            nppfdm = self.npfdm[ii]
            npgs = self.npgs[ii]
            npus = self.npu[ii]

            # Start creating the covariance matrix with noise
            Cov[nindex:nindex+npobs, nindex:nindex+npobs] = np.diag(psr.Nvec)

            if psr.Fmat.shape[1] == nppf:
                totFmat[nindex:nindex+npobs, findex:findex+nppf] = psr.Fmat

            if not psr.DF is None and psr.DF.shape[1] == nppfdm:
                totDFmat[nindex:nindex+npobs, fdmindex:fdmindex+nppfdm] = psr.DF
                totDmat[nindex:nindex+npobs, nindex:nindex+npobs] = psr.Dmat

            if not psr.Umat is None and psr.Umat.shape[1] == npus:
                totU[nindex:nindex+npobs, uindex:uindex+npus] = psr.Umat

            totG[nindex:nindex+npobs, gindex:gindex+npgs] = self.ptapsrs[ii].Hmat
            tottoas[nindex:nindex+npobs] = self.ptapsrs[ii].toas
            tottoaerrs[nindex:nindex+npobs] = self.ptapsrs[ii].toaerrs


        # TODO: time-domain piece is very outdated. Update!
        if timedomain:
            # The time-domain matrices for red noise and DM variations
            Cr = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals
            Cdm = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))    # Time domain red DM signals

            # Do time-domain stuff explicitly here, for now
            for m2signal in self.ptasignals:
                sparameters = m2signal['pstart'].copy()
                sparameters[m2signal['bvary']] = \
                        parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']]

                if m2signal['stype'] == 'powerlaw' and m2signal['corr'] == 'single':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    nindex = np.sum(self.npobs[:m2signal['pulsarind']])
                    ncurobs = self.npobs[m2signal['pulsarind']]

                    Cr[nindex:nindex+ncurobs,nindex:nindex+ncurobs] +=\
                            Cred_sec(self.ptapsrs[m2signal['pulsarind']].toas,\
                            alpha=0.5*(3-Si),\
                            fL=1.0/100) * (Amp**2)
                elif m2signal['stype'] == 'dmpowerlaw' and m2signal['corr'] == 'single':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    nindex = np.sum(self.npobs[:m2signal['pulsarind']])
                    ncurobs = self.npobs[m2signal['pulsarind']]

                    Cdm[nindex:nindex+ncurobs,nindex:nindex+ncurobs] +=\
                            Cred_sec(self.ptapsrs[m2signal['pulsarind']].toas,\
                            alpha=0.5*(3-Si),\
                            fL=1.0/100) * (Amp**2)

            Cov += Cr
            Cov += np.dot(totDmat, np.dot(Cdm, totDmat))
        else:
            # Construct them from Phi/Theta
            if not self.Phi is None:
                Cov += np.dot(totFmat, np.dot(self.Phi, totFmat.T))

            if self.Thetavec is not None and len(self.Thetavec) == totDFmat.shape[1]:
                Cov += np.dot(totDFmat, np.dot(np.diag(self.Thetavec), totDFmat.T))

            # Include jitter
            qvec = np.array([])
            for pp, psr in enumerate(self.ptapsrs):
                qvec = np.append(qvec, psr.Jvec)

            if totU.shape[1] == len(qvec):
                Cov += np.dot(totU, (qvec * totU).T)

        # Create the projected covariance matrix, and decompose it
        # WARNING: for now we are ignoring the G-matrix when generating data
        if False:
            totG = np.eye(Cov.shape[0])
            GCG = Cov
        else:
            GCG = np.dot(totG.T, np.dot(Cov, totG))

        try:
            cf = sl.cholesky(GCG).T
        except np.linalg.LinAlgError as err:
            U, s, Vh = sl.svd(GCG)
            if not np.all(s > 0):
                raise ValueError("ERROR: GCG singular according to SVD")
            # TODO: check if this is the right order?
            cf = np.dot(U, np.diag(np.sqrt(s)))

        # Generate the data in the Cholesky-basis
        xi = np.random.randn(GCG.shape[0])
        ygen = np.dot(totG, np.dot(cf, xi))

        # Save the data
        tindex = 0
        for ii in range(len(self.ptapsrs)):
            nobs = len(self.ptapsrs[ii].residuals)
            self.ptapsrs[ii].residuals = ygen[tindex:tindex+nobs]
            tindex += nobs

        # Add the deterministic sources:
        if self.haveDetSources:
            self.updateDetSources(parameters)

            for psr in self.ptapsrs:
                psr.residuals = 2 * psr.residuals + psr.detresiduals

        # If libstempo is installed, also update libstempo objects
        if t2 is not None:
            for psr in self.ptapsrs:
                psr.initLibsTempoObject()
        else:
            print "WARNING: libstempo not imported. Par/tim files will not be updated"

        #"""
        # Display the data
        #plt.errorbar(tottoas, ygen, yerr=tottoaerrs, fmt='.', c='blue')
        #plt.errorbar(self.ptapsrs[0].toas, \
        #        self.ptapsrs[0].residuals, \
        #        yerr=self.ptapsrs[0].toaerrs, fmt='.', c='blue')
        psr = self.ptapsrs[0]
        plt.scatter(psr.toas, psr.residuals)
        plt.axis([psr.toas.min(), psr.toas.max(), psr.residuals.min(), psr.residuals.max()])
                
        plt.grid(True)
        plt.show()
        #"""

        # If required, write all this to HDF5 file
        if filename != None:
            h5df = DataFile(filename)

            for ii, psr in enumerate(self.ptapsrs):
                h5df.addData(psr.name, 'prefitRes', psr.residuals, overwrite=True)
                h5df.addData(psr.name, 'postfitRes', psr.residuals, overwrite=True)

                # If libstempo is installed, update the objects, and write the
                # par/tim files
                if t2 is not None:
                    # Create ideal toas (subtract old residuals, add new ones)
                    psr.t2psr.stoas[:] -= psr.t2psr.residuals() / pic_spd
                    psr.t2psr.stoas[:] += psr.residuals / pic_spd
                    psr.t2psr.fit()

                    # Write temporary par/tim files, and read in memory
                    parfilename = tempfile.mktemp()
                    timfilename = tempfile.mktemp()
                    psr.t2psr.savepar(parfilename)
                    psr.t2psr.savetim(timfilename)
                    with open(parfilename, 'r') as content_file:
                        psr.parfile_content = content_file.read()
                    with open(timfilename, 'r') as content_file:
                        psr.timfile_content = content_file.read()
                    os.remove(parfilename)
                    os.remove(timfilename)

                    # Write the par/tim files to the HDF5 file
                    h5df.addData(psr.name, 'parfile', psr.parfile_content, overwrite=True)
                    h5df.addData(psr.name, 'timfile', psr.timfile_content, overwrite=True)



    """
    Blah

    TODO: check that the jitter stuff is set (make it mark4)
    """
    def gensigjit(self, parameters=None, filename=None, timedomain=False):
        if parameters == None:
            parameters = self.pstart.copy()

        npsrs = len(self.ptapsrs)

        self.setPsrNoise(parameters)

        if self.haveStochSources:
            self.constructPhiAndTheta(parameters)

        # Allocate some auxiliary matrices
        Cov = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
        totFmat = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        totDFmat = np.zeros((np.sum(self.npobs), np.sum(self.npf)))
        #totDmat = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
        totDvec = np.zeros(np.sum(self.npobs))
        totG = np.zeros((np.sum(self.npobs), np.sum(self.npgs)))
        totU = np.zeros((np.sum(self.npobs), np.sum(self.npu)))
        tottoas = np.zeros(np.sum(self.npobs))
        tottoaerrs = np.zeros(np.sum(self.npobs))

        # Fill the auxiliary matrices
        for ii in range(npsrs):
            nindex = np.sum(self.npobs[:ii])
            findex = np.sum(self.npf[:ii])
            fdmindex = np.sum(self.npfdm[:ii])
            uindex = np.sum(self.npu[:ii])
            gindex = np.sum(self.npgs[:ii])
            npobs = self.npobs[ii]
            nppf = self.npf[ii]
            nppfdm = self.npfdm[ii]
            npgs = self.npgs[ii]
            npus = self.npu[ii]
            #if self.ptapsrs[ii].twoComponentNoise:
            #    pass
            #else:
            #    pass
            Cov[nindex:nindex+npobs, nindex:nindex+npobs] = np.diag(self.ptapsrs[ii].Nvec)
            totFmat[nindex:nindex+npobs, findex:findex+nppf] = self.ptapsrs[ii].Fmat

            if self.ptapsrs[ii].DF is not None:
                totDFmat[nindex:nindex+npobs, fdmindex:fdmindex+nppfdm] = self.ptapsrs[ii].DF
                #totDmat[nindex:nindex+npobs, nindex:nindex+npobs] = self.ptapsrs[ii].Dmat
                totDvec[nindex:nindex+npobs] = self.ptapsrs[ii].Dvec

            totG[nindex:nindex+npobs, gindex:gindex+npgs] = self.ptapsrs[ii].Hmat
            totU[nindex:nindex+npobs, uindex:uindex+npus] = self.ptapsrs[ii].Umat
            tottoas[nindex:nindex+npobs] = self.ptapsrs[ii].toas
            tottoaerrs[nindex:nindex+npobs] = self.ptapsrs[ii].toaerrs


        if timedomain:
            # The time-domain matrices for red noise and DM variations
            Cr = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))     # Time domain red signals
            Cdm = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))    # Time domain red DM signals

            # Do time-domain stuff explicitly here, for now
            for m2signal in self.ptasignals:
                sparameters = m2signal['pstart'].copy()
                sparameters[m2signal['bvary']] = \
                        parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']]

                if m2signal['stype'] == 'powerlaw' and m2signal['corr'] == 'single':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    nindex = np.sum(self.npobs[:m2signal['pulsarind']])
                    ncurobs = self.npobs[m2signal['pulsarind']]

                    Cr[nindex:nindex+ncurobs,nindex:nindex+ncurobs] +=\
                            Cred_sec(self.ptapsrs[m2signal['pulsarind']].toas,\
                            alpha=0.5*(3-Si),\
                            fL=1.0/100) * (Amp**2)
                elif m2signal['stype'] == 'dmpowerlaw' and m2signal['corr'] == 'single':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    nindex = np.sum(self.npobs[:m2signal['pulsarind']])
                    ncurobs = self.npobs[m2signal['pulsarind']]

                    Cdm[nindex:nindex+ncurobs,nindex:nindex+ncurobs] +=\
                            Cred_sec(self.ptapsrs[m2signal['pulsarind']].toas,\
                            alpha=0.5*(3-Si),\
                            fL=1.0/100) * (Amp**2)


            Cov += Cr
            Cov += np.dot(np.diag(totDvec), np.dot(Cdm, np.diag(totDvec)))
        else:
            # Construct them from Phi/Theta
            Cov += np.dot(totFmat, np.dot(self.Phi, totFmat.T))
            if self.Thetavec is not None and len(self.Thetavec) == totDFmat.shape[1]:
                Cov += np.dot(totDFmat, np.dot(np.diag(self.Thetavec), totDFmat.T))

            # Include jitter
            qvec = np.array([])
            for pp, psr in enumerate(self.ptapsrs):
                qvec = np.append(qvec, psr.Jvec)
            Cov += np.dot(totU, (qvec * totU).T)

        # Create the projected covariance matrix, and decompose it
        # WARNING: for now we are ignoring the G-matrix when generating data
        if True:
            totG = np.eye(Cov.shape[0])
            GCG = Cov
        else:
            GCG = np.dot(totG.T, np.dot(Cov, totG))

        try:
            cf = sl.cholesky(GCG).T
        except np.linalg.LinAlgError as err:
            U, s, Vh = sl.svd(GCG)
            if not np.all(s > 0):
                raise ValueError("ERROR: GCG singular according to SVD")
            # TODO: check if this is the right order?
            cf = np.dot(U, np.diag(np.sqrt(s)))

        # Generate the data in the Cholesky-basis
        xi = np.random.randn(GCG.shape[0])
        ygen = np.dot(totG, np.dot(cf, xi))

        # Save the data
        tindex = 0
        for ii in range(len(self.ptapsrs)):
            nobs = len(self.ptapsrs[ii].residuals)
            self.ptapsrs[ii].residuals = ygen[tindex:tindex+nobs]
            tindex += nobs

        # Add the deterministic sources:
        if self.haveDetSources:
            self.updateDetSources(parameters)

            for psr in self.ptapsrs:
                psr.residuals = 2 * psr.residuals + psr.detresiduals

        """
        # Display the data
        #plt.errorbar(tottoas, ygen, yerr=tottoaerrs, fmt='.', c='blue')
        plt.errorbar(self.ptapsrs[0].toas, \
                self.ptapsrs[0].residuals, \
                yerr=self.ptapsrs[0].toaerrs, fmt='.', c='blue')
                
        plt.grid(True)
        plt.show()
        """

        # If required, write all this to HDF5 file
        if filename != None:
            h5df = DataFile(filename)

            for ii, psr in enumerate(self.ptapsrs):
                h5df.addData(psr.name, 'prefitRes', psr.residuals, overwrite=True)
                h5df.addData(psr.name, 'postfitRes', psr.residuals, overwrite=True)


    # TODO: The following few functions all do virtually the same thing: use
    #       maximum likelihood parameter values to construct ML covariance
    #       matrices in order to calculate things. Factorise it.


    """
    Using the (maximum likelihood) parameter values provided by the user, update
    the parameters of the par-file. If timing model parameters are taken into
    account numerically, the values provided by the user are used of course. If
    they are marginalised over analytically, then we need to calculate their
    value using a GLS fit. 

    chi = (M^T C^{-1} M)^{-1} M^{T} C^{-1} x
    x: the timing residuals
    C: the ML covariance matrix
    M: the design matrix including only the analytical timing model
    chi: the timing model parameters that need to be updated

    @param mlparameters:    The full vector of maximum likelihood parameters
                            (only those varying in the MCMC)

    @return:                Analytic updates to the timing model parameters

    TODO: between-pulsar correlations are ignored here!
    """
    def calculateAnalyticTMPars(self, mlparameters):
        npsrs = len(self.ptapsrs)

        self.setPsrNoise(mlparameters)

        self.constructPhiAndTheta(mlparameters)

        if self.haveDetSources:
            self.updateDetSources(mlparameters)

        # The full covariance matrix components
        allPhi = self.Phi.copy()
        allThetavec = self.Thetavec.copy()

        # The full C-matrix
        Cfull = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))

        # The analytic parameters
        chi = np.array([])

        for pp, psr in enumerate(self.ptapsrs):
            findex = np.sum(self.npf[:pp])
            fdmindex = np.sum(self.npfdm[:pp])
            nfreq = int(self.npf[pp]/2)
            nfreqdm = int(self.npfdm[pp]/2)
            nindex = np.sum(self.npobs[:pp])
            nobs = self.npobs[pp]

            # Include white noise
            Cfull[nindex:nindex+nobs, nindex:nindex+nobs] = np.diag(psr.Nvec)

            # Note: this excludes between-pulsar correlations!
            # Include red noise
            Cfull[nindex:nindex+nobs, nindex:nindex+nobs] += \
                    np.dot(psr.Fmat, np.dot(allPhi[findex:findex+2*nfreq], psr.Fmat.T))

            # Include DM Variations (what if it doesn't exist?)
            if len(self.getSignalNumbersFromDict(self.ptasignals, \
                    stype='dmpowerlaw', corr='single', psrind=pp)) > 0 or \
                    len(self.getSignalNumbersFromDict(self.ptasignals, \
                    stype='dmspectrum', corr='single', psrind=pp)) > 1:
                Cfull[nindex:nindex+nobs, nindex:nindex+nobs] += \
                        np.dot(psr.DF, (allThetavec[fdmindex:fdmindex+2*nfreqdm] * psr.DF).T)

            # Include jitter noise
            if len(self.getSignalNumbersFromDict(self.ptasignals, \
                    stype='jitter', corr='single', psrind=pp)) > 0:
                Cfull[nindex:nindex+nobs, nindex:nindex+nobs] += \
                        np.dot(psr.Umat, (psr.Jvec * psr.Umat).T)
                #        np.dot(psr.Umat, np.dot(psr.Qamp * np.eye(psr.Umat.shape[1]), \
                #        psr.Umat.T))

            # To get the adjusted design matrix, we need to jump through hoops.
            # TODO: put this in a separate function!
            # Get the timing model parameters we have numerically
            signals = self.ptasignals
            linsigind = self.getSignalNumbersFromDict(signals,
                    stype='lineartimingmodel', psrind=pp)
            nlsigind = self.getSignalNumbersFromDict(signals,
                    stype='nonlineartimingmodel', psrind=pp)

            tmsigpars = []    # All the timing model parameters of this pulsar
            for ss in np.append(linsigind, nlsigind):
                tmsigpars += signals[ss]['parid']

            # Obtain the indices
            ind = []
            for jj, parid in enumerate(psr.ptmdescription):
                if parid in tmsigpars:
                    ind += [False]
                else:
                    ind += [True]
            ind = np.array(ind, dtype=np.bool)

            # Now select the partial design matrix we need
            Mmat = psr.Mmat[:, ind]

            # Invert Cfull
            try:
                cfC = sl.cho_factor(Cfull)
                MCM = np.dot(Mmat.T, sl.cho_solve(cfC, Mmat))
                MCr = np.dot(Mmat.T, sl.cho_solve(cfC, psr.detresiduals))
                cfM = sl.cho_factor(MCM)
                psrchi = sl.cho_solve(cfM, MCr)
            except np.linalg.LinAlgError as err:
                U, s, Vh = sl.svd(Cfull)
                MCM = np.dot(Mmat.T, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, Mmat))))
                MCr = np.dot(Mmat.T, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, psr.detresiduals))))
                U, s, Vh = sl.svd(MCM)
                psrchi = np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, MCr)))

            chi = np.append(chi, psrchi)

        return chi


    """
    Update the tempo2 timing model parameters from the 'ML' parameters provided
    by the user. The parameters provided by the user are the fitting
    parameters of the sampling. The values of psr.ptmpars are updated, as are
    the tempo2 psr objects (if tempo2 is available)

    @param mlparameters:    The ML parameters
    @param parfiles:        If not none, a list of par-file names to write to
    """
    def updatePsrsFromFit(self, mlparameters, parfiles=None):
        # Obtain the deviations to the timing model parameters
        chi = self.calculateAnalyticTMPars(mlparameters)

        index = 0
        for pp, psr in enumerate(self.ptapsrs):
            # To get the adjusted design matrix, we need to jump through hoops.
            # TODO: put this in a separate function!
            # Get the timing model parameters we have numerically
            signals = self.ptasignals
            linsigind = self.getSignalNumbersFromDict(signals,
                    stype='lineartimingmodel', psrind=pp)
            nlsigind = self.getSignalNumbersFromDict(signals,
                    stype='nonlineartimingmodel', psrind=pp)

            tmsigpars = []    # All the timing model parameters of this pulsar
            for ss in np.append(linsigind, nlsigind):
                tmsigpars += signals[ss]['parid']

                # Update the psr parameters:
                for pi, parid in enumerate(signals[ss]['parid']):
                    ppind = list(psr.ptmdescription).index(parid)       # Index in pulsar
                    spind = np.sum(signals[ss]['bvary'][:pi])           # Index in mlparameters
                    tmpdelta = mlparameters[signals[ss]['parindex']+spind]

                    psr.ptmpars[ppind] = tmpdelta

            # Obtain the indices
            ind = []
            anpars = []
            for jj, parid in enumerate(psr.ptmdescription):
                if parid in tmsigpars:
                    ind += [False]
                else:
                    ind += [True]
                    anpars += [parid]
            ind = np.array(ind, dtype=np.bool)

            # The new, adjusted timing model parameters updates are then:
            tmpdelta = chi[index:index+np.sum(ind)]   # * psr.unitconversion[ind]

            # Update the analytically included timing model parameters
            psr.ptmpars[ind] += tmpdelta

            index += np.sum(ind)

            # Update the libstempo parameters if we have 'm
            if psr.t2psr is None:
                try:
                    psr.initLibsTempoObject()
                except ImportError:
                    print "WARNING: no libstempo present. Not updating parameters"
                    continue

            # Save/write the content of the new parfile
            for pi, parid in enumerate(psr.ptmdescription):
                if not parid == 'Offset' and psr.t2psr[parid].set:
                    psr.t2psr[parid].val = psr.ptmpars[pi]

            if parfiles is None:
                parfile = tempfile.mktemp()
            else:
                parfile = parfiles[pp]

            psr.t2psr.savepar(parfile)

            with open(parfile, 'r') as content_file:
                psr.parfile_content = content_file.read()

            if parfiles is None:
                os.remove(parfile)







    """
    Based on the signal number and the maximum likelihood parameters, this
    function reconstructs the signal of signum. Very useful to reconstruct only
    the white noise, or red noise, timing residuals.

    This function returns two arrays: the reconstructed signal, and the error

    If the user wants the actual DM signal, he/she can do that him/herself from
    the returned residuals

    TODO: Gr does not include detresiduals!!!! FIX THIS
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
        for ii, psr in enumerate(self.ptapsrs):
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
            totDvec[nindex:nindex+nobs] = self.ptapsrs[ii].Dvec

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
            totDvec[nindex:nindex+nobs] = self.ptapsrs[ii].Dvec

            totGFp[gindexp:gindexp+ngsp, findexp:findexp+2*nfreqp] = \
                    np.dot(predlikob.ptapsrs[ii].Gmat.T, predlikob.ptapsrs[ii].Fmat)
            totFp[nindexp:nindexp+nobsp, findexp:findexp+2*nfreqp] = \
                    predlikob.ptapsrs[ii].Fmat
            totGp[nindexp:nindexp+nobsp, gindexp:gindexp+ngsp] = predlikob.ptapsrs[ii].Gmat
            totGrp[gindexp:gindexp+ngsp] = predlikob.ptapsrs[ii].Gr
            totDvecp[nindexp:nindexp+nobsp] = predlikob.ptapsrs[ii].Dvec

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
This function creates a new set of simulated PTA datasets, based on a set of
existing par/tim files.

@param parlist: the par-files of the pulsars.
@param timlist: the tim-files of the pulsars. Using as input for the simulation
@param simlist: the tim-files with generated TOAs, based on the original tim-files
@param parameters: parameters of the model from which to generate the mock data
@param h5file:  the hdf5-file we will create which holds the newly simulated data
@param kwargs   all the same parameters given to 'makeModelDict', from which the model
                is built. The model should be compatible with 'parameters'
    
"""
def simulateFullSet(parlist, timlist, simlist, parameters, h5file, **kwargs):
    if len(parlist) != len(timlist) or len(parlist) != len(simlist):
        raise IOError("ERROR: list of par/tim/sim files should be of equal size")

    # 5reate the hdf5-file from the par/tim files
    h5df = DataFile(h5file)
    for ii in range(len(parlist)):
        h5df.addTempoPulsar(parlist[ii], timlist[ii])

    # Apply the model, and generate a realisation of data
    likob = ptaLikelihood(h5file)
    modeldict = likob.makeModelDict(**kwargs)
    likob.initModel(modeldict)
    likob.gensig(parameters=parameters, filename=h5file)

    # Write the sim-files to disk
    for ii in range(len(parlist)):
        psr = t2.tempopulsar(parlist[ii], timlist[ii])
        psr.stoas[:] -= psr.residuals() / pic_spd

        psr.stoas[:] += likob.ptapsrs[ii].residuals / pic_spd
        psr.savetim(simlist[ii])

        print "Writing mock TOAs of ", parlist[ii], "/", likob.ptapsrs[ii].name, \
                " to ", simlist[ii]
