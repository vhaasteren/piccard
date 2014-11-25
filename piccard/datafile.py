#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
datafile.py

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

from constants import *


try:    # If without libstempo, can still read hdf5 files
    import libstempo
    t2 = libstempo
except ImportError:
    t2 = None

# To do coordinate transformations, we need pyephem
try:
    import ephem
except ImportError:
    ephem = None


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


    def hasField(self, psrname, field):
        """
        Return whether or not this pulsar has a specific field present

        @param psrname:     Name of the pulsar we are reading data from
        @param field:       Field name of the data we are requestion
        """
        self.h5file = h5.File(self.filename, 'r')
        psrGroup = self.getPulsarGroup(psrname, delete=False)

        hasfield = (field in psrGroup)

        self.h5file.close()

        return hasfield

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
    @param maxobs:      Maximum number of observations (if None, use standard)
    """
    def addTempoPulsar(self, parfile, timfile, iterations=1, mode='replace',
            maxobs=None):
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
        t2pulsar = t2.tempopulsar('./'+relparfile, './'+reltimfile, \
                maxobs=maxobs)

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

        # Write the position of the pulsar, even if it is in ecliptic
        # coordinates
        if 'RAJ' in t2pulsar and 'DECJ' in t2pulsar:
            raj = t2pulsar.prefit['RAJ'].val
            decj = t2pulsar.prefit['DECJ'].val
        elif 'ELONG' in t2pulsar and 'ELAT' in t2pulsar:
            if ephem is None:
                raise ImportError("pyephem not installed")

            # tempo/tempo2 RAJ/DECJ always refer precession/nutation-wise to
            # epoch J2000. Posepoch does not apply here (only to proper motion)
            ec = ephem.Ecliptic(t2pulsar.prefit['ELONG'].val, t2pulsar.prefit['ELAT'].val)
            eq = ephem.Equatorial(ec, epoch=ephem.J2000)
            raj = np.float(eq.ra)
            decj = np.float(eq.dec)
        else:
            raise ValueError("Pulsar position not properly available")
        self.writeData(psrGroup, 'raj', np.float(raj), overwrite=overwrite)
        self.writeData(psrGroup, 'decj', np.float(decj), overwrite=overwrite)

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
    def readPulsar(self, psr, psrname, sorting='jitterext'):
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
        if self.hasField(psrname, 'raj'):
            psr.raj = np.float(self.getData(psrname, 'raj'))
        else:
            rajind = np.flatnonzero(np.array(psr.ptmdescription) == 'RAJ')
            psr.raj = np.array(self.getData(psrname, 'tmp_valpre'))[rajind]

        if self.hasField(psrname, 'decj'):
            psr.decj = np.float(self.getData(psrname, 'decj'))
        else:
            decjind = np.flatnonzero(np.array(psr.ptmdescription) == 'DECJ')
            psr.decj = np.array(self.getData(psrname, 'tmp_valpre'))[decjind]

        # Obtain residuals, TOAs, etc.
        psr.toas = np.array(self.getData(psrname, 'TOAs'))
        psr.toaerrs = np.array(self.getData(psrname, 'toaErr'))
        psr.prefitresiduals = np.array(self.getData(psrname, 'prefitRes'))
        psr.residuals = np.array(self.getData(psrname, 'postfitRes'))
        psr.detresiduals = np.array(self.getData(psrname, 'prefitRes'))
        psr.freqs = np.array(self.getData(psrname, 'freq'))
        psr.Mmat = np.array(self.getData(psrname, 'designmatrix'))

        # We do not read the (co)G-matrix anymore here. Happens when
        # initialising the model
