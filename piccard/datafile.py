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


class DataFile(object):
    """
    The DataFile class is the class that supports the HDF5 file format. All HDF5
    file interactions happen in this class.
    """
    filename = None
    h5file = None

    def __init__(self, filename=None):
        """
        Initialise the structure.

        @param filename:    name of the HDF5 file
        """
        # Open the hdf5 file?
        self.filename = filename

    def __del__(self):
        # Delete the instance, and close the hdf5 file?
        pass

    def getPulsarList(self):
        """
        Return a list of pulsars present in the HDF5 file
        """
        # 'r' means: read file, must exist
        self.h5file = h5.File(self.filename, 'r')
        psrlist = list(self.h5file)
        self.h5file.close()

        return psrlist

    def getPulsarGroup(self, psrname, delete=False):
        """
        Obtain the hdf5 group of pulsar psrname, create it if it does not exist. If
        delete is toggled, delete the content first. This function assumes the hdf5
        file is opened (not checked)

        @param psrname: The name of the pulsar
        @param delete:  If True, the pulsar group will be emptied before use
        """
        # datagroup = h5file.require_group('Data')

        if psrname in self.h5file and delete:
            del self.h5file[psrname]

        pulsarGroup = self.h5file.require_group(psrname)

        return pulsarGroup

    def addData(self, psrname, field, data, overwrite=True):
        """
        Add data to a specific pulsar. Here the hdf5 file is opened, and the right
        group is selected

        @param psrname:     The name of the pulsar
        @param field:       The name of the field we will be writing to
        @param data:        The data we are writing to the field
        @param overwrite:   Whether the data should be overwritten if it exists
        """
        if self.filename is None:
            raise RuntimeError, "HDF5 filename not provided"

        # 'a' means: read/write if exists, create otherwise
        self.h5file = h5.File(self.filename, 'a')

        psrGroup = self.getPulsarGroup(psrname, delete=False)
        self.writeData(psrGroup, field, data, overwrite=overwrite)

        self.h5file.close()
        self.h5file = None

        

    def getData(self, psrname, field, subgroup=None, \
            dontread=False, required=True, isort=None):
        """
        Read data from a specific pulsar. If the data is not available, the hdf5
        file is properly closed, and an exception is thrown

        @param psrname:     Name of the pulsar we are reading data from
        @param field:       Field name of the data we are requestion
        @param subgroup:    If the data is in a subgroup, get it from there
        @param dontread:    If set to true, do not actually read anything
        @param required:    If not required, do not throw an exception, but return
                            'None'
        @param isort:       If not None, use this as a slice when reading
        """
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

        if isort is not None:
            data = data[isort]

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

    def getShape(self, psrname, field, subgroup=None):
        """
        Retrieve the shape of a specific dataset

        @param psrname:     Name of the pulsar we are reading data from
        @param field:       Field name of the data we are requestion
        @param subgroup:    If the data is in a subgroup, get it from there
        """
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


    def writeData(self, dataGroup, field, data, overwrite=True):
        """
        (Over)write a field of data for a specific pulsar/group. Data group is
        required, instead of a name.

        @param dataGroup:   Group object
        @param field:       Name of field that we are writing to
        @param data:        The data that needs to be written
        @param overwrite:   If True, data will be overwritten (default True)
        """
        if field in dataGroup and overwrite:
            del dataGroup[field]

        if not field in dataGroup:
            try:
                dataGroup.create_dataset(field, data=data)
            except ValueError:
                print("WARNING: h5py too old to support empty arrays: {0}".
                        format(field))

    def get_used_t2pars(sefl, t2pulsar, inc_inactive=False):
        """For a libstempo2 object t2pulsar, return the parameters that 'act'"""
        actpars = []

        M = t2pulsar.designmatrix(fixunits=True, fixsigns=True, incoffset=False)
        for ii, par in enumerate(t2pulsar.pars(which='fit')):
            if np.any(M[:,ii]) or inc_inactive:
                actpars.append(par)

        return actpars

    def get_designmatrix(self, t2pulsar, pars):
        """For the parameters 'pars', retrieve the design matrix"""
        M = t2pulsar.designmatrix(fixunits=True, fixsigns=True, incoffset=True)
        fitpars = t2pulsar.pars(which='fit')

        msk = np.zeros(M.shape[1], dtype=np.bool)
        for par in pars:
            if par == 'Offset':
                ind = 0
            else:
                ind = fitpars.index(par)+1

            msk[ind] = True

        return M[:,msk]

    def addTempoPulsar(self, parfile, timfile, iterations=1, mode='replace', \
            dofit = False, maxobs=20000, inc_inactive=False):
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
        @param dofit:       Whether or not to do a fit at first (default: False)
        @param maxobs:      Maximum number of observations (if None, use standard)
        @param inc_inactive:Strip parameters that do not have any effect.
        """
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
                dofit=dofit, maxobs=maxobs)

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
        self.writeData(psrGroup, 'prefitRes', np.double(t2pulsar.residuals()), overwrite=overwrite)  # Seconds
        self.writeData(psrGroup, 'postfitRes', np.double(t2pulsar.residuals()), overwrite=overwrite)  # Seconds
        self.writeData(psrGroup, 'toaErr', np.double(1e-6*t2pulsar.toaerrs), overwrite=overwrite)    # Seconds
        self.writeData(psrGroup, 'freq', np.double(t2pulsar.ssbfreqs()), overwrite=overwrite)    # MHz

        # Write the position of the pulsar, even if it is in ecliptic
        # coordinates
        if 'RAJ' in t2pulsar and 'DECJ' in t2pulsar:
            raj = t2pulsar['RAJ'].val
            decj = t2pulsar['DECJ'].val
        elif 'ELONG' in t2pulsar and 'ELAT' in t2pulsar:
            if ephem is None:
                raise ImportError("pyephem not installed")

            # tempo/tempo2 RAJ/DECJ always refer precession/nutation-wise to
            # epoch J2000. Posepoch does not apply here (only to proper motion)
            ec = ephem.Ecliptic(t2pulsar['ELONG'].val, t2pulsar['ELAT'].val)
            eq = ephem.Equatorial(ec, epoch=ephem.J2000)
            raj = np.float(eq.ra)
            decj = np.float(eq.dec)
        else:
            raise ValueError("Pulsar position not properly available")
        self.writeData(psrGroup, 'raj', np.float(raj), overwrite=overwrite)
        self.writeData(psrGroup, 'decj', np.float(decj), overwrite=overwrite)
        self.writeData(psrGroup, 'f0', np.float(t2pulsar['F0'].val), overwrite=overwrite)

        # Now obtain and write the timing model parameters
        actpars = self.get_used_t2pars(t2pulsar, inc_inactive)
        tmpname = ['Offset'] + actpars #+ list(t2pulsar.pars(which='fit'))
        tmpvalpre = np.zeros(len(tmpname))
        tmpvalpost = np.zeros(len(tmpname))
        tmperrpre = np.zeros(len(tmpname))
        tmperrpost = np.zeros(len(tmpname))
        #for i in range(len(t2pulsar.pars(which='fit'))):
        for i in range(len(actpars)):
            tmpvalpre[i+1] = t2pulsar[tmpname[i+1]].val
            tmpvalpost[i+1] = t2pulsar[tmpname[i+1]].val
            tmperrpre[i+1] = t2pulsar[tmpname[i+1]].err
            tmperrpost[i+1] = t2pulsar[tmpname[i+1]].err

        self.writeData(psrGroup, 'tmp_name', tmpname, overwrite=overwrite)          # TMP name
        self.writeData(psrGroup, 'tmp_valpre', tmpvalpre, overwrite=overwrite)      # TMP pre-fit value
        self.writeData(psrGroup, 'tmp_valpost', tmpvalpost, overwrite=overwrite)    # TMP post-fit value
        self.writeData(psrGroup, 'tmp_errpre', tmperrpre, overwrite=overwrite)      # TMP pre-fit error
        self.writeData(psrGroup, 'tmp_errpost', tmperrpost, overwrite=overwrite)    # TMP post-fit error

        # TODO: writing the design matrix should be done irrespective of the fitting flag
        #desmat = t2pulsar.designmatrix(fixunits=True, fixsigns=True, incoffset=True)
        desmat = self.get_designmatrix(t2pulsar, tmpname)
        self.writeData(psrGroup, 'designmatrix', desmat, overwrite=overwrite)

        # Get the flag group for this pulsar. Create if not there
        flagGroup = psrGroup.require_group('Flags')

        # Obtain the unique flags in this dataset, and write to file
        #uflags = list(set(t2pulsar.flags))
        uflags = t2pulsar.flags()
        for flagid in uflags:
            #self.writeData(flagGroup, flagid, t2pulsar.flags[flagid], overwrite=overwrite)
            self.writeData(flagGroup, flagid, t2pulsar.flagvals(flagid), overwrite=overwrite)

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

        # Delete the libstempo object
        del t2pulsar

        # Close the HDF5 file
        self.h5file.close()
        self.h5file = None

    def addMetronome(self, name, F0, toafile, mode='replace'):
        """
        Add a metronome to the HDF5 file, given a file with TOAs. It is treated
        as a pulsar henceforth, but it has a simpler timing model.  No extra
        model matrices and auxiliary variables are added to the HDF5 file.

        @param name:        Name of this metronome
        @param F0:          Initially determined pulse period
        @param toafile:     Name of file with TOAs
        @param mode:        Can be replace/overwrite/new. Replace first deletes the
                            entire pulsar group. Overwrite overwrites all data, but
                            does not delete the auxiliary fields. New requires the
                            pulsar not to exist, and throws an exception otherwise.
        """
        # Check whether the two files exist
        if not os.path.isfile(toafile):
            raise IOError, "Cannot find toafile (%s)!" % (toafile)

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

        # Load the TOA data, and save it in the HDF5 file
        mdata = np.loadtxt(toafile)
        with open(toafile, 'r') as content_file:
            timfile_content = content_file.read()
        parfile_content = "METRONOME-{0}".format(name)

        # Get the pulsar group
        psrGroup = self.getPulsarGroup(name, delete=deletepsr)

        # Save the par-file and the tim-file to the HDF5 file
        self.writeData(psrGroup, 'parfile', parfile_content, overwrite=overwrite)
        self.writeData(psrGroup, 'timfile', timfile_content, overwrite=overwrite)

        # Set the raw data
        toas = mdata[:,0]
        toaerrs = mdata[:,1]
        freqs = np.ones_like(toas) * 1000       # 1 kHz per default
        raj = 1.0
        decj = 1.0

        # Determine the pulse number of the first observation
        pn_0 = int(toas[0] * F0 + 0.5)
        rphase_0 = toas[0] - pn_0 / F0      # Residual phase first obs
        
        # Expected TOAs, parameters, and residuals now are:
        toas_exp = rphase_0 + np.arange(pn_0, pn_0+len(toas)) / F0
        residuals_0 = toas - toas_exp
        pars_0 = np.array([rphase_0, F0])

        # Construct the design matrix
        M = np.zeros((len(toas), 2))
        M[:,0] = 1.0
        M[:,1] = -toas / F0

        # Perform a linear least-squares fit
        # deltapars = (M.T N^{-1} M)^{-1} M.T N^{-1} residuals
        Nvec = toaerrs**2
        MNM = np.dot(M.T / Nvec, M)
        cf = sl.cho_factor(MNM)
        dpars = sl.cho_solve(cf, np.dot(M.T, residuals_0 / Nvec))
        Sigma = sl.cho_sovle(cf, np.eye(len(MNM)))

        # Update the parameters and the residuals
        pars = pars_0 + dpars
        parerrs = np.sqrt(np.diag(Sigma))
        residuals = residuals_0 - np.dot(M, dpars)

        self.writeData(psrGroup, 'TOAs', toas, overwrite=overwrite)    # Seconds
        self.writeData(psrGroup, 'prefitRes', residuals_0, overwrite=overwrite)  # Seconds
        self.writeData(psrGroup, 'postfitRes', residuals, overwrite=overwrite)  # Seconds
        self.writeData(psrGroup, 'toaErr', toaerrs, overwrite=overwrite)    # Seconds
        self.writeData(psrGroup, 'freq', freqs, overwrite=overwrite)    # MHz
        self.writeData(psrGroup, 'designmatrix', M, overwrite=overwrite)
        self.writeData(psrGroup, 'raj', raj, overwrite=overwrite)
        self.writeData(psrGroup, 'decj', decj, overwrite=overwrite)
        self.writeData(psrGroup, 'f0', F0, overwrite=overwrite)

        # Now obtain and write the timing model parameters
        tmpname = ['Offset', 'F0']
        tmpvalpre = pars_0
        tmpvalpost = pars
        tmperrpre = parerrs
        tmperrpost = parerrs

        self.writeData(psrGroup, 'tmp_name', tmpname, overwrite=overwrite)          # TMP name
        self.writeData(psrGroup, 'tmp_valpre', tmpvalpre, overwrite=overwrite)      # TMP pre-fit value
        self.writeData(psrGroup, 'tmp_valpost', tmpvalpost, overwrite=overwrite)    # TMP post-fit value
        self.writeData(psrGroup, 'tmp_errpre', tmperrpre, overwrite=overwrite)      # TMP pre-fit error
        self.writeData(psrGroup, 'tmp_errpost', tmperrpost, overwrite=overwrite)    # TMP post-fit error

        # Get the flag group for this pulsar. Create if not there
        flagGroup = psrGroup.require_group('Flags')

        # Obtain the unique flags in this dataset, and write to file
        uflags = ['telID']
        for flagid in uflags:
            flagtel = ['metronome'] * len(toas)
            self.writeData(flagGroup, flagid, np.array(flagtel), overwrite=overwrite)

        if not "efacequad" in flagGroup:
            # Check if the sys-flag is present in this set. If it is, add an
            # efacequad flag with pulsarname+content of the sys-flag. If it
            # isn't, check for a be-flag and try the same. Otherwise, add an
            # efacequad flag with the pulsar name as it's elements.
            efacequad = []
            nobs = len(toas)
            pulsarname = map(str, [name] * nobs)

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
            nobs = len(toas)
            pulsarname = map(str, [name] * nobs)
            self.writeData(flagGroup, "pulsarname", pulsarname, overwrite=overwrite)

        # Close the HDF5 file
        self.h5file.close()
        self.h5file = None


    def addH5Pulsar(self, h5file, pulsars='all', mode='add'):
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

    def readPulsar(self, psr, psrname):
        """
        Read the basic quantities of a pulsar from an HDF5 file into a ptaPulsar
        object. No extra model matrices and auxiliary variables are read from the
        HDF5 file, not even residuals. If any field is not present in the HDF5 file,
        an IOError exception is raised

        @param psr:     The ptaPulsar object we need to fill with data
        @param psrname: The name of the pulsar to be read from the HDF5 file

        TODO: The HDF5 file is opened and closed every call of 'getData'. That seems
              kind of inefficient
        TODO: This function should not 'know' about the psr object. Reading/writing
              of these quantities should take place outside of this
        """
        print("WARNING: readPulsar has been deprecated!")
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
        psr.P0 = 1.0 / np.float(self.getData(psrname, 'f0'))

        # We do not read the (co)G-matrix anymore here. Happens when
        # initialising the model
