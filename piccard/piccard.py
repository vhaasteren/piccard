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
import healpy as hp

from constants import *
from datafile import *
from signals import *
from seplik import *
from ptafuncs import *
from jitterext import *
#from . import anisotropygammas as ang  # Internal module

from AnisCoefficients_pix import CorrBasis

from triplot import *
from acf import *

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

class ptaPulsar(object):
    """
    Pulsar class. Keeps track of all the per-pulsar quantities, allows for
    reading/writing HDF5 files, and creates auxiliary quantities.
    """

    def __init__(self):
        self.parfile_content = None      # The actual content of the original par-file
        self.timfile_content = None      # The actual content of the original tim-file
        self.t2psr = None

        self.raj = 0
        self.decj = 0
        self.toas = None
        self.toaerrs = None
        self.prefitresiduals = None
        self.residuals = None
        self.detresiduals = None        # Residuals after subtraction of deterministic sources
        self.gibbsresiduals = None      # Residuals used in Gibbs sampling
                                        # NOTE: Replace with more specific pars
        self.gibbscoefficients = None   # Coefficients used in Gibbs sampling
                                        # NOTE: Replace with more specific pars
        self.freqs = None
        self.Gmat = None
        self.Gcmat = None
        self.Mmat = None
        self.ptmpars = []
        self.ptmparerrs = []
        self.ptmdescription = []
        self.flags = None
        self.name = "J0000+0000"
        self.Tmax = None

        self.isort = slice(None, None, None)        # Slice map: hdf5file -> psr
        self.iisort = slice(None, None, None)       # Slice map: psr -> hdf5file

        # Design matrix indices for different Gibbs conditionals
        self.Mmat_g = None
        self.Mmask_F = None
        self.Mmask_D = None
        self.Mmask_U = None

        # Temporary quantities for various Gibbs quantities
        self.gibbs_N_iter = 0
        self.gibbs_N_residuals = None
        self.gibbs_N_sinds = []
        self.gibbs_NJ_sinds = []

        # The auxiliary quantities
        self.Fmat = None
        self.SFmat = None            # Fmatrix for the frequency lines
        self.FFmat = None            # Total of Fmat and SFmat
        self.Fdmmat = None
        self.Hmat = None             # The compression matrix
        self.Homat = None            # The orthogonal-to compression matrix
        self.Hcmat = None            # The co-compression matrix
        self.Hocmat = None           # The orthogonal co-compression matrix
        self.Umat = None            # Quantization matrix
        self.Uimat = None           # Pseudo-inverse of Umat
        self.Uinds = None           # Indices replacing Umat (when sorted)
        self.avetoas = None         # Epoch-averaged TOA epoch
        self.SFdmmat = None         # Fdmmatrix for the dm frequency lines
        self.Dvec = None
        self.DF = None
        self.DSF = None
        self.DFF = None              # Total of DF and DSF
        self.Ffreqs = None          # Frequencies of the red noise
        self.SFfreqs = None      # Single modelled frequencies
        self.SFdmfreqs = None
        self.frequencyLinesAdded = 0      # Whether we have > 1 single frequency line
        self.dmfrequencyLinesAdded = 0      # Whether we have > 1 single frequency line
        self.Fdmfreqs = None
        self.Emat = None
        self.EEmat = None
        self.Gr = None
        self.GGr = None
        self.GtF = None
        self.GtD = None
        self.FBmats = np.array([])

        self.Zmat = None         # For the Gibbs sampling, this is the Fmat/Emat
        self.Zmat_M = None       # For the Gibbs sampling, this is the Fmat/Emat
        self.Zmat_F = None       # For the Gibbs sampling, this is the Fmat/Emat
        self.Zmat_D = None
        self.Zmat_U = None
        self.Zmask_F = None      # Which columns of the full Zmat are in Zmat_F
        self.Zmask_D = None
        self.Zmask_U = None
        self.Zmask_M = None
        self.Gr = None
        self.GGr = None
        self.GtF = None
        self.GtD = None
        self.GtU = None
        self.AGr = None      # Replaces GGr in 2-component noise model
        self.AoGr = None     #   Same but for orthogonal basis (when compressing)
        self.AGF = None      # Replaces GGtF in 2-component noise model
        self.AGD = None      # Replaces GGtD in 2-component noise model
        self.AGE = None      # Replaces GGtE in 2-component noise model
        self.AGU = None      # Replace GGtU in 2-component noise model

        # Auxiliaries used in the likelihood
        self.twoComponentNoise = False       # Whether we use the 2-component noise model
        self.Nvec = None             # The total white noise (eq^2 + ef^2*err)
        self.Wvec = None             # The weights in 2-component noise
        self.Wovec = None            # The weights in 2-component orthogonal noise
        self.Nwvec = None            # Total noise in 2-component basis (eq^2 + ef^2*Wvec)
        self.Nwovec = None           # Total noise in 2-component orthogonal basis
        self.Jweight = None          # The weight of the jitter noise in compressed basis
        self.Jvec = None

        # To select the number of Frequency modes
        self.bfinc = None        # Number of modes of all internal matrices
        self.bfdminc = None      # Number of modes of all internal matrices (DM)
        self.bprevfinc = None     # Current number of modes in RJMCMC
        self.bprevfdminc = None   # Current number of modes in RJMCMC

        # Indices for when we are in mark11
        self.fourierind = None
        self.dmfourierind = None
        self.jitterfourierind = None

    def readFromH5_old(self, h5df, psrname):
        h5df.readPulsar_old(self, psrname)

    def readFromH5(self, h5df, psrname, sort='jitterext'):
        """
        Read the pulsar data (TOAs, residuals, design matrix, etc..) from an
        HDF5 file

        @param h5df:        The DataFile object we are reading from
        @param psrname:     Name of the Pulsar to be read from the HDF5 file
        @param sort:        In which way do we want to sort the TOAs
                            (None, 'time', 'jitterext')
        """
        self.name = psrname

        # Read the content of the par/tim files in a string
        self.parfile_content = str(h5df.getData(psrname, 'parfile', required=False))
        self.timfile_content = str(h5df.getData(psrname, 'timfile', required=False))

        # Read the timing model parameter descriptions
        self.ptmdescription = map(str, h5df.getData(psrname, 'tmp_name'))
        self.ptmpars = np.array(h5df.getData(psrname, 'tmp_valpre'))
        self.ptmparerrs = np.array(h5df.getData(psrname, 'tmp_errpre'))

        # Read the position of the pulsar
        if h5df.hasField(psrname, 'raj'):
            self.raj = np.float(h5df.getData(psrname, 'raj'))
        else:
            rajind = np.flatnonzero(np.array(self.ptmdescription) == 'RAJ')
            self.raj = np.array(h5df.getData(psrname, 'tmp_valpre'))[rajind]

        if h5df.hasField(psrname, 'decj'):
            self.decj = np.float(h5df.getData(psrname, 'decj'))
        else:
            decjind = np.flatnonzero(np.array(self.ptmdescription) == 'DECJ')
            self.decj = np.array(h5df.getData(psrname, 'tmp_valpre'))[decjind]

        # Read the flags, and the residuals, and determine the sorting/slicing
        toas = np.array(h5df.getData(psrname, 'TOAs'))
        flags = np.array(map(str, h5df.getData(psrname, 'efacequad', 'Flags')))

        self.isort, self.iisort = argsortTOAs(toas, flags, which=sort)

        self.toas = np.array(h5df.getData(psrname, 'TOAs'))[self.isort]
        self.flags = np.array(map(str, h5df.getData(psrname, 'efacequad', 'Flags')))[self.isort]

        self.toaerrs = np.array(h5df.getData(psrname, 'toaErr'))[self.isort]
        self.prefitresiduals = np.array(h5df.getData(psrname, 'prefitRes'))[self.isort]
        self.residuals = np.array(h5df.getData(psrname, 'postfitRes'))[self.isort]
        self.detresiduals = np.array(h5df.getData(psrname, 'prefitRes'))[self.isort]
        self.freqs = np.array(h5df.getData(psrname, 'freq'))[self.isort]
        self.Mmat = np.array(h5df.getData(psrname, 'designmatrix'))[self.isort,:]

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

            # Construct the G-matrices
            U, s, Vh = sl.svd(newM)
            newG = U[:, (newM.shape[1]):].copy()
            newGc = U[:, :(newM.shape[1])].copy()
        else:
            newM = oldMmat.copy()
            newptmdescription = np.array(oldptmdescription)
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
            newptmpars = oldptmpars[indkeep]

            # Construct the G-matrices
            U, s, Vh = sl.svd(newM)
            newG = U[:, (newM.shape[1]):].copy()
            newGc = U[:, :(newM.shape[1])].copy()
        else:
            newM = oldMmat.copy()
            newptmdescription = np.array(oldptmdescription)
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
            elif tmpar == 'Offset' and keep:
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
    @param complement:  Evaluate the compression complement
    """
    # TODO: selection of timing-model parameters should apply to _all_ forms of
    # compression. Still possible to do frequencies and include timing model
    # parameters, as long as we include the complement function
    def constructCompressionMatrix(self, compression='None', \
            nfmodes=-1, ndmodes=-1, likfunc='mark4', threshold=1.0, \
            tmpars = None, complement=False):
        if compression == 'average':
            # To be sure, just construct the averages again. But is already done
            # in 'createPulsarAuxiliaries'
            if likfunc[:5] != 'mark4':
                (self.avetoas, self.Umat) = quantize_fast(self.toas)
                print("WARNING: ignoring per-backend epoch averaging in compression")
                Wjit = np.sum(self.Umat, axis=0)
                self.Jweight = np.sum(Wjit * self.Umat, axis=1)

            #"""
            GU = np.dot(self.Gmat.T, self.Umat)
            #GUUG = np.dot(GU, GU.T)
            #"""

            # Construct an orthogonal basis, and singular values
            #svech, Vmath = sl.eigh(GUUG)
            #Vmat, svec, Vhsvd = sl.svd(GUUG)
            Vmat, svec, Vhsvd = sl.svd(GU, full_matrices=not complement)

            # Decide how many basis vectors we'll take. (Would be odd if this is
            # not the number of columns of self.U. How to test? For now, use
            # 99.9% of rms power
            cumrms = np.cumsum(svec**2)
            totrms = np.sum(svec**2)
            #print "svec:   ", svec**2
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
            H = np.dot(self.Gmat, Bmat)

            if complement:
                Bomat = Vmat[:, l:].copy()
                Ho = np.dot(self.Gmat, Bomat)

            # Use another SVD to construct not only Hmat, but also Hcmat
            # We use this version of Hmat, and not H from above, in case of
            # linear dependences...
            #svec, Vmat = sl.eigh(H)
            Vmat, s, Vh = sl.svd(H, full_matrices=not complement)
            self.Hmat = Vmat[:, :l]
            self.Hcmat = Vmat[:, l:]

            # For compression-complements, construct Ho and Hoc
            if complement > 0:
                Vmat, s, Vh = sl.svd(Ho)
                self.Homat = Vmat[:, :Ho.shape[1]]
                self.Hocmat = Vmat[:, Ho.shape[1]:]
            else:
                self.Homat = np.zeros((Vmat.shape[0], 0))
                self.Hocmat = np.eye(Vmat.shape[0])

        elif compression == 'frequencies':
            # Use a power-law spectrum with spectral-index of 4.33
            #freqpy = self.Ffreqs * pic_spy
            #phivec = (pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-4.33)
            #phivec = (pic_spy**3 / (12*np.pi*np.pi * self.Tmax)) * freqpy ** (-1.00)
            #GF = np.dot(self.Gmat.T, self.Fmat * phivec)

            GF = np.dot(self.Gmat.T, self.Fmat)
            #GFFG = np.dot(GF, GF.T)
            #Vmat, svec, Vhsvd = sl.svd(GFFG)
            Vmat, svec, Vhsvd = sl.svd(GF, full_matrices=not complement)

            cumrms = np.cumsum(svec**2)
            totrms = np.sum(svec**2)
            # print "Freqs: ", cumrms / totrms
            l = np.flatnonzero( (cumrms/totrms) >= threshold )[0] + 1

            # Use the number of frequencies, instead of a threshold now:
            l = self.Fmat.shape[1]

            print("Using {0} components for pulsar {1}".format(\
                    l, self.name))

            # H is the compression matrix
            Bmat = Vmat[:, :l].copy()
            H = np.dot(self.Gmat, Bmat)

            if complement:
                Bomat = Vmat[:, l:].copy()
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

    def gibbs_set_design(self, gibbsmodel):
        """
        We construct the orthogonalized versions of the design matrix here. Two
        things are done:
        1) The design matrix is split up into 'conditional subsections'.
        Basically, for every conditional probability in the collapsed Gibbs
        sampler, we include a sub-set of design matrix parameters to
        analytically marginalize over. These parameters are different for every
        conditional probability (and the sub-blocks are non-overlapping)
        2) Every sub-block is orthogonalized with an SVD; this is the reason why
        they are not allowed to overlap. The orthogonalization is necessary for
        the numerical stability of the linear algebra.

        @param gibbsmodel:  For which conditional we are constructing the
                            matricees
        """
        # The parameters that need to be included in the various conditionals
        F_list = ['Offset', \
                'LAMBDA', 'BETA', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', \
                'ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'TASC', 'EPS1', 'EPS2', \
                'XDOT', 'PBDOT', 'KOM', 'KIN', 'EDOT', 'MTOT', 'SHAPMAX', \
                'GAMMA', 'X2DOT', 'XPBDOT', 'E2DOT', 'OM_1', 'A1_1', 'XOMDOT', \
                'PMLAMBDA', 'PMBETA', 'PX', 'PB', 'A1', 'E', 'ECC', \
                'T0', 'OM', 'OMDOT', 'SINI', 'A1', 'M2']
        F_front_list = ['JUMP', 'F']
        D_list = ['DM', 'DM1', 'DM2', 'DM3', 'DM4']
        U_list = []     # U_list needs to stay empty, otherwise 'joinNJ' in
                        # Gibbs mark2 will not work anymore -- RvH

        #isolated_list = ['Offset', 'F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', \
        #            'PX', 'DM', 'DM1', 'DM2']

        # Make the conditional lists
        self.Mmask_F = np.array([0]*len(self.ptmdescription), dtype=np.bool)
        self.Mmask_D = np.array([0]*len(self.ptmdescription), dtype=np.bool)
        self.Mmask_U = np.array([0]*len(self.ptmdescription), dtype=np.bool)
        self.Mmat_g = np.zeros(self.Mmat.shape)
        for ii, par in enumerate(self.ptmdescription):
            incrn = False
            for par_front in F_front_list:
                if par[:len(par_front)] == par_front:
                    incrn = True

            if (par in F_list or incrn) and 'rednoise' in gibbsmodel:
                self.Mmask_F[ii] = True

            if par in D_list and 'dm' in gibbsmodel:
                self.Mmask_D[ii] = True

            if par in U_list and 'jitter' in gibbsmodel:
                self.Mmask_U[ii] = True

        # Make sure these guys do not overlap
        if np.sum(np.logical_and(self.Mmask_F, self.Mmask_D)) > 0 or \
                np.sum(np.logical_and(self.Mmask_F, self.Mmask_U)) > 0 or \
                np.sum(np.logical_and(self.Mmask_D, self.Mmask_U)) > 0:
            raise ValueError("Conditional lists cannot overlap")

        # Create left-over list
        #self.Mmask_M = np.logical_not(np.logical_or(\
        #        np.logical_or(self.Mmask_F, self.Mmask_D),\
        #        self.Mmask_U))
        self.Mmask_M = np.array([1]*self.Mmat_g.shape[1], dtype=np.bool)
        if 'rednoise' in gibbsmodel:
            self.Mmask_M = np.logical_and(self.Mmask_M, \
                    np.logical_not(self.Mmask_F))
        if 'dm' in gibbsmodel:
            self.Mmask_M = np.logical_and(self.Mmask_M, \
                    np.logical_not(self.Mmask_D))
        if 'jitter' in gibbsmodel:
            self.Mmask_M = np.logical_and(self.Mmask_M, \
                    np.logical_not(self.Mmask_U))

        # Create orthogonals for all of these
        if np.sum(self.Mmask_F) > 0:
            U, s, Vt = sl.svd(self.Mmat[:,self.Mmask_F], full_matrices=False)
            self.Mmat_g[:, self.Mmask_F] = U

        if np.sum(self.Mmask_D) > 0:
            U, s, Vt = sl.svd(self.Mmat[:,self.Mmask_D], full_matrices=False)
            self.Mmat_g[:, self.Mmask_D] = U

        if np.sum(self.Mmask_U) > 0:
            U, s, Vt = sl.svd(self.Mmat[:,self.Mmask_U], full_matrices=False)
            self.Mmat_g[:, self.Mmask_U] = U

        if np.sum(self.Mmask_M) > 0:
            U, s, Vt = sl.svd(self.Mmat[:,self.Mmask_M], full_matrices=False)
            self.Mmat_g[:, self.Mmask_M] = U

        #U, s, Vh = sl.svd(self.Mmat)
        #self.Gmat = U[:, self.Mmat.shape[1]:]
        #self.Gcmat = U[:, :self.Mmat.shape[1]]
        #self.Mmat_g = self.Gcmat.copy()
        #self.Mmat_g = self.Mmat / np.mean(self.Mmat, axis=0) 

    def getMmask(self, which='all'):
        """
        Given a 'which', return the right design matrix mask

        @param which:   Can be all, F, D, U

        @return:    Which columns of the design matrix are included in the
                    conditional that marginalises over the partial design matrix
        """
        if which in ['F', 'B']:
            mask = self.Mmask_F
        elif which == 'D':
            mask = self.Mmask_D
        elif which == 'U':
            mask = self.Mmask_U
        elif which == 'M':
            mask = self.Mmask_M
        elif which in ['all', 'N']:
            mask = np.array([1]*self.Mmat_g.shape[1], dtype=np.bool)
        else:
            mask = np.array([1]*self.Mmat_g.shape[1], dtype=np.bool)
        
        return mask


    def getZmat(self, gibbsmodel, which='all'):
        """
        Return Zmat, given the gibbsmodel, and which sources to include

        @param gibbsmodel:  List of which quadratics are available
        @param which:       Which quadratics to vary (not fixed)
                            Options: all, F, B, D, U, N (N means all but U)

        @return:    The partial Z-matrix, and it's mask compared to the full
                    Z-matrix
        """
        nf = self.Fmat.shape[1]
        ndmf = self.Fdmmat.shape[1]
        zmask = np.zeros(0, dtype=np.bool)

        # dmask is the mask for the design matrix part (in case we have
        # non-linear analysis, or Gibbs sampling with specialized conditionals)
        dmask = self.getMmask(which=which)

        if which in ['F', 'B']:
            dmask = self.Mmask_F
            #dmask = self.getMmask(which='M')

        if 'design' in gibbsmodel:
            Ft_1 = self.Mmat_g[:, dmask]
        else:
            Ft_1 = np.zeros((self.Mmat_g.shape[0], 0))

        zmask = np.append(zmask, dmask)

        if nf > 0 and 'rednoise' in gibbsmodel and \
                (which in ['all', 'F', 'N']):
            Ft_2 = np.append(Ft_1, self.Fmat, axis=1)

            zmask = np.append(zmask, \
                    np.array([1]*self.Fmat.shape[1], dtype=np.bool))
        else:
            Ft_2 = Ft_1

            if nf > 0 and 'rednoise' in gibbsmodel:
                zmask = np.append(zmask, \
                        np.array([0]*self.Fmat.shape[1], dtype=np.bool))

        if nf > 0 and 'freqrednoise' in gibbsmodel and \
                (which in ['all', 'B', 'N']):
            for FBmat in self.FBmats:
                Ft_2 = np.append(Ft_2, FBmat, axis=1)
                zmask = np.append(zmask, \
                        np.array([1]*FBmat.shape[1], dtype=np.bool))
        elif nf > 0 and 'freqrednoise' in gibbsmodel:
            for FBmat in self.FBmats:
                zmask = np.append(zmask, \
                        np.array([0]*FBmat.shape[1], dtype=np.bool))

        if ndmf > 0 and 'dm' in gibbsmodel and \
            (which in ['all', 'D', 'N']):
            Ft_3 = np.append(Ft_2, self.DF, axis=1)

            zmask = np.append(zmask, \
                    np.array([1]*self.DF.shape[1], dtype=np.bool))
        else:
            Ft_3 = Ft_2

            if ndmf > 0 and 'dm' in gibbsmodel:
                zmask = np.append(zmask, \
                        np.array([0]*self.DF.shape[1], dtype=np.bool))

        if 'jitter' in gibbsmodel and \
            (which in ['all', 'U']):
            Ft_4 = np.append(Ft_3, self.Umat, axis=1)

            zmask = np.append(zmask, \
                    np.array([1]*self.Umat.shape[1], dtype=np.bool))
        else:
            Ft_4 = Ft_3

            if 'jitter' in gibbsmodel:
                zmask = np.append(zmask, \
                        np.array([0]*self.Umat.shape[1], dtype=np.bool))

        if 'correx' in gibbsmodel and \
            (which in ['all', 'F', 'N']):
            Zmat = np.append(Ft_4, self.Fmat, axis=1)

            zmask = np.append(zmask, \
                    np.array([1]*self.Fmat.shape[1], dtype=np.bool))
        else:
            Zmat = Ft_4

            if 'correx' in gibbsmodel:
                zmask = np.append(zmask, \
                        np.array([0]*self.Fmat.shape[1], dtype=np.bool))

        return Zmat, zmask

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
    @param noGmatWrite:     Whether or not to save the G-matrix to file
    @param threshold:       Threshold for compression precision
    @param gibbsmodel:      What coefficients to include in the Gibbs model
    @param trimquant:       Whether to trim the quantization matrix
    @param bandRedNoise:    Frequency bands for band-limited red noise
    @param complement:  Evaluate the compression complement

    """
    def createPulsarAuxiliaries(self, h5df, Tmax, nfreqs, ndmfreqs, \
            twoComponent=False, nSingleFreqs=0, nSingleDMFreqs=0, \
            compression='None', likfunc='mark3', write='likfunc', \
            tmsigpars=None, noGmatWrite=False, threshold=1.0, \
            gibbsmodel=[], trimquant=False, bandRedNoise=[], \
            complement=False):
        # For creating the auxiliaries it does not really matter: we are now
        # creating all quantities per default
        # TODO: set this parameter in another place?
        if twoComponent and not likfunc in ['mark11', 'mark12', 'gibbs']:
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
            self.Dvec = pic_DMk / (self.freqs**2)
            self.DF = (self.Dvec * self.Fdmmat.T).T
        else:
            self.Fdmmat = np.zeros((len(self.toas),0))
            self.Fdmfreqs = np.zeros(0)
            self.Dvec = pic_DMk / (self.freqs**2)
            self.DF = np.zeros((len(self.freqs), 0))

        # Create the daily averaged residuals
        if trimquant:
            (self.avetoas, self.Umat, self.Uimat) = \
                    quantize_split(self.toas, self.flags, calci=True)
            self.Umat, self.Uimat, self.avetoas, jflags = \
                    quantreduce(self.Umat, self.avetoas, self.flags, calci=True)
        else:
            (self.avetoas, self.Umat, self.Uimat) = \
                    quantize_fast(self.toas, calci=True)

        self.Uinds = quant2ind(self.Umat)
        Wjit = np.sum(self.Umat, axis=0)
        self.Jweight = np.sum(Wjit * self.Umat, axis=1)

        # Write these quantities to disk
        if write != 'no':
            h5df.addData(self.name, 'pic_Fmat', self.Fmat[self.iisort,:])
            h5df.addData(self.name, 'pic_Ffreqs', self.Ffreqs)
            h5df.addData(self.name, 'pic_Fdmmat', self.Fdmmat[self.iisort,:])
            h5df.addData(self.name, 'pic_Fdmfreqs', self.Fdmfreqs)
            h5df.addData(self.name, 'pic_Dvec', self.Dvec[self.iisort])
            h5df.addData(self.name, 'pic_DF', self.DF[self.iisort,:])

            h5df.addData(self.name, 'pic_avetoas', self.avetoas)
            h5df.addData(self.name, 'pic_Umat', self.Umat[self.iisort,:])
            h5df.addData(self.name, 'pic_Uimat', self.Uimat[:,self.iisort])
            h5df.addData(self.name, 'pic_Uinds', self.Uinds)
            h5df.addData(self.name, 'pic_Jweight', self.Jweight[self.iisort])

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
                    ndmodes=2*ndmf, threshold=threshold, tmpars=tmpars,
                    complement=complement)
            if write != 'no':
                h5df.addData(self.name, 'pic_Hcmat', self.Hcmat[self.iisort,:])
                h5df.addData(self.name, 'pic_Gcmat', self.Gcmat[self.iisort,:])

                if not noGmatWrite:
                    h5df.addData(self.name, 'pic_Gmat', self.Gmat[self.iisort,:])
                    h5df.addData(self.name, 'pic_Hmat', self.Hmat[self.iisort,:])
                    h5df.addData(self.name, 'pic_Homat', self.Homat[self.iisort,:])
                    h5df.addData(self.name, 'pic_Hocmat', self.Hocmat[self.iisort,:])



        # Now, write such quantities on a per-likelihood basis
        if likfunc == 'mark1' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            self.GtD = np.dot(self.Hmat.T, self.DF)
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
                if self.twoComponentNoise:
                    h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                    h5df.addData(self.name, 'pic_Amat', self.Amat)
                    h5df.addData(self.name, 'pic_AGr', self.AGr)
                    h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                    h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                    h5df.addData(self.name, 'pic_AoGr', self.AoGr)

        if likfunc in ['mark3', 'mark3fa', 'mark3nc'] or write == 'all':
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_UtF', self.UtF)
                h5df.addData(self.name, 'pic_UtD', self.UtD)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat[self.iisort, :])
                h5df.addData(self.name, 'pic_FFmat', self.FFmat[self.iisort, :])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                #h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat[self.iisort, :])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                #h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat[self.iisort, :])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat[self.iisort, :])
                h5df.addData(self.name, 'pic_FFmat', self.FFmat[self.iisort, :])
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
                h5df.addData(self.name, 'pic_GGr', self.GGr[self.iisort])
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                #h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat[self.iisort, :])
                #h5df.addData(self.name, 'pic_GGtE', self.GGtE)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat[self.iisort, :])
                h5df.addData(self.name, 'pic_SFdmmat', self.SFdmmat[self.iisort, :])
                h5df.addData(self.name, 'pic_FFmat', self.FFmat[self.iisort, :])
                h5df.addData(self.name, 'pic_SFfreqs', self.SFfreqs)
                h5df.addData(self.name, 'pic_DSF', self.DSF[self.iisort, :])
                h5df.addData(self.name, 'pic_DFF', self.DFF[self.iisort, :])
                h5df.addData(self.name, 'pic_EEmat', self.EEmat[self.iisort, :])
                h5df.addData(self.name, 'pic_GGtEE', self.GGtEE[self.iisort, :])

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

        if likfunc in ['mark12', 'gibbs'] or write == 'all':
            # Set the band-limited F-matrices
            self.FBmats = []
            self.Fbands = []
            for bb, band in enumerate(bandRedNoise):
                # Band-limited noise has zero response outside frequency band,
                # but is the same as Red Noise elsewhere
                FBmat = self.Fmat.copy()
                mask = np.logical_or(self.freqs < band[0], \
                        self.freqs > band[1])
                FBmat[mask, :] = 0.0
                self.FBmats.append(FBmat)
                self.Fbands.append(band)

            self.FBmats = np.array(self.FBmats)
            self.Fbands = bandRedNoise

            # Prepare the new design matrix bases
            self.gibbs_set_design(gibbsmodel)

            self.Zmat, self.Zmask = self.getZmat(gibbsmodel, which='all')
            #self.Zmask_M[self.Mmat_g.shape[1]:] = False
            self.Zmat_M, self.Zmask_M = self.getZmat(gibbsmodel, which='M')
            self.Zmat_F, self.Zmask_F = self.getZmat(gibbsmodel, which='F')
            self.Zmat_B, self.Zmask_B = self.getZmat(gibbsmodel, which='B')
            self.Zmat_D, self.Zmask_D = self.getZmat(gibbsmodel, which='D')
            self.Zmat_U, self.Zmask_U = self.getZmat(gibbsmodel, which='U')
            self.Zmat_N, self.Zmask_N = self.getZmat(gibbsmodel, which='N')
            self.gibbsresiduals = np.zeros(len(self.toas))


            self.gibbscoefficients = np.zeros(self.Zmat.shape[1])

            self.Wvec = np.zeros(self.Mmat.shape[0]-self.Mmat.shape[1])
            self.Wovec = np.zeros(0)
            #self.Gr = np.dot(self.Hmat.T, self.residuals)

            # Converstion between orthogonal and non-orthogonal timing
            # model parameters
            MtM = np.dot(self.Mmat.T, self.Mmat)
            MtGc = np.dot(self.Mmat.T, self.Mmat_g)
            try:
                cf = sl.cho_factor(MtM)
                self.tmpConv = sl.cho_solve(cf, MtGc)
            except np.linalg.LinAlgError:
                U, s, Vh = sl.svd(MtM)
                self.tmpConv = np.dot(np.dot(Vh.T, np.dot(np.diag(1.0/s), U.T)), MtGc)

            # Also store the inverse
            try:
                Qs,Rs = sl.qr(self.tmpConv)
                self.tmpConvi = sl.solve(Rs, Qs.T)
            except np.linalg.LinAlgError:
                U, s, Vt = sl.svd(self.tmpConv)
                self.tmpConvi = np.dot(U * (1.0 / s), Vt)

            if write != 'none':
                # Write all these quantities to the HDF5 file
                #h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_Zmat', self.Zmat[self.iisort, :])
                h5df.addData(self.name, 'pic_tmpConv', self.tmpConv)
                h5df.addData(self.name, 'pic_tmpConvi', self.tmpConvi)



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
    @param noGmat:          Whether or not to read in the G-matrix
    @param gibbsmodel:      What coefficients to include in the Gibbs model
    @param bandRedNoise:    Frequency bands for band-limited red noise

    """
    def readPulsarAuxiliaries(self, h5df, Tmax, nfreqs, ndmfreqs, \
            twoComponent=False, nSingleFreqs=0, nSingleDMFreqs=0, \
            compression='None', likfunc='mark3', \
            evalCompressionComplement=True, memsave=True, noGmat=False, \
            gibbsmodel=[], bandRedNoise=[], complement=False):
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
        vslice = self.isort
        mslice = (self.isort, slice(None, None, None))
        if not likfunc in ['mark12', 'gibbs']:
            self.Gmat = np.array(h5df.getData(self.name, 'pic_Gmat', \
                    dontread=memsave, isort=mslice))
            self.Gcmat = np.array(h5df.getData(self.name, 'pic_Gcmat', \
                    dontread=memsave, isort=mslice))
            self.Homat = np.array(h5df.getData(self.name, 'pic_Homat', \
                    dontread=memsave, isort=mslice))
            self.Hocmat = np.array(h5df.getData(self.name, 'pic_Hocmat',
                dontread=(not evalCompressionComplement), isort=mslice))

            self.Gr = np.array(h5df.getData(self.name, 'pic_Gr'))
            self.GGr = np.array(h5df.getData(self.name, 'pic_GGr', \
                    dontread=memsave, isort=vslice))
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
        self.Dvec = np.array(h5df.getData(self.name, 'pic_Dvec', isort=vslice))

        if ndmf > 0:
            self.Fdmfreqs = np.array(h5df.getData(self.name, 'pic_Fdmfreqs', \
                    isort=mslice))
        else:
            self.Fdmfreqs = np.zeros(0)

        # If compression is not done, but Hmat represents a compression matrix,
        # we need to re-evaluate the lot. Raise an error
        if not noGmat and not likfunc in ['mark12', 'gibbs']:
            if compression == 'dont':
                pass
            elif (compression == 'None' or compression is None) and \
                    h5df.getShape(self.name, 'pic_Gmat')[1] != \
                    h5df.getShape(self.name, 'pic_Hmat')[1]:
                raise ValueError("Compressed file detected. Re-calculating all quantities.")
            elif (compression != 'None' and compression != None) and \
                    h5df.getShape(self.name, 'pic_Gmat')[1] == \
                    h5df.getShape(self.name, 'pic_Hmat')[1]:
                raise ValueError("Uncompressed file detected. Re-calculating all quantities.")

        if likfunc == 'mark1':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                    dontread=noGmat, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat',
                dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF'))
            self.GtD = np.array(h5df.getData(self.name, 'pic_GtD'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', \
                            dontread=memsave, isort=mslice))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', \
                    dontread=memsave, isort=mslice))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    dontread=memsave, isort=mslice))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark2':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                    dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc in ['mark3', 'mark3fa', 'mark3nc']:
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat',
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(not self.twoComponentNoise)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    isort=mslice))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark4':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.UtF = np.array(h5df.getData(self.name, 'pic_UtF'))
            self.UtD = np.array(h5df.getData(self.name, 'pic_UtD', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGU = np.array(h5df.getData(self.name, 'pic_AGU',
                dontread=(not self.twoComponentNoise)))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))
            self.Umat = np.array(h5df.getData(self.name, 'pic_Umat', \
                    isort=mslice))
            self.Jweight = np.array(h5df.getData(self.name, 'pic_Jweight', \
                    isort=vslice))
            self.Uimat = np.array(h5df.getData(self.name, 'pic_Uimat', \
                    isort=(slice(None, None, None), self.isort)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    dontread=memsave, isort=mslice))

        if likfunc == 'mark4ln':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.UtF = np.array(h5df.getData(self.name, 'pic_UtF', dontread=memsave))
            self.UtD = np.array(h5df.getData(self.name, 'pic_UtD'))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', \
                    dontread=memsave, isort=mslice))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', \
                            dontread=memsave, isort=mslice))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.UtFF = np.array(h5df.getData(self.name, 'pic_UtFF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGU = np.array(h5df.getData(self.name, 'pic_AGU',
                dontread=(not self.twoComponentNoise)))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))
            self.Umat = np.array(h5df.getData(self.name, 'pic_Umat', \
                    isort=mslice))
            self.Jweight = np.array(h5df.getData(self.name, 'pic_Jweight', \
                    isort=vslice))
            self.Uimat = np.array(h5df.getData(self.name, 'pic_Uimat', \
                    isort=(slice(None, None, None), self.isort)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    dontread=memsave, isort=mslice))

        if likfunc == 'mark6' or likfunc == 'mark6fa':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat', \
                    isort=mslice))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE',
                dontread=(not self.twoComponentNoise)))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', \
                            dontread=memsave, isort=mslice))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', \
                    dontread=memsave, isort=mslice))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    dontread=memsave, isort=mslice))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark7':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    isort=mslice))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark8':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat', \
                    dontread=memsave, isort=mslice))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE',
                dontread=(memsave and not self.twoComponentNoise)))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', \
                    dontread=memsave, isort=mslice))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', \
                    isort=mslice))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    isort=mslice))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark9':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', \
                    dontread=memsave, isort=mslice))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', \
                    dontread=memsave, isort=mslice))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr',
                dontread=(not self.twoComponentNoise)))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF',
                dontread=(memsave and not self.twoComponentNoise)))
            self.AGFF = np.array(h5df.getData(self.name, 'pic_AGFF',
                dontread=(not self.twoComponentNoise)))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    isort=mslice))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark10':
            self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat', \
                dontread=memsave, isort=mslice))
            self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat', \
                    dontread=self.twoComponentNoise))
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat', \
                    isort=mslice))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', \
                    dontread=memsave, isort=mslice))
            self.SFdmmat = np.array(h5df.getData(self.name, 'pic_SFdmmat', \
                    dontread=memsave, isort=mslice))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', \
                    dontread=memsave, isort=mslice))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.DSF = np.array(h5df.getData(self.name, 'pic_DSF', \
                    dontread=memsave, isort=mslice))
            self.DFF = np.array(h5df.getData(self.name, 'pic_DFF', \
                    isort=mslice))
            self.EEmat = np.array(h5df.getData(self.name, 'pic_EEmat', \
                    isort=mslice))
            self.GGtEE = np.array(h5df.getData(self.name, 'pic_GGtEE', \
                    isort=mslice))
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
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', \
                    dontread=memsave, isort=mslice))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', \
                    isort=mslice))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    isort=mslice))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc in ['mark12', 'gibbs']:
            self.Zmat = np.array(h5df.getData(self.name, 'pic_Zmat', \
                    isort=mslice))
            self.tmpConv = np.array(h5df.getData(self.name, 'pic_tmpConv'))
            self.tmpConvi = np.array(h5df.getData(self.name, 'pic_tmpConvi'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))
            self.Umat = np.array(h5df.getData(self.name, 'pic_Umat', \
                    isort=mslice))
            self.Uinds = np.array(h5df.getData(self.name, 'pic_Uinds'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', \
                    isort=mslice))
            self.FBmats = []
            self.Fbands = []
            for bb, band in enumerate(bandRedNoise):
                # Band-limited noise has zero response outside frequency band,
                # but is the same as Red Noise elsewhere
                FBmat = self.Fmat.copy()
                mask = np.logical_or(self.freqs < band[0], \
                        self.freqs > band[1])
                FBmat[mask, :] = 0.0
                self.FBmats.append(FBmat)
                self.Fbands.append(Fbands)

            self.FBmats = np.array(self.FBmats)
            self.Fbands = bandRedNoise

            if len(self.Fdmfreqs) > 0:
                self.DF = np.array(h5df.getData(self.name, 'pic_DF', \
                        isort=mslice))
                self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', \
                        isort=mslice))
            else:
                self.Fdmmat = np.zeros((len(self.toas),0))
                self.DF = np.zeros((len(self.toas),0))

            # Prepare the new design matrix bases
            self.gibbs_set_design(gibbsmodel)

            # Yeah, we've already got Zmat from file... But we've got to do
            # this, partially because we might have more now with the
            # band-limited stuff
            self.Zmat, self.Zmask = self.getZmat(gibbsmodel, which='all')
            self.Zmat_M, self.Zmask_M = self.getZmat(gibbsmodel, which='M')
            self.Zmat_F, self.Zmask_F = self.getZmat(gibbsmodel, which='F')
            self.Zmat_B, self.Zmask_B = self.getZmat(gibbsmodel, which='B')
            self.Zmat_D, self.Zmask_D = self.getZmat(gibbsmodel, which='D')
            self.Zmat_U, self.Zmask_U = self.getZmat(gibbsmodel, which='U')
            self.Zmat_N, self.Zmask_N = self.getZmat(gibbsmodel, which='N')
            self.gibbsresiduals = np.zeros(len(self.toas))

            self.gibbscoefficients = np.zeros(self.Zmat.shape[1])

            self.Wvec = np.zeros(self.Mmat.shape[0]-self.Mmat.shape[1])
            self.Wovec = np.zeros(0)



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

                if not likfunc in ['mark1', 'mark2', 'mark3', 'mark3fa',
                                    'mark3nc', 'mark4', 'mark7', 'mark9']:
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

                if not likfunc in ['mark1', 'mark2', 'mark3', 'mark3fa',
                                    'mark3nc', 'mark4', 'mark7', 'mark9']:
                    self.lEmat = np.append(self.Fmat[:,bf], self.DF[:,bfdm], axis=1)

                if likfunc in ['mark9', 'mark10']:
                    bff = np.append(bf, [True]*self.FFmat.shape[1])

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
    """
    Basic implementation of the model/likelihood.

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


    """
    Constructor. Read data/model if filenames are given

    @param h5filename:      HDF5 filename with pulsar data
    @param jsonfilename:    JSON file with model
    @param pulsars:         Which pulsars to read ('all' = all, otherwise provide a
                            list: ['J0030+0451', 'J0437-4715', ...])
                            WARNING: duplicates are _not_ checked for.
    @param sort:            How to internally sort the TOAs (None, 'time',
                            'jitterext'). Default: 'jitterext'
    """
    def __init__(self, h5filename=None, jsonfilename=None, pulsars='all', \
                 auxFromFile=True, noGmat=False, \
                 verbose=False, noCreate=False, sort='jitterext'):
        self.clear()

        if h5filename is not None:
            self.initFromFile(h5filename, pulsars=pulsars, sort=sort)

            if jsonfilename is not None:
                self.initModelFromFile(jsonfilename, auxFromFile=auxFromFile, \
                                       noGmat=noGmat, verbose=verbose, \
                                       noCreate=noCreate)

    """
    Clear all the structures present in the object
    """
    # TODO: Do we need to delete all with 'del'?
    def clear(self):
        self.h5df = None                        # The DataFile object
        self.ptapsrs = []                       # The ptaPulsar objects
        self.ptasignals = []                    # The model/signals description

        # Gibbs 'signals': which coefficients are included? Do this more elegantly
        # in the future. Possible values for now:
        # design, rednoise, dm, jitter, correx/corrim
        self.gibbsmodel = []


        self.dimensions = 0
        self.pmin = None
        self.pmax = None
        self.pstart = None
        self.pwidth = None

        self.pardes = None
        self.haveStochSources = False
        self.haveDetSources = False

        self.likfunc = 'mark3'                  # What likelihood function to use
        self.compression = 'None'
        evallikcomp = False    # Evaluate complement of compressed likelihood

        # Whether we use the option of forcing the frequency lines to be ordered in
        # the prior
        self.orderFrequencyLines = False
        self.haveStochSources = False
        self.haveDetSources = False

        # Additional informative quantities (reset after RJMCMC jump)
        self.npf = None      # Number of frequencies per pulsar (red noise/signal)
        self.npff = None     # Number of frequencies per pulsar (single-freq components)
        self.npfdm = None    # Number of frequencies per pulsar (DM)
        self.npe = None      # Number of frequencies per pulsar (rn + DM)
        self.npobs = None    # Number of observations per pulsar
        self.npgs = None     # Number of non-projected observations per pulsar (columns Hmat)
        self.npgos = None    # Number of orthogonal non-projected observations per pulsar (columns Homat)
        self.npu = None      # Number of avetoas per pulsar
        self.npm = None      # Number of columns of design matrix (full design matrix)
        self.npz = None      # Number of columns of Zmat
        self.npz_f = None      # Number of columns of Zmat
        self.npz_d = None      # Number of columns of Zmat
        self.npz_U = None      # Number of columns of Zmat
        self.Tmax = None     # One Tmax to rule them all...

        # Which pulsars have which red noise frequencies...
        self.freqmask = None     # Mask which have which
        self.freqb = None        # Quadratic Fourier coefficients, in a 2D-array

        # The Phi, Theta, and Sigma matrices
        self.Phi = None          # mark1, mark3, mark?, mark6                (Noise & corr)
        self.Phivec = None       # mark1, mark3, mark?, mark6          gibbs (Noise)
        self.Thetavec = None     #               mark?, mark6          gibbs (DM)
        self.Beta = None         #                                     gibbs (Band Noise)
        self.Muvec = None        #                             mark11        (Jitter)
        self.Svec = None         #                                     gibbs (GWB PSD)
        self.Scor = None         #                                     gibbs (GWB corr)
        self.Scor_im = None      #                                     gibbs (GWB impl)
        self.Scor_im_cf = None   #                                     gibbs (GWB impl)
        self.Scor_im_inv = None  #                                     gibbs (GWB impl)
        self.Sigma = None        #        mark3, mark?, mark6                (everything)
        self.Sigma_F = None      #                                     gibbs
        self.Sigma_F_cf = None   #                                     gibbs
        self.GNGldet = None      # mark1, mark3, mark?, mark6                (log-det)

        self.Fmat_gw = None      # The F-matrix, but for GWs
        self.Ffreqs_gw = None    # Frequencies of the GWs

        # Other quantities that we do not want to re-initialise every likelihood call
        self.rGr = None          # mark1, mark3, mark?, mark6
        self.rGFa = None         # mark1
        self.aFGFa = None        # mark1
        self.avec = None         # mark1
        self.rGF = None          #        mark3, mark?
        self.rGE = None          #                      mark6
        self.rGZ_F = None        #                                     gibbs
        self.rGZ_D = None        #                                     gibbs
        self.rGZ_U = None        #                                     gibbs
        self.FGGNGGF = None      #        mark3, mark?
        self.EGGNGGE = None      #                      mark6
        self.ZGGNGGZ = None      #                              gibbs
        self.NGGF = None         #        mark3, mark?  mark6


        # Gibbs auxiliaries for the marginalised conditionals
        self.GcNiGc_inv = []
        self.NiGc = []
        self.GcNiGcF = []
        self.GcNiGcD = []
        self.GcNiGcU = []
        self.NiF = []
        self.NiD = []
        self.NiU = []
        self.rGr_F = np.zeros(0)
        self.rGr_D = np.zeros(0)
        self.rGr_U = np.zeros(0)
        self.rGD = []
        self.rGU = []
        #self.rGF = np.zeros(0)
        self.UGGNGGU = []
        #self.FGGNGGF = np.zeros((0, 0))

        # Gibbs auxiliaries for the generative conditionals
        self.gibbs_ll_N = None
        self.gibbs_ll_D = None           
        self.gibbs_ll_U = None
        self.gibbs_ll_F = None

        self.gibbs_current_a = []         # List of quadratic parameters
        self.gibbs_proposed_a = []         # List of quadratic parameters

        # Whether we have already called the likelihood in one call, so we can skip
        # some things in comploglikelihood
        self.skipUpdateToggle = False


    """
    Initialise this likelihood object from an HDF5 file

    @param filename:    Name of the HDF5 file we will be reading
    @param pulsars:     Which pulsars to read ('all' = all, otherwise provide a
                        list: ['J0030+0451', 'J0437-4715', ...])
                        WARNING: duplicates are _not_ checked for.
    @param append:      If set to True, do not delete earlier read-in pulsars
    @param sort:        How to internally sort the residuals
                        (None, 'time', 'jitterext'. Need jitterext for fast
                        jitter)
    """
    def initFromFile(self, filename, pulsars='all', append=False, sort='jitterext'):
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
            newpsr.readFromH5(self.h5df, psrname, sort=sort)
            self.ptapsrs.append(newpsr)

    def getPulsarNames(self):
        """
        Get a list of the pulsar names
        """
        psrnames = []
        for psr in self.ptapsrs:
            psrnames.append(psr.name)

        return psrnames


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

        if 'freqband' in signal:
            signal['freqband'] = np.array(signal['freqband'])

        # Add the signal
        if signal['stype']=='efac':
            # Efac
            self.addSignalEfac(signal)
        elif signal['stype'] in ['equad', 'jitter']:
            # Equad or Jitter
            self.addSignalEquad(signal)
        elif signal['stype'] in ['powerlaw', 'spectrum', 'spectralModel', \
                'freqpowerlaw', 'freqspectrum', 'freqspectralModel']:
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
            ind = self.ptapsrs[signal['pulsarind']].flags != signal['flagvalue']
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
            ind = psr.flags != signal['flagvalue']
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
    @param flag:    (flagname, flagval) that must be matched

    @return:        Index array with signals that qualify
    """
    def getSignalNumbersFromDict(self, signals, stype='powerlaw', \
            corr='single', psrind=-2, flag=None):
        signalNumbers = []

        for ii, signal in enumerate(signals):
            if signal['stype'] == stype and signal['corr'] == corr:
                if psrind == -2 or signal['pulsarind'] == psrind:
                    if flag is None or \
                            (flag[0] == signal['flagname'] and \
                            flag[1] == signal['flagvalue']):
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
        self.npfb = np.zeros(npsrs, dtype=np.int)
        self.npu = np.zeros(npsrs, dtype=np.int)
        self.npff = np.zeros(npsrs, dtype=np.int)
        self.npfdm = np.zeros(npsrs, dtype=np.int)
        self.npffdm = np.zeros(npsrs, dtype=np.int)
        self.npobs = np.zeros(npsrs, dtype=np.int)
        self.npgs = np.zeros(npsrs, dtype=np.int)
        self.npgos = np.zeros(npsrs, dtype=np.int)
        self.npm = np.zeros(npsrs, dtype=np.int)
        self.npm_f = np.zeros(npsrs, dtype=np.int)
        self.npm_d = np.zeros(npsrs, dtype=np.int)
        self.npm_u = np.zeros(npsrs, dtype=np.int)
        self.npz = np.zeros(npsrs, dtype=np.int)
        self.npz_f = np.zeros(npsrs, dtype=np.int)
        self.npz_b = np.zeros(npsrs, dtype=np.int)
        self.npz_d = np.zeros(npsrs, dtype=np.int)
        self.npz_u = np.zeros(npsrs, dtype=np.int)
        for ii, psr in enumerate(self.ptapsrs):
            psr.Nvec = np.zeros(len(psr.toas))
            psr.Jvec = np.zeros(len(psr.avetoas))

            if not self.likfunc in ['mark2']:
                self.npf[ii] = len(psr.Ffreqs)
                self.npff[ii] = self.npf[ii]
                self.npfb[ii] = len(psr.Ffreqs)*len(psr.FBmats)

            if self.likfunc in ['mark4ln', 'mark9', 'mark10']:
                self.npff[ii] += len(psr.SFfreqs)

            self.npobs[ii] = len(psr.toas)

            self.npu[ii] = len(psr.avetoas)

            if self.likfunc in ['mark1', 'mark4', 'mark4ln', 'mark6', \
                    'mark6fa', 'mark8', 'mark10', 'mark12', 'gibbs']:
                self.npfdm[ii] = len(psr.Fdmfreqs)
                self.npffdm[ii] = len(psr.Fdmfreqs)

            if self.likfunc in ['mark10']:
                self.npffdm[ii] += len(psr.SFdmfreqs)

            if self.likfunc in ['mark1', 'mark2', 'mark3', 'mark3fa', 'mark3nc', 'mark4', \
                    'mark4ln', 'mark6', 'mark6fa', 'mark7', 'mark8', 'mark9', \
                    'mark10']:
                self.npgs[ii] = len(psr.Gr)
                self.npgos[ii] = len(psr.toas) - self.npgs[ii] #- psr.Mmat.shape[1]
                psr.Nwvec = np.zeros(self.npgs[ii])
                psr.Nwovec = np.zeros(self.npgos[ii])
            elif self.likfunc in ['mark12', 'gibbs']:
                self.npgs[ii] = len(psr.toas) - psr.Mmat.shape[1]
                self.npgos[ii] = len(psr.toas) - self.npgs[ii] #- psr.Mmat.shape[1]
                psr.Nwvec = np.zeros(self.npgs[ii])
                psr.Nwovec = np.zeros(self.npgos[ii])

            if self.likfunc in ['mark12', 'gibbs']:
                self.npm[ii] = psr.Mmat.shape[1]
                self.npz[ii] = psr.Zmat.shape[1]
                self.npm_f[ii] = np.sum(psr.Mmask_F)
                self.npm_d[ii] = np.sum(psr.Mmask_D)
                self.npm_u[ii] = np.sum(psr.Mmask_U)
                self.npz_f[ii] = psr.Zmat_F.shape[1]
                self.npz_b[ii] = psr.Zmat_B.shape[1]
                self.npz_d[ii] = psr.Zmat_D.shape[1]
                self.npz_u[ii] = psr.Zmat_U.shape[1]

        # These quantities are mostly used in Gibbs sampling, for when we
        # are numerically dealing with the quadratic coefficients.
        maxfreqs = np.max(self.npf)
        self.freqmask = np.zeros((npsrs, maxfreqs), dtype=np.bool)
        self.freqb = np.zeros((npsrs, maxfreqs))

        for jj, psr in enumerate(self.ptapsrs):
            self.freqmask[jj, :self.npf[jj]] = True  # No npff here?

        # Prepare the hyper-parameter covariance quantities
        self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        self.Phivec = np.zeros(np.sum(self.npf))
        self.Thetavec = np.zeros(np.sum(self.npfdm))
        self.Muvec = np.zeros(np.sum(self.npu))
        self.Svec = np.zeros(np.max(self.npf))
        self.Scor = np.zeros((len(self.ptapsrs), len(self.ptapsrs)))

        if self.likfunc == 'mark1':
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)

            self.Gr = np.zeros(np.sum(self.npgs))
            self.GCG = np.zeros((np.sum(self.npgs), np.sum(self.npgs)))
        elif self.likfunc == 'mark2':
            self.GNGldet = np.zeros(npsrs)
            self.rGr = np.zeros(npsrs)
        elif self.likfunc in ['mark3', 'mark3fa', 'mark3nc', 'mark7']:
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
        elif self.likfunc in ['mark12', 'gibbs']:
            zlen_f = np.sum(self.npz_f)
            zlen_d = np.sum(self.npz_d)
            zlen_u = np.sum(self.npz_u)

            self.Sigma_F = np.zeros((zlen_f, zlen_f))
            self.GNGldet = np.zeros(npsrs)
            self.Thetavec = np.zeros(np.sum(self.npfdm))
            self.Betavec = np.zeros(np.sum(self.npfb))
            self.rGZ_F = np.zeros(zlen_f)
            self.rGZ_D = np.zeros(zlen_d)
            self.rGZ_U = np.zeros(zlen_u)

            self.rGr = np.zeros(npsrs)

            # Select the largest F-matrix for GWs:
            ind = np.argmax(self.npf)
            self.Fmat_gw = self.ptapsrs[ind].Fmat.copy()
            self.Ffreqs_gw = self.ptapsrs[ind].Ffreqs.copy()

            # For the blocked Gibbs sampler with marginalised posteriors
            # conditionals, we quite a few auxiliaries
            # (Not needed anymore)
            self.GcNiGc_inv = []
            self.NiGc = []
            self.GcNiGcF = []
            self.GcNiGcD = []
            self.GcNiGcU = []
            self.NiF = []
            self.NiD = []
            self.NiU = []
            self.rGr_F = np.zeros(npsrs)
            self.rGr_D = np.zeros(npsrs)
            self.rGr_U = np.zeros(npsrs)
            self.rGD = []
            self.rGU = []
            self.rGF = np.zeros(np.sum(self.npff))
            self.UGGNGGU = []


            # self.FGGNGGF = np.zeros((np.sum(self.npff), np.sum(self.npff)))
            self.FNF = np.zeros((np.sum(self.npz_f), np.sum(self.npz_f)))
            #self.DND = np.zeros((np.sum(self.npz_d), np.sum(self.npz_d)))
            #self.UNU = np.zeros((np.sum(self.npz_u), np.sum(self.npz_u)))

            self.gibbs_ll_N = np.zeros(npsrs)
            self.gibbs_ll_D = np.zeros(npsrs)
            self.gibbs_ll_U = np.zeros(npsrs)
            self.gibbs_ll_F = 0.0

            for ii, psr in enumerate(self.ptapsrs):
                self.GcNiGc_inv.append(np.zeros((\
                        psr.Mmat.shape[1], psr.Mmat.shape[1])))
                self.NiGc.append(np.zeros(psr.Mmat.shape))

                if 'rednoise' in self.gibbsmodel:
                    self.GcNiGcF.append(\
                            np.zeros((psr.Mmat.shape[1], self.npff[ii])))
                    self.NiF.append(np.zeros(psr.Fmat.shape))

                if 'dm' in self.gibbsmodel:
                    self.GcNiGcD.append(\
                            np.zeros((psr.Mmat.shape[1], self.npffdm[ii])))
                    self.NiD.append(np.zeros(psr.DF.shape))
                    self.rGD.append(np.zeros(self.npffdm[ii]))

                if 'jitter' in self.gibbsmodel:
                    self.GcNiGcU.append(\
                            np.zeros((psr.Mmat.shape[1], self.npu[ii])))
                    self.NiU.append(np.zeros(psr.Umat.shape))
                    self.rGU.append(np.zeros(self.npu[ii]))
                    #self.UGGNGGU.append(np.zeros(\
                    #        (self.npu[ii], self.npu[ii])))


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
            bandRedNoise=None, bandNoiseModel='blpowerlaw', \
            noisePrior='flatlog', dmPrior='flatlog', \
            incDM=False, dmModel='powerlaw', \
            incClock=False, clockModel='powerlaw', \
            incGWB=False, gwbModel='powerlaw', \
            incDipole=False, dipoleModel='powerlaw', \
            incAniGWB=False, anigwbModel='powerlaw', lAniGWB=1, \
            incPixelGWB=False, pixelgwbModel='powerlaw', npixels=4, \
            incBWM=False, \
            incTimingModel=False, nonLinear=False, \
            keepTimingModelPars = None, \
            varyEfac=True, incEquad=False, \
            separateCEquads=False, separateEquads=False, \
            separateEfacs=False, \
            incCEquad=False, expandCEquad=True, \
            incJitter=False, expandJitter=True, \
            incSingleFreqNoise=False, \
                                        # True
            singlePulsarMultipleFreqNoise=None, \
                                        # [True, ..., False]
            multiplePulsarMultipleFreqNoise=None, \
                                        # [0, 3, 2, ..., 4]
            dmFrequencyLines=None, \
                                        # [0, 3, 2, ..., 4]
            orderFrequencyLines=False, \
            compression = 'None', \
            evalCompressionComplement = False, \
            explicitGWBcomponents = False, \
            likfunc='mark12'):
        # We have to determine the number of frequencies we'll need
        numNoiseFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)
        numDMFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)
        numSingleFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)
        numSingleDMFreqs = np.zeros(len(self.ptapsrs), dtype=np.int)

        # For Gibbs sampling, keep track of the included coefficients
        gibbsmodel = ['design']

        # If we have Gibbs sampling, do not do any compression:
        if likfunc in ['mark12', 'gibbs'] and compression == "None":
            compression = 'dont'

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

        for pp, m2psr in enumerate(self.ptapsrs):
            if separateEfacs:
                uflagvals = list(set(m2psr.flags))  # Unique flags
                for flagval in uflagvals:
                    newsignal = OrderedDict({
                        "stype":"efac",
                        "corr":"single",
                        "pulsarind":pp,
                        "flagname":"efacequad",
                        "flagvalue":flagval,
                        "bvary":[varyEfac],
                        "pmin":[0.001],
                        "pmax":[50.0],
                        "pwidth":[0.1],
                        "pstart":[1.0],
                        "prior":'flat'
                        })
                    signals.append(newsignal)
            else:
                newsignal = OrderedDict({
                    "stype":"efac",
                    "corr":"single",
                    "pulsarind":pp,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":[varyEfac],
                    "pmin":[0.001],
                    "pmax":[50.0],
                    "pwidth":[0.1],
                    "pstart":[1.0],
                    "prior":'flat'
                    })
                signals.append(newsignal)

            if incEquad:
                if separateEquads:
                    uflagvals = list(set(m2psr.flags))  # Unique flags
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype":"equad",
                            "corr":"single",
                            "pulsarind":pp,
                            "flagname":"efacequad",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.2],
                            "pstart":[-8.0],
                            "prior":'flatlog'
                            })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype":"equad",
                        "corr":"single",
                        "pulsarind":pp,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":[True],
                        "pmin":[-10.0],
                        "pmax":[-4.0],
                        "pwidth":[0.2],
                        "pstart":[-8.0],
                        "prior":'flatlog'
                        })
                    signals.append(newsignal)

            if incCEquad or incJitter:
                if not 'jitter' in gibbsmodel and (expandJitter and expandCEquad):
                    # We are expanding the Jitter/CEquad in the Gibbs sampler
                    gibbsmodel.append('jitter')
                if separateCEquads and not likfunc in ['mark12', 'gibbs']:
                    uflagvals = list(set(m2psr.flags))  # Unique flags
                    for flagval in uflagvals:
                        newsignal = OrderedDict({
                            "stype":"jitter",
                            "corr":"single",
                            "pulsarind":pp,
                            "flagname":"efacequad",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.3],
                            "pstart":[-8.0],
                            "prior":'flatlog'
                            })
                        signals.append(newsignal)
                elif separateCEquads and likfunc in ['mark12', 'gibbs']:
                    # Need to decide on number of jitter parameters, and
                    # number of epochs with jitter first for Gibbs sampler
                    if not checkTOAsort(m2psr.toas, m2psr.flags, \
                            which='jitterext'):
                        raise ValueError("TOAs not jitter-sorted")
                    (m2psr.avetoas, m2psr.Umat, m2psr.Uimat) = \
                            quantize_split(m2psr.toas, m2psr.flags, calci=True)
                    m2psr.Umat, m2psr.Uimat, m2psr.avetoas, jflags = \
                            quantreduce(m2psr.Umat, m2psr.avetoas, m2psr.flags, calci=True)
                    if not checkquant(m2psr.Umat, m2psr.flags, jflags):
                        raise ValueError("Quantization matrix error!")
                    for flagval in jflags:
                        newsignal = OrderedDict({
                            "stype":"jitter",
                            "corr":"single",
                            "pulsarind":pp,
                            "flagname":"efacequad",
                            "flagvalue":flagval,
                            "bvary":[True],
                            "pmin":[-10.0],
                            "pmax":[-4.0],
                            "pwidth":[0.3],
                            "pstart":[-8.0],
                            "prior":'flatlog'
                            })
                        signals.append(newsignal)
                else:
                    newsignal = OrderedDict({
                        "stype":"jitter",
                        "corr":"single",
                        "pulsarind":pp,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":[True],
                        "pmin":[-10.0],
                        "pmax":[-4.0],
                        "pwidth":[0.3],
                        "pstart":[-8.0],
                        "prior":'flatlog'
                        })
                    signals.append(newsignal)

            if incRedNoise:
                if not 'rednoise' in gibbsmodel:
                    gibbsmodel.append('rednoise')
                if noiseModel=='spectrum':
                    nfreqs = numNoiseFreqs[pp]
                    bvary = [True]*nfreqs
                    pmin = [-18.0]*nfreqs
                    pmax = [-7.0]*nfreqs
                    pstart = [-10.0]*nfreqs
                    pwidth = [0.1]*nfreqs
                elif noiseModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-20.0, 0.02, 1.0e-11]
                    pmax = [-10.0, 6.98, 3.0e-9]
                    pstart = [-15.0, 2.01, 1.0e-10]
                    pwidth = [0.3, 0.3, 5.0e-11]
                elif noiseModel=='spectralModel':
                    bvary = [True, True, True]
                    pmin = [-28.0, 0.0, -8.0]
                    pmax = [15.0, 12.0, 2.0]
                    pstart = [-22.0, 2.0, -1.0]
                    pwidth = [-0.2, 0.1, 0.1]
                else:
                    raise ValueError("ERROR: option {0} not known".
                            format(noiseModel))

                newsignal = OrderedDict({
                    "stype":noiseModel,
                    "corr":"single",
                    "pulsarind":pp,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":noisePrior
                    })
                signals.append(newsignal)

            if bandRedNoise is not None:
                # We have band-limited red noise, at these frequency bands:
                bandRedNoise = np.atleast_2d(bandRedNoise)

                if not 'freqrednoise' in gibbsmodel:
                    gibbsmodel.append('freqrednoise')
                for ii, band in enumerate(bandRedNoise):
                    # Every row contains a low-high freq combination
                    if bandNoiseModel=='blspectrum':
                        nfreqs = numNoiseFreqs[pp]
                        bvary = [True]*nfreqs
                        pmin = [-18.0]*nfreqs
                        pmax = [-7.0]*nfreqs
                        pstart = [-10.0]*nfreqs
                        pwidth = [0.1]*nfreqs
                    elif bandNoiseModel=='blpowerlaw':
                        bvary = [True, True, False]
                        pmin = [-20.0, 0.02, 1.0e-11]
                        pmax = [-10.0, 6.98, 3.0e-9]
                        pstart = [-15.0, 2.01, 1.0e-10]
                        pwidth = [0.3, 0.3, 5.0e-11]
                    elif bandNoiseModel=='blspectralModel':
                        bvary = [True, True, True]
                        pmin = [-28.0, 0.0, -8.0]
                        pmax = [15.0, 12.0, 2.0]
                        pstart = [-22.0, 2.0, -1.0]
                        pwidth = [-0.2, 0.1, 0.1]
                    else:
                        raise ValueError("ERROR: option {0} not known".
                                format(bandNoiseModel))

                    newsignal = OrderedDict({
                        "stype":bandNoiseModel,
                        "corr":"single",
                        "freqband":band,
                        "pulsarind":pp,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":bvary,
                        "pmin":pmin,
                        "pmax":pmax,
                        "pwidth":pwidth,
                        "pstart":pstart,
                        "prior":noisePrior
                        })
                    signals.append(newsignal)

            if incDM:
                if not 'dm' in gibbsmodel:
                    gibbsmodel.append('dm')
                if dmModel=='dmspectrum':
                    nfreqs = numDMFreqs[pp]
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
                    pwidth = [0.3, 0.3, 5.0e-11]
                    #dmModel = 'dmpowerlaw'
                else:
                    raise ValueError("ERROR: option {0} not known".
                            format(dmModel))

                newsignal = OrderedDict({
                    "stype":dmModel,
                    "corr":"single",
                    "pulsarind":pp,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":dmPrior
                    })
                signals.append(newsignal)

            for jj in range(numSingleFreqs[pp]):
                newsignal = OrderedDict({
                    "stype":'frequencyline',
                    "corr":"single",
                    "pulsarind":pp,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":[True, True],
                    "pmin":[-9.0, -18.0],
                    "pmax":[-5.0, -9.0],
                    "pwidth":[-0.1, -0.1],
                    "pstart":[-7.0, -10.0],
                    "prior":'flatlog'
                    })
                signals.append(newsignal)

            for jj in range(numSingleDMFreqs[pp]):
                newsignal = OrderedDict({
                    "stype":'dmfrequencyline',
                    "corr":"single",
                    "pulsarind":pp,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":[True, True],
                    "pmin":[-9.0, -18.0],
                    "pmax":[-5.0, -9.0],
                    "pwidth":[-0.1, -0.1],
                    "pstart":[-7.0, -10.0],
                    "prior":'flatlog'
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

                if keepTimingModelPars is None and not likfunc == 'mark11':
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
                elif likfunc != 'mark11':
                    newptmdescription = m2psr.getNewTimingModelParameterList( \
                            keep=False, tmpars=keepTimingModelPars)
                else:
                    # Actually, if we are in likfunc == 'mark11', we need _all_
                    # TM parameters in the model
                    newptmdescription = []

                    if keepTimingModelPars is not None:
                        print("WARNING: cannot analytically treat timing model -- ignoring")

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
                    "pulsarind":pp,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "parid":parids,
                    "prior":'flat'
                    })
                signals.append(newsignal)

            if likfunc == 'mark11':
                nmodes = 2*numNoiseFreqs[pp]
                bvary = [True]*nmodes
                pmin = [-1.0e-3]*nmodes
                pmax = [1.0e-3]*nmodes
                pstart = [0.0]*nmodes
                pwidth = [1.0e-8]*nmodes

                newsignal = OrderedDict({
                    "stype":'fouriermode',
                    "corr":"single",
                    "pulsarind":pp,
                    "flagname":"pulsarname",
                    "flagvalue":m2psr.name,
                    "bvary":bvary,
                    "pmin":pmin,
                    "pmax":pmax,
                    "pwidth":pwidth,
                    "pstart":pstart,
                    "prior":'flat'
                    })
                signals.append(newsignal)

                if incDM:
                    nmodes = 2*numDMFreqs[pp]
                    bvary = [True]*nmodes
                    pmin = [-1.0]*nmodes
                    pmax = [1.0]*nmodes
                    pstart = [0.0]*nmodes
                    pwidth = [1.0e-5]*nmodes

                    newsignal = OrderedDict({
                        "stype":'dmfouriermode',
                        "corr":"single",
                        "pulsarind":pp,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":bvary,
                        "pmin":pmin,
                        "pmax":pmax,
                        "pwidth":pwidth,
                        "pstart":pstart,
                        "prior":'flat'
                        })
                    signals.append(newsignal)

                if incJitter or incCEquad:
                    # Still part of 'mark11'
                    (avetoas, Umat) = quantize_fast(m2psr.toas)
                    print("WARNING: per-backend epoch averaging not supported in mark11")
                    nmodes = len(avetoas)
                    bvary = [True]*nmodes
                    pmin = [-1.0e3]*nmodes
                    pmax = [1.0e3]*nmodes
                    pstart = [0.0]*nmodes
                    pwidth = [1.0e-8]*nmodes

                    newsignal = OrderedDict({
                        "stype":'jitterfouriermode',
                        "corr":"single",
                        "pulsarind":pp,
                        "flagname":"pulsarname",
                        "flagvalue":m2psr.name,
                        "bvary":bvary,
                        "pmin":pmin,
                        "pmax":pmax,
                        "pwidth":pwidth,
                        "pstart":pstart,
                        "prior":'flat'
                        })
                    signals.append(newsignal)

                    del avetoas
                    del Umat

        if incGWB:
            if not 'rednoise' in gibbsmodel:
                gibbsmodel.append('rednoise')
            if not 'correx' in gibbsmodel and explicitGWBcomponents:
                gibbsmodel.append('correx')
            elif not 'corrim' in gibbsmodel and not explicitGWBcomponents:
                gibbsmodel.append('corrim')
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
                pwidth = [0.3, 0.3, 5.0e-11]
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
                "pstart":pstart,
                "prior":'flatlog'
                })
            signals.append(newsignal)

        if incClock:
            if not 'rednoise' in gibbsmodel:
                gibbsmodel.append('rednoise')
            if not 'correx' in gibbsmodel and explicitGWBcomponents:
                gibbsmodel.append('correx')
            elif not 'corrim' in gibbsmodel and not explicitGWBcomponents:
                gibbsmodel.append('corrim')
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
                pwidth = [0.3, 0.3, 5.0e-11]
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
                "pstart":pstart,
                "prior":'flatlog'
                })
            signals.append(newsignal)

        if incDipole:
            if not 'rednoise' in gibbsmodel:
                gibbsmodel.append('rednoise')
            if not 'correx' in gibbsmodel and explicitGWBcomponents:
                gibbsmodel.append('correx')
            elif not 'corrim' in gibbsmodel and not explicitGWBcomponents:
                gibbsmodel.append('corrim')
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
                pwidth = [0.3, 0.3, 5.0e-11]
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
                "pstart":pstart,
                "prior":'flatlog'
                })
            signals.append(newsignal)

        if incAniGWB:
            if not 'rednoise' in gibbsmodel:
                gibbsmodel.append('rednoise')
            if not 'correx' in gibbsmodel and explicitGWBcomponents:
                gibbsmodel.append('correx')
            elif not 'corrim' in gibbsmodel and not explicitGWBcomponents:
                gibbsmodel.append('corrim')
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
                pwidth = [0.3, 0.3, 5.0e-11] + clmwidth
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
                "lAniGWB":lAniGWB,
                "prior":'flatlog'
                })
            signals.append(newsignal)

        if incPixelGWB:
            if not 'rednoise' in gibbsmodel:
                gibbsmodel.append('rednoise')
            if not 'correx' in gibbsmodel and explicitGWBcomponents:
                gibbsmodel.append('correx')
            elif not 'corrim' in gibbsmodel and not explicitGWBcomponents:
                gibbsmodel.append('corrim')
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
                pwidth = [0.3, 0.3, 5.0e-11] + [0.1, 0.1] * npixels
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
                "npixels":npixels,
                "prior":'flatlog'
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
                "pstart":[0.5*(toamax+toamin), -15.0, 3.0, 1.0, 1.0],
                "prior":'flatlog'
                })
            signals.append(newsignal)

        # The list of signals
        modeldict = OrderedDict({
            "file version":2014.12,
            "author":"piccard-makeModel",
            "numpulsars":len(self.ptapsrs),
            "pulsarnames":[self.ptapsrs[pp].name for pp in range(len(self.ptapsrs))],
            "numNoiseFreqs":list(numNoiseFreqs),
            "numDMFreqs":list(numDMFreqs),
            "compression":compression,
            "orderFrequencyLines":orderFrequencyLines,
            "evalCompressionComplement":evalCompressionComplement,
            "likfunc":likfunc,
            "Tmax":Tmax,
            "gibbsmodel":gibbsmodel,
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

            if 'freqband' in signals[-1]:
                signals[-1]['freqband'] = map(float, signals[-1]['freqband'])

        modeldict = OrderedDict({
            "file version":2014.12,
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
            "gibbsmodel":self.gibbsmodel,
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
            raise IOError, "Number of pulsars does not match sum of dictionary models: {0}, {1}".format(ndictpsrs, len(self.ptapsrs))

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

    def getPsrFreqBands(self, modeldict, pp):
        """
        Given a model dictionary, and a pulsar number, return the
        band-limited-noise frequency bands.

        @param modeldict:   Model dictionary
        @param pp:          Pulsar number
        """
        signals = modeldict['signals']
        bandRedNoise = []

        for ii, signal in enumerate(signals):
            if 'freqband' in signal and signal['pulsarind'] == pp:
                bandRedNoise.append(signal['freqband'])

        return np.array(bandRedNoise)


    """
    Initialise the model.
    @param numNoiseFreqs:       Dictionary with the full model
    @param fromFile:            Try to read the necessary Auxiliaries quantities
                                from the HDF5 file
    @param verbose:             Give some extra information about progress
    """
    def initModel(self, fullmodel, fromFile=True, verbose=False, \
                  noGmatWrite=False, noCreate=False, \
                  addDMQSD=False, threshold=1.0):
        numNoiseFreqs = fullmodel['numNoiseFreqs']
        numDMFreqs = fullmodel['numDMFreqs']
        compression = fullmodel['compression']
        evalCompressionComplement = fullmodel['evalCompressionComplement']
        orderFrequencyLines = fullmodel['orderFrequencyLines']
        likfunc = fullmodel['likfunc']
        signals = fullmodel['signals']
        if 'gibbsmodel' in fullmodel:
            gibbsmodel = fullmodel['gibbsmodel']
        elif likfunc == 'gibbs':
            raise ValueError("gibbsmodel not set in model")

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
        self.gibbsmodel = gibbsmodel
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
        if self.likfunc[:5] == 'mark6':
            separateEfacs = (numEfacs + numEquads) > 2
        else:
            separateEfacs = (numEfacs + numEquads + numJits) > 2

        if self.likfunc in ['mark12', 'gibbs']:
            # For now, only trim the quantization matrix Umat for Gibbs/mark12
            trimquant = True
        else:
            # All mark's do not support quantization trimming yet
            trimquant = False

        # When doing Gibbs, we really do not want to separate this stuff
        # TODO: Why are we always separating efacs for Gibbs?
        if likfunc in ['gibbs', 'mark11', 'mark12']:
            separateEfacs[:] = True

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

            # Get the frequency bands for the band-limited noise
            bandRedNoise = self.getPsrFreqBands(fullmodel, pindex)

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
                        memsave=True, noGmat=noGmatWrite, \
                        gibbsmodel=self.gibbsmodel, bandRedNoise=bandRedNoise, \
                        complement=self.evallikcomp)
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
                                noGmatWrite=noGmatWrite, threshold=threshold, \
                                gibbsmodel=self.gibbsmodel, trimquant=trimquant, \
                                bandRedNoise=bandRedNoise, complement=self.evallikcomp)

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
        self.registerModel()

    def registerModel(self):
        """
        After changing the model, we re-set the number of dimensions, the prior,
        and all the model dictionaries.
        """
        self.setPsrNoise_inds()
        self.setDimensions()
        self.initPrior()
        self.pardes = self.getModelParameterList()
        self.pardesgibbs = self.getGibbsModelParameterList()


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
                elif sig['stype'] == 'blpowerlaw':
                    flagname = 'BLpowerlaw'
                    flagvalue = ['BL-Amplitude', 'BL-spectral-index', 'BL-low-frequency-cutoff'][jj]
                elif sig['stype'] == 'blspectralModel':
                    flagname = 'blspectralModel'
                    flagvalue = ['BL-SM-Amplitude', 'BL-SM-spectral-index', 'BL-SM-corner-frequency'][jj]
                elif sig['stype'] == 'blspectrum':
                    flagname = 'blfrequency'
                    flagvalue = str(self.ptapsrs[psrindex].Ffreqs[2*jj])
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
    Get a list of all the model parameters that are extra coefficients in the
    case of Gibbs sampling.

    Note, at the moment, the Gibbs sampling routines do not check for parameters
    collisions between the Fourier/timing model modes and the deterministic
    parameters. This needs to be changed in the future to account for non-linear
    timing model parameters
    """
    def getGibbsModelParameterList(self):
        gpardes = []

        # The number of 'dimensions' in the likelihood does not count these
        # 'secondary' parameters we describe here. Use 'dimensions' as the start
        # index
        index = self.dimensions
        if self.likfunc == 'gibbs':
            for pp, psr in enumerate(self.ptapsrs):
                # First do the timing model parameters
                if 'design' in self.gibbsmodel:
                    for ii in range(psr.Mmat.shape[1]):
                        flagname = "Timing-model-"+psr.name
                        flagvalue = psr.ptmdescription[ii]

                        gpardes.append(\
                                {'index': index, 'pulsar': pp, 'sigindex': -1, \
                                    'sigtype': 'Mmat', 'correlation': 'single', \
                                    'name': flagname, 'id': flagvalue, 'pulsarname': \
                                    psr.name})
                        index += 1

                if 'rednoise' in self.gibbsmodel:
                    for ii in range(psr.Fmat.shape[1]):
                        if ii % 2 == 0:
                            flagname = "RN-Cos-"+psr.name
                        else:
                            flagname = "RN-Sin-"+psr.name

                        flagvalue = str(psr.Ffreqs[ii])

                        gpardes.append(\
                                {'index': index, 'pulsar': pp, 'sigindex': -1, \
                                    'sigtype': 'Fmat', 'correlation': 'single', \
                                    'name': flagname, 'id': flagvalue, 'pulsarname': \
                                    psr.name})
                        index += 1

                if 'dm' in self.gibbsmodel:
                    for ii in range(psr.Fdmfreqs.shape[0]):
                        if ii % 2 == 0:
                            flagname = "DM-Cos-"+psr.name
                        else:
                            flagname = "DM-Sin-"+psr.name

                        flagvalue = str(psr.Fdmfreqs[ii])

                        gpardes.append(\
                                {'index': index, 'pulsar': pp, 'sigindex': -1, \
                                    'sigtype': 'Fmat_dm', 'correlation': 'single', \
                                    'name': flagname, 'id': flagvalue, 'pulsarname': \
                                    psr.name})
                        index += 1

                if 'jitter' in self.gibbsmodel:
                    for ii in range(psr.Umat.shape[1]):
                        flagname = "Ave-res-"+psr.name
                        flagvalue = str(psr.avetoas[ii])

                        gpardes.append(\
                                {'index': index, 'pulsar': pp, 'sigindex': -1, \
                                    'sigtype': 'Residual_ave', 'correlation': 'single', \
                                    'name': flagname, 'id': flagvalue, 'pulsarname': \
                                    psr.name})
                        index += 1

                if 'correx' in self.gibbsmodel:
                    for ii in range(self.Fmat_gw.shape[1]):
                        flagname = "Corr-sig-"+psr.name
                        flagvalue = str(psr.Ffreqs[ii])

                        gpardes.append(\
                                {'index': index, 'pulsar': pp, 'sigindex': -1, \
                                    'sigtype': 'Fmat_corr', 'correlation': 'single', \
                                    'name': flagname, 'id': flagvalue, 'pulsarname': \
                                    psr.name})
                        index += 1

        return gpardes


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

        # Do the same for the Gibbs parameter descriptions, if we have 'm
        if self.likfunc == 'gibbs':
            for pdg in self.pardesgibbs:
                fil.write("{0:d} \t{1:d} \t{2:s} \t{3:s} \t{4:s} \t{5:s} \t{6:s}\n".format(\
                        pdg['index'],
                        pdg['pulsar'],
                        pdg['sigtype'],
                        pdg['correlation'],
                        pdg['name'],
                        pdg['id'],
                        pdg['pulsarname']))

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
                fil.write("{0} \t{1} \t{2} \t{3} \t{4}\n".format(\
                        pic_T0 + psr.toas[ii]/pic_spd, \
                        psr.residuals[ii], \
                        psr.toaerrs[ii], \
                        psr.flags[ii], \
                        psr.freqs[ii]))
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
        for ii, psr in enumerate(self.ptapsrs):
            if psr.twoComponentNoise:
                psr.Nwvec[:] = 0
                psr.Nwovec[:] = 0
            #else:
            psr.Nvec[:] = 0
            psr.Jvec[:] = 0

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


    def constructPhiAndTheta(self, parameters, selection=None, \
            make_matrix=True, noise_vec=False, gibbs_expansion=False):
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

        @param parameters:      Value of all the model parameters
        @param selection:       Boolean array that selects signals to include
        @param make_matrix:   Whether or not to actually build the matrix
                                (otherwise just the Phi vector)
        @param noise_vec:       If true, place noise only in the phi vector, not
                                in the matrix
        @param gibbs_expansion: Do not build the Phi matrix, but place the
                                correlated signals (only) in two tensor-product
                                components (H&D or other)
        """

        if make_matrix:
            self.Phi[:] = 0.0         # Start with a fresh matrix

        self.Phivec[:] = 0.0      # ''
        self.Thetavec[:] = 0.0    # ''
        self.Svec[:] = 0.0
        self.Scor[:] = 0.0
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
                        if make_matrix and not noise_vec:
                            self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += 10**pcdoubled

                        self.Phivec[findex:findex+2*nfreq] += 10**pcdoubled
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
                            pcdoubled = np.array([sparameters[:-2*npixels], \
                                    sparameters[:-2*npixels]]).T.flatten()

                        if make_matrix:
                            # Only add it if we need the full matrix
                            indexa = 0
                            indexb = 0
                            for aa in range(npsrs):
                                for bb in range(npsrs):
                                    # Some pulsars may have fewer frequencies than
                                    # others (right?). So only use overlapping ones
                                    nof = np.min([self.npf[aa], self.npf[bb], 2*nfreq])
                                    di = np.diag_indices(nof)

                                    self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += 10**pcdoubled[:nof] * corrmat[aa, bb]
                                    indexb += self.npff[bb]
                                indexb = 0
                                indexa += self.npff[aa]

                        self.Svec += 10**pcdoubled
                        if gibbs_expansion:
                            # Expand in spectrum and correlations
                            self.Scor = corrmat.copy()     # Yes, well, there can be only one
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

                        if make_matrix and not noise_vec:
                            self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled

                        self.Phivec[findex:findex+2*nfreq] += pcdoubled
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

                        if make_matrix:
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

                                    self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * corrmat[aa, bb]
                                    indexb += self.npff[bb]
                                indexb = 0
                                indexa += self.npff[aa]

                        self.Svec += pcdoubled
                        if gibbs_expansion:
                            # Expand in spectrum and correlations
                            self.Scor = corrmat.copy()
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

                        if make_matrix and not noise_vec:
                            self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled

                        self.Phivec[findex:findex+2*nfreq] += pcdoubled
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

                        if make_matrix:
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

                                    self.Phi[indexa:indexa+nof,indexb:indexb+nof][di] += pcdoubled[:nof] * corrmat[aa, bb]
                                    indexb += self.npff[bb]
                                indexb = 0
                                indexa += self.npff[aa]
                        
                        self.Svec += pcdoubled
                        if gibbs_expansion:
                            # Expand in spectrum and correlations
                            self.Scor = corrmat.copy()     # Yes, well, there can be only one
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

                    if make_matrix and not noise_vec:
                        self.Phi[findex:findex+2, findex:findex+2][di] += 10**pcdoubled

                    self.Phivec[findex:findex+2] += 10**pcdoubled
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
                            (sparameters-m2signal['pstart']))

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



    def setSinglePsrNoise(self, parameters, selection=None, pp=0):
        """
        Same as setPsrNoise, but now for a single pulsar
        
        @param parameters:  the full array of parameters
        @param selection::  mask of signals to include
        @param pp:          index of pulsar to do
        """

        psr = self.ptapsrs[pp]

        # Re-set all the pulsar noise vectors
        if psr.twoComponentNoise:
            psr.Nwvec[:] = 0
            psr.Nwovec[:] = 0
        #else:
        psr.Nvec[:] = 0
        psr.Jvec[:] = 0

        if selection is None:
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # Loop over all white noise signals, and fill the pulsar Nvec
        for ss, m2signal in enumerate(self.ptasignals):
            if m2signal['pulsarind'] == pp and selection[ss]:
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


    def setSinglePsrNoise_fast(self, parameters, pp=0, joinNJ=True):
        """
        Same as setPsrNoise, but now for a single pulsar
        
        @param parameters:  the full array of parameters
        @param pp:          index of pulsar to do
        @param joinNJ:      Include jitter/ecorr
        """

        psr = self.ptapsrs[pp]

        if joinNJ:
            inds = psr.gibbs_NJ_sinds
        else:
            inds = psr.gibbs_N_sinds

        # Re-set all the pulsar noise vectors
        if psr.twoComponentNoise:
            psr.Nwvec[:] = 0
            psr.Nwovec[:] = 0

        psr.Nvec[:] = 0
        psr.Jvec[:] = 0

        for ss in inds:
            m2signal = self.ptasignals[ss]
            if m2signal['pulsarind'] == pp:
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
                elif m2signal['stype'] in ['jitter', 'cequad']:
                    if m2signal['npars'] == 1:
                        pequadsqr = 10**(2*parameters[m2signal['parindex']])
                    else:
                        pequadsqr = 10**(2*m2signal['pstart'][0])

                    psr.Jvec += m2signal['Jvec'] * pequadsqr

    def setPsrNoise_inds(self):
        """
        Set the noise signal indices: gibbs_N_sinds. Used so that we don't have
        to loop over all the signals everytime.
        """
        for pp, psr in enumerate(self.ptapsrs):
            psr.gibbs_N_sinds = []
            psr.gibbs_NJ_sinds = []

        slistNJ = ['efac', 'equad', 'jitter', 'cequad']
        slistN = ['efac', 'equad']

        # Loop over all white noise signals, and fill the pulsar Nvec
        for ss, m2signal in enumerate(self.ptasignals):
            pp = m2signal['pulsarind']
            psr = self.ptapsrs[pp]
            if m2signal['stype'] in slistNJ:
                psr.gibbs_NJ_sinds.append(ss)
            if m2signal['stype'] in slistN:
                psr.gibbs_N_sinds.append(ss)

        psr.gibbs_N_sinds = np.array(psr.gibbs_N_sinds)
        psr.gibbs_NJ_sinds = np.array(psr.gibbs_NJ_sinds)



    def setTheta(self, parameters, selection=None, pp=0):
        """
        Same as constructPhiAndTheta, but now for a single pulsar and only DM
        variations (Theta)
        
        @param parameters:  the full array of parameters
        @param selection::  mask of signals to include
        @param pp:          index of pulsar to do
        """
        psr = self.ptapsrs[pp]

        # Re-set the relevant parts of Theta
        inds = np.sum(self.npfdm[:pp])
        inde = inds + self.npfdm[pp]
        self.Thetavec[inds:inde] = 0.0

        if selection is None:
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # Loop over all signals, and fill the phi matrix
        #for m2signal in self.ptasignals:
        for ss, m2signal in enumerate(self.ptasignals):
            if m2signal['pulsarind'] == pp and selection[ss]:
                # Create a parameters array for this particular signal
                sparameters = m2signal['pstart'].copy()
                sparameters[m2signal['bvary']] = \
                        parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']]
                if m2signal['stype'] == 'dmspectrum':
                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npffdm[:m2signal['pulsarind']])
                        nfreq = int(self.npfdm[m2signal['pulsarind']]/2)

                        pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                        # Fill the Theta matrix
                        self.Thetavec[findex:findex+2*nfreq] += 10**pcdoubled
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
                elif m2signal['stype'] == 'dmfrequencyline':
                    # For a DM frequency line, the DFF is assumed to be set elsewhere
                    findex = np.sum(self.npffdm[:m2signal['pulsarind']]) + \
                            self.npfdm[m2signal['pulsarind']] + 2*m2signal['npsrdmfreqindex']

                    pcdoubled = np.array([sparameters[1], sparameters[1]])
                    self.Thetavec[findex:findex+2] += 10**pcdoubled

    def setPhi(self, parameters, selection=None, gibbs_expansion=True):
        """
        Same as constructPhiAndTheta, but now only for the red noise/GWB signal
        
        @param parameters:      the full array of parameters
        @param selection::      mask of signals to include
        @param gibbs_expansion: Place correlated signals (only) in two
                                tensor-product components

        Note: to clarify, this function will not populate Phi. Only Phivec, and
        optionally the correlated Gibbs expansion units.
        """

        self.Phivec[:] = 0      # ''
        self.Svec[:] = 0
        self.Scor[:] = 0
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

                        self.Phivec[findex:findex+2*nfreq] += 10**pcdoubled
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
                            pcdoubled = np.array([sparameters[:-2*npixels], \
                                    sparameters[:-2*npixels]]).T.flatten()


                        self.Svec += 10**pcdoubled
                        if gibbs_expansion:
                            # Expand in spectrum and correlations
                            self.Scor = corrmat.copy()     # Yes, well, there can be only one
                elif m2signal['stype'] == 'powerlaw':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npff[:m2signal['pulsarind']])
                        nfreq = int(self.npf[m2signal['pulsarind']]/2)
                        freqpy = self.ptapsrs[m2signal['pulsarind']].Ffreqs * pic_spy
                        pcdoubled = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * m2signal['Tmax'])) * freqpy ** (-Si)

                        self.Phivec[findex:findex+2*nfreq] += pcdoubled
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

                        self.Svec += pcdoubled
                        if gibbs_expansion:
                            # Expand in spectrum and correlations
                            self.Scor = corrmat.copy()
                elif m2signal['stype'] == 'spectralModel':
                    Amp = 10**sparameters[0]
                    alpha = sparameters[1]
                    fc = 10**sparameters[2] / pic_spy

                    if m2signal['corr'] == 'single':
                        findex = np.sum(self.npff[:m2signal['pulsarind']])
                        nfreq = int(self.npf[m2signal['pulsarind']]/2)
                        freqpy = self.ptapsrs[m2signal['pulsarind']].Ffreqs
                        pcdoubled = (Amp * pic_spy**3 / m2signal['Tmax']) * ((1 + (freqpy/fc)**2)**(-0.5*alpha))

                        self.Phivec[findex:findex+2*nfreq] += pcdoubled
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

                        self.Svec += pcdoubled
                        if gibbs_expansion:
                            # Expand in spectrum and correlations
                            self.Scor = corrmat.copy()     # Yes, well, there can be only one
                elif m2signal['stype'] == 'frequencyline':
                    # For a frequency line, the FFmatrix is assumed to be set elsewhere
                    findex = np.sum(self.npff[:m2signal['pulsarind']]) + \
                            self.npf[m2signal['pulsarind']] + 2*m2signal['npsrfreqindex']

                    pcdoubled = np.array([sparameters[1], sparameters[1]])

                    self.Phivec[findex:findex+2] += 10**pcdoubled

    def setBeta(self, parameters, selection=None, pp=0):
        """
        Same as constructPhiAndTheta, but now for a single pulsar and only
        band-limited red noise (Beta)
        
        @param parameters:  the full array of parameters
        @param selection::  mask of signals to include
        @param pp:          index of pulsar to do
        """
        psr = self.ptapsrs[pp]

        # Re-set the relevant parts of Theta
        inds = np.sum(self.npfb[:pp])
        inde = inds + self.npfb[pp]
        self.Betavec[inds:inde] = 0.0

        if selection is None:
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # Loop over all signals, and fill the phi matrix
        #for m2signal in self.ptasignals:
        for ss, m2signal in enumerate(self.ptasignals):
            if m2signal['pulsarind'] == pp and selection[ss]:
                # Create a parameters array for this particular signal
                sparameters = m2signal['pstart'].copy()
                sparameters[m2signal['bvary']] = \
                        parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']]
                if m2signal['stype'] == 'blspectrum':
                    # Find the FB index for the band we are in
                    bind = np.where(np.all(psr.Fbands==m2signal['freqband'],axis=1))[0][0]
                    pp = m2signal['pulsarind']
                    findex = np.sum(self.npfb[:pp]) + bind*self.npf[pp]
                    nfreq = int(self.npf[pp]/2)

                    pcdoubled = np.array([sparameters, sparameters]).T.flatten()

                    # Fill the Beta matrix
                    self.Betavec[findex:findex+2*nfreq] += 10**pcdoubled

                elif m2signal['stype'] == 'blpowerlaw':
                    Amp = 10**sparameters[0]
                    Si = sparameters[1]

                    bind = np.where(np.all(psr.Fbands==m2signal['freqband'],axis=1))[0][0]
                    pp = m2signal['pulsarind']
                    findex = np.sum(self.npfb[:pp]) + bind*self.npf[pp]
                    nfreq = int(self.npf[pp]/2)

                    freqpy = self.ptapsrs[pp].Ffreqs * pic_spy
                    pcdoubled = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * m2signal['Tmax'])) * freqpy ** (-Si)
                    # Fill the Beta matrix
                    self.Betavec[findex:findex+2*nfreq] += pcdoubled
                elif m2signal['stype'] == 'blspectralModel':
                    Amp = 10**sparameters[0]
                    alpha = sparameters[1]
                    fc = 10**sparameters[2] / pic_spy

                    #Amp = 10**sparameters[0]
                    #Si = sparameters[1]

                    bind = np.where(np.all(psr.Fbands==m2signal['freqband'],axis=1))[0][0]
                    pp = m2signal['pulsarind']
                    findex = np.sum(self.npfb[:pp]) + bind*self.npf[pp]
                    nfreq = int(self.npf[pp]/2)

                    #freqpy = self.ptapsrs[pp]].Ffreqs * pic_spy
                    #pcdoubled = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * m2signal['Tmax'])) * freqpy ** (-Si)
                    freqpy = self.ptapsrs[pp].Ffreqs
                    pcdoubled = (Amp * pic_spy**3 / m2signal['Tmax']) * ((1 + (freqpy/fc)**2)**(-0.5*alpha))
                    # Fill the Beta matrix
                    self.Betavec[findex:findex+2*nfreq] += pcdoubled


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
                    np.dot(psr.Hmat.T, (psr.Nvec * psr.Hmat.T).T)

            # Create the total GtF and GtD lists for addition of Red(DM) noise
            GtFtot.append(psr.GtF)
            GtDtot.append(psr.GtD)
            GtUtot.append(psr.GtU)
            uvec = np.append(uvec, psr.Jvec)

            self.Gr[gindex:gindex+ng] = np.dot(psr.Hmat.T, psr.detresiduals)

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

        if np.sum(uvec) > 0:
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
            if psr.twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                self.rGr[ii] = np.sum(psr.AGr ** 2 / psr.Nwvec)
                self.GNGldet[ii] = np.sum(np.log(psr.Nwvec))
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
                    Nir = psr.detresiduals / psr.Nvec
                    NiGc = ((1.0/psr.Nvec) * psr.Hcmat.T).T
                    Jldet = np.sum(np.log(psr.Nvec))

                GcNiGc = np.dot(psr.Hcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, psr.detresiduals)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = Jldet + \
                            2*np.sum(np.log(np.diag(cf[0])))
                    GcNiGcr = sl.cho_solve(cf, GcNir)
                except np.linalg.LinAlgError:
                    print "MAJOR ERROR"

                self.rGr[ii] = np.dot(psr.detresiduals, Nir) \
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

        self.constructPhiAndTheta(parameters, make_matrix=(self.likfunc != 'mark3nc'))

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
        if npsrs == 1 or self.likfunc == 'mark3nc':
            mult = 1.0 * int(self.likfunc == 'mark3nc')
            PhiLD = np.sum(np.log(self.Phivec + mult * self.Svec))
            #Phiinv = np.diag(1.0 / self.Phivec)

            SigmaLD = 0.0
            rGSigmaGr = 0.0

            for ii, psr in enumerate(self.ptapsrs):
                findex = np.sum(self.npf[:ii])
                nfreq = int(self.npf[ii]/2)

                slc = slice(findex, findex+2*nfreq)

                di = np.diag_indices(2*nfreq)
                Sigma_psr = self.FGGNGGF[slc, slc].copy()# + Phiinv[slc, slc]
                Sigma_psr[di] += 1.0 / (self.Phivec[slc] + mult*self.Svec[slc])

                try:
                    cf = sl.cho_factor(Sigma_psr)
                    SigmaLD += 2*np.sum(np.log(np.diag(cf[0])))
                    rGSigmaGr += np.dot(self.rGF[slc], sl.cho_solve(cf, self.rGF[slc]))
                except np.linalg.LinAlgError:
                    try:
                        U, s, Vh = sl.svd(Sigma_psr)
                        if not np.all(s > 0):
                            raise ValueError("ERROR: Sigma singular according to SVD")
                        SigmaLD += np.sum(np.log(s))
                        rGSigmaGr += np.dot(self.rGF[slc], np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGF[slc]))))
                    except np.linalg.LinAlgError:
                        return -np.inf
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

            if psr.twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                NGGF = ((1.0/psr.Nwvec) * psr.AGF.T).T

                self.rGr[ii] = np.sum(psr.AGr ** 2 / psr.Nwvec)
                self.rGF[findex:findex+2*nfreq] = np.dot(psr.AGr, NGGF)
                self.GNGldet[ii] = np.sum(np.log(psr.Nwvec))
                self.FGGNGGF[findex:findex+2*nfreq, findex:findex+2*nfreq] = np.dot(psr.AGF.T, NGGF)
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
                            psri.Jvec
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
        self.constructPhiAndTheta(parameters, make_matrix=False)

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


    def mark12loglikelihood(self, parameters):
        """
        mark12 loglikelihood of the pta model/likelihood implementation

        This likelihood uses the T-matrix/Z-matrix definitions, while
        marginalizing over the ecorr/jitter using the Cython jitter extension.
        So... we need the sorted TOAs, obviously

        # For the model, the 'gibbsmodel' description is used
        """
        npsrs = len(self.ptapsrs)

        # The red signals (don't form matrices (slow); use the Gibbs expansion)
        # Is this faster: ?
        #self.setPhi(parameters, gibbs_expansion=True) 
        #self.setTheta(parameters, pp=pp)
        self.constructPhiAndTheta(parameters, make_matrix=False, \
                gibbs_expansion=True)
        self.gibbs_construct_all_freqcov()

        # Band-limited red noise
        for pp, psr in enumerate(self.ptapsrs):
            self.setBeta(parameters, pp=pp)

        # The white noise
        self.setPsrNoise(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # Size of the full matrix is:
        nt = np.sum(self.npz) - np.sum(self.npu)# Size full T-matrix
        nf = np.sum(self.npf)                   # Size full F-matrix
        Sigma = np.zeros((nt, nt))              # Full Sigma matrix
        FPhi = np.zeros((nf, nf))               # RN full Phi matrix
        Finds = np.zeros(nf, dtype=np.int)      # T<->F translation indices
        ZNZ = np.zeros((nt, nt))                # Full ZNZ matrix
        Zx = np.zeros(nt)
        Jldet = np.zeros(npsrs)                 # All log(det(N)) values
        ThetaLD = 0.0
        PhiLD = 0.0
        BetaLD = 0.0
        rGr = np.zeros(npsrs)                   # All rNr values
        sind = 0                                # Sigma-index
        phind = 0                               # Z/T-index

        for ii, psr in enumerate(self.ptapsrs):
            # The T-matrix is just the Z-matrix, minus the ecorr/jitter
            tsize = psr.Zmat.shape[1] - np.sum(psr.Zmask_U)
            Zmat = psr.Zmat[:,:tsize]

            # Use jitter extension for ZNZ
            Jldet[ii], ZNZp = cython_block_shermor_2D(Zmat, psr.Nvec, \
                    psr.Jvec, psr.Uinds)
            Nx = cython_block_shermor_0D(psr.detresiduals, \
                    psr.Nvec, psr.Jvec, psr.Uinds)
            Zx[sind:sind+tsize] = np.dot(Zmat.T, Nx)

            Jldet[ii], rGr[ii] = cython_block_shermor_1D(\
                    psr.detresiduals, psr.Nvec, psr.Jvec, psr.Uinds)

            ZNZ[sind:sind+tsize, sind:sind+tsize] = ZNZp

            # Create the prior (Sigma = ZNZ + Phi)
            nms = self.npm[ii]
            nfs = self.npf[ii]
            nfbs = self.npfb[ii]
            nfdms = self.npfdm[ii]
            findex = np.sum(self.npf[:ii])
            fdmindex = np.sum(self.npfdm[:ii])
            fbindex = np.sum(self.npfb[:ii])
            if 'design' in self.gibbsmodel:
                phind += nms         
            if 'rednoise' in self.gibbsmodel:
                # Red noise, excluding GWS

                if npsrs == 1:
                    inds = slice(sind+phind, sind+phind+nfs)
                    di = np.diag_indices(nfs)
                    Sigma[inds, inds][di] += 1.0 / ( \
                            self.Phivec[findex:findex+nfs] + \
                            self.Svec[findex:findex+nfs])
                    PhiLD += np.sum(np.log(self.Phivec[findex:findex+nfs] + \
                            self.Svec[findex:findex+nfs]))
                    phind += nfs
                elif npsrs > 1:
                    # We need to do the full array at once. Do that below
                    # Here, we construct the indexing matrices
                    Finds[findex:findex+nfs] = np.arange(phind, phind+nfs)
                    phind += nfs
            if 'freqrednoise' in self.gibbsmodel:
                inds = slice(sind+phind, sind+phind+nfbs)
                di = np.diag_indices(nfbs)
                Sigma[inds, inds][di] += 1.0 / self.Betavec[fbindex:fbindex+nfbs]
                BetaLD += np.sum(np.log(self.Betavec[fbindex:fbindex+nfbs]))
                phind += nfbs
            if 'dm' in self.gibbsmodel:
                inds = slice(sind+phind, sind+phind+nfdms)
                di = np.diag_indices(nfdms)
                Sigma[inds, inds][di] += 1.0 / self.Thetavec[fdmindex:fdmindex+nfdms]
                ThetaLD += np.sum(np.log(self.Thetavec[fdmindex:fdmindex+nfdms]))
                phind += nfdms

            sind += tsize

        if npsrs > 1 and 'rednoise' in self.gibbsmodel:
            msk_ind = np.zeros(self.freqmask.shape, dtype=np.int)
            msk_ind[self.freqmask] = np.arange(np.sum(self.freqmask))
            msk_zind = np.arange(np.sum(self.npf))
            for mode in range(0, self.freqmask.shape[1], 2):
                freq = int(mode/2)          # Which frequency

                # We had pre-calculated the Cholesky factor and the inverse
                # (in gibbs_construct_all_freqcov)
                rncov_inv = self.Scor_im_inv[freq]
                cf = self.Scor_im_cf[freq]
                PhiLD += 4 * np.sum(np.log(np.diag(cf[0])))

                # We have the inverse for the individual modes now. Add them to
                # the full prior covariance matrix

                # Firstly the Cosine mode
                newmsk = np.zeros(self.freqmask.shape, dtype=np.bool)
                newmsk[:, mode] = self.freqmask[:, mode]
                mode_ind = msk_ind[newmsk]
                z_ind = msk_zind[mode_ind]
                FPhi[np.array([z_ind]).T, z_ind] += rncov_inv

                # Secondly the Sine mode
                newmsk[:] = False
                newmsk[:, mode+1] = self.freqmask[:, mode+1]
                mode_ind = msk_ind[newmsk]
                z_ind = msk_zind[mode_ind]
                FPhi[np.array([z_ind]).T, z_ind] += rncov_inv

            Sigma[np.array([Finds]).T, Finds] += FPhi

        # If we have a non-trivial prior matrix, invert that stuff
        if 'rednoise' in self.gibbsmodel or \
                'dm' in self.gibbsmodel or \
                'freqrednoise' in self.gibbsmodel:
            Sigma += ZNZ

            # With Sigma constructed, we can invert it
            try:
                cf = sl.cho_factor(Sigma)
                SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
                rSr = np.dot(Zx, sl.cho_solve(cf, Zx))
            except np.linalg.LinAlgError:
                print "Using SVD... return -inf"
                return -np.inf

                U, s, Vh = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")
                SigmaLD = np.sum(np.log(s))
                rSr = np.dot(Nx, np.dot(Vh.T / s, np.dot(U.T, Nx)))
        else:
            SigmaLD = 0.0
            ThetaLD = 0.0
            PhiLD = 0.0
            BetaLD = 0.0
            rSr = 0.0

        return -0.5*np.sum(self.npobs-self.npm)*np.log(2*np.pi) \
                -0.5*np.sum(Jldet) - 0.5*np.sum(rGr) \
                +0.5*rSr - 0.5*SigmaLD - 0.5*PhiLD - 0.5*ThetaLD - 0.5*BetaLD


    # Now we will define the Gibbs likelihood functions, all preceded with
    # gibbs_.... First a few extra auxiliary functions to make that possible

    def gibbs_set_freqb(self, b):
        """
        Instead of having a list with all the red noise Fourier modes, we place them
        in a numpy array here

        @param b:   a list (yes, a list) of numpy arrays of quadratic
                    coefficients. In this case, the Fourier modes
        """
        if len(b) != len(self.ptapsrs):
            raise ValueError('Incompatible mode match')

        for ii in range(self.freqb.shape[0]):
            self.freqb[ii,self.freqmask[ii,:]] = b[ii]

    def gibbs_construct_mode_covariance(self, mode):
        """
        Given the mode number, this function calculates the full-pulsar correlation
        matrix of the red noise/GWs. Data is used from likob.Phivec, and likob.Scor,
        and likob.Svec. Remember that likob.Scor is contains only the correlations
        of one single (the last in the array) signal. So this only works for one
        correlated signal in the model (for now).

        @param mode:    The Fourier mode number
        @param calcInv: Whether we need the inverse matrix, too

        NOTE: Only use Cholesky for now. Do we need mor
        """
        gw_pcdoubled = self.Svec
        msk = self.freqmask[:, mode]

        cov = self.Scor[msk,:][:,msk] * gw_pcdoubled[mode]
        for ii, psr in enumerate(self.ptapsrs):
            ind = np.sum(msk[:ii])
            cov[ind, ind] += self.Phivec[np.sum(self.npf[:ii])+mode]

        return cov

    def gibbs_construct_all_freqcov(self):
        """
        The Scor and Phivec quantities have been set. This function creates the
        Cholesky factors and inverses of all the full-array frequency
        components. All is saved in the lists Scor_im_cf, Scor_im_inv, which are
        lists, with entries for all frequencies (not modes)
        """
        self.Scor_im_cf = []
        self.Scor_im_inv = []

        if 'corrim' in self.gibbsmodel:
            for mode in range(0, self.freqmask.shape[1], 2):
                rncov = self.gibbs_construct_mode_covariance(mode)

                try:
                    self.Scor_im_cf.append(sl.cho_factor(rncov))
                    cf = self.Scor_im_cf[-1]
                    self.Scor_im_inv.append(sl.cho_solve(cf, np.eye(rncov.shape[0])))
                except np.linalg.LinAlgError:
                    print "rncov not positive definite!"
                    raise
        else:
            for mode in range(0, self.freqmask.shape[1], 2):
                rncov = self.gibbs_construct_mode_covariance(mode)

                self.Scor_im_cf.append((np.diag(np.sqrt(np.diag(rncov))), False))
                self.Scor_im_inv.append(np.diag(1.0 / np.diag(rncov)))


    def gibbs_psr_corrs_im(self, psrindex, a):
        """
        Get the Gibbs coefficient quadratic offsets for the correlated signals, for
        a specific pulsar, when the correlated signal is not modelled explicitly
        with it's own Fourier coefficients

        @param psrindex:    Index of the pulsar
        @param a:           List of Gibbs coefficient of all pulsar (of previous step)

        @return:    (pSinv_vec, pPvec), the quadratic offsets
        """
        psr = self.ptapsrs[psrindex]

        #Sinv = []
        pSinv_vec = np.zeros(self.npf[psrindex])
        for ii in range(self.npf[psrindex]):
            mode_ind = int(ii * 0.5)
            pSinv_vec[ii] = self.Scor_im_inv[mode_ind][psrindex, psrindex]

        # The inverse of the GWB correlations are easy
        #pSinv_vec = (1.0 / self.Svec[:self.npf[psrindex]]) * \
        #        self.Scor_inv[psrindex,psrindex]

        # For the quadratic offsets, we'll need to do some splice magic
        # First select the slice we'll need from the correlation matrix
        temp = np.arange(len(self.ptapsrs))
        psrslice = np.delete(temp, psrindex)

        # The quadratic offset we'll return
        pPvec = np.zeros(psr.Fmat.shape[1])

        # Pre-compute the GWB-index offsets of all the pulsars
        corrmode_offset = []
        for ii in range(len(self.ptapsrs)):
            nms = self.npm[ii]
            nfs = self.npf[ii]

            ntot = 0
            if 'design' in self.gibbsmodel:
                ntot += nms

            corrmode_offset.append(ntot)


        for jj in psrslice:
            psrj = self.ptapsrs[jj]
            minfreqs = min(len(psrj.Ffreqs), len(psr.Ffreqs))
            inda = corrmode_offset[jj]
            indb = corrmode_offset[jj] + minfreqs
            #pPvec[:minfreqs] += a[jj][inda:indb] * \
            #        self.Scor_inv[psrindex,jj] / self.Svec[:minfreqs]
            for ii in range(minfreqs):
                mode_ind = int(ii * 0.5)

                #print("pVec[ii]: ", pPvec[ii])
                #print self.Scor_im_inv[mode_ind]
                #print self.Scor_im_inv[mode_ind][psrindex, jj]
                #print a[jj][inda+ii]

                pPvec[ii] += self.Scor_im_inv[mode_ind][psrindex, jj] * a[jj][inda+ii]
                #print "ii/jj = ", ii, jj

        return (pSinv_vec, pPvec)



    def gibbs_get_signal_mask(self, pp, stypes, addFalse=0):
        """
        Given the pulsar number, get the parameter mask for the requested
        parameters

        @param pp:          Pulsar number. -2 means all
        @param stypes:      Which signals to include (e.g. ['efac', 'equad'])
        @param addFalse:    How many entries of False to add at the end

        @return:    Boolean mask that indicates which parameters are selected
        """
        ndims = self.dimensions
        msk = np.zeros(ndims, dtype=np.bool)
        napp = np.zeros(addFalse, dtype=np.bool)

        for ss, m2signal in enumerate(self.ptasignals):
            if (m2signal['pulsarind'] == pp or pp == -2) and \
                    m2signal['stype'] in stypes:
                inds = m2signal['parindex']
                inde = inds + m2signal['npars']
                msk[inds:inde] = True

        return np.append(msk, napp)



    def gibbs_get_custom_subresiduals(self, pp, mask):
        """
        Given the full parameter vector, the pulsar number, and the quadratic
        mask, obtain a custom vector of pre-subtracted residuals

        @param pp:              Pulsar number
        @param mask:            Boolean mask, selecting which quadratic
                                parameters are pre-selected for subtraction

        @return:    Returns the single-pulsar vector of pre-subtracted residuals
        """
        psr = self.ptapsrs[pp]
        #zoffset = np.sum(self.npz[:pp])
        #ndim = self.dimensions
        #qpars = allparameters[ndim+zoffset:ndim+zoffset+self.npz[pp]]
        qpars = self.gibbs_current_a[pp]

        return psr.detresiduals - np.dot(psr.Zmat[:,mask], qpars[mask])


    def gibbs_get_initial_quadratics(self, pp):
        """
        Given a pulsar number, return an initial value for the quadratic
        parameters based on absolutely nothing
        """
        qarr = np.zeros(self.npz[pp])
        ind = 0
        nms = self.npm[pp]
        nfs = self.npf[pp]
        nfdms = self.npfdm[pp]
        npus = self.npu[pp]

        if 'design' in self.gibbsmodel:
            qarr[ind:ind+nms] = 0.0
            ind += nms
        if 'rednoise' in self.gibbsmodel:
            qarr[ind:ind+nfs] = 2.0e-9 * np.random.randn(nfs)
            ind += nfs
        if 'dm' in self.gibbsmodel:
            qarr[ind:ind+nfdms] = 2.0e-9 * np.random.randn(nfdms)
            ind += nfdms
        if 'jitter' in self.gibbsmodel:
            qarr[ind:ind+npus] = 2.0e-9 * np.random.randn(npus)
            ind += npus

        return qarr


    def gibbs_sample_psr_quadratics(self, parameters, b, pp, which='all', \
            ml=False, joinNJ=True):
        """
        Given the values of the hyper parameters, generate new quadratic
        parameters for pulsar pp.

        @param parameters:  The hyper-parameters of the likelihood
        @param b:           List of all quadratic parameters (transformed & overwritten)
        @param pp:          For which pulsar to generate the quadratics
        @param which:       Which quadratics to generate and which to fix
                            (all, F, D, U, M, N)
        @param ml:          Whether to provide ML estimates, or to sample

        @return:            a, b, fulladdcoefficients, xi2
                            (for now, fulladdcoefficients == b[pp])
        """
        npsrs = len(self.ptapsrs)
        xi2 = 0                     # Xi2 for this pulsar

        psr = self.ptapsrs[pp]
        
        # The mask, selecting the quadratic residuals for this pulsar
        dmask = psr.getMmask(which)

        # Indices to keep track of the parameters
        #nms = self.npm[pp]
        nms = np.sum(dmask)
        findex = np.sum(self.npf[:pp])
        nfs = self.npf[pp]
        fdmindex = np.sum(self.npfdm[:pp])
        nfdms = self.npfdm[pp]
        npus = self.npu[pp]

        #residuals = psr.detresiduals.copy()
        # TODO: can we do all this here on the fly from getZmat? Seems shorter:
        #       Zmat, Zmask = psr.getZmat(self.gibbsmodel, which=which)
        if which == 'all':
            Zmat = psr.Zmat
            zmask = np.array([1]*Zmat.shape[1], dtype=np.bool)
        elif which == 'F':
            Zmat = psr.Zmat_F
            zmask = psr.Zmask_F
        elif which == 'D':
            Zmat = psr.Zmat_D
            zmask = psr.Zmask_D
        elif which == 'U':
            Zmat = psr.Zmat_U
            zmask = psr.Zmask_U
        elif which == 'M':
            Zmat = psr.Zmat_M
            zmask = psr.Zmask_M
        elif which == 'N':
            Zmat = psr.Zmat_N
            zmask = psr.Zmask_N

        if np.sum(zmask) == 0:
            # No parameters to fit for
            return self.gibbs_current_a

        # If we do joinNJ, then we should not subtract the jitter/ecorr
        submask = np.logical_not(zmask)
        if joinNJ:
            submask = np.logical_and(submask, np.logical_not(psr.Zmask_U))

        # Form the sub-residuals (only the which-selected ones)
        residuals = self.gibbs_get_custom_subresiduals(pp, submask)

        # Make ZNZ and Sigma
        #ZNZ = np.dot(Zmat.T, ((1.0/psr.Nvec) * Zmat.T).T)
        if joinNJ:
            Jldet, ZNZ = cython_block_shermor_2D(Zmat, psr.Nvec, \
                    psr.Jvec, psr.Uinds)

            # ahat is the slice ML value for the coefficients. Need ENx
            Nx = cython_block_shermor_0D(residuals, \
                    psr.Nvec, psr.Jvec, psr.Uinds)
            ENx = np.dot(Zmat.T, Nx)
        else:
            # We are not including the jitter/ecorr with the white noise
            ZNZ = np.dot(Zmat.T / psr.Nvec, Zmat)
            ENx = np.dot(Zmat.T, residuals / psr.Nvec)

        Sigma = ZNZ.copy()

        # Depending on what signals are in the Gibbs model, we'll have to add
        # prior-covariances to ZNZ
        zindex = 0
        if 'design' in self.gibbsmodel:
            # Do nothing, she'll be 'right
            zindex += nms

        if 'corrim' in self.gibbsmodel:
            if which in ['all', 'F', 'N']:
                ind = range(zindex, zindex + nfs)
                (pSinv_vec, pPvec) = self.gibbs_psr_corrs_im(pp, b)
                Sigma[ind, ind] += pSinv_vec
                ENx[ind] -= pPvec

                zindex += nfs
        elif 'rednoise' in self.gibbsmodel:
            if which in ['all', 'F', 'N']:
                # Don't do this if it is included in corrim
                ind = range(zindex, zindex+nfs)
                Sigma[ind, ind] += 1.0 / self.Phivec[findex:findex+nfs]

                zindex += nfs

        if 'dm' in self.gibbsmodel:
            if which in ['all', 'D', 'N']:
                ind = range(zindex, zindex+nfdms)
                Sigma[ind, ind] += 1.0 / self.Thetavec[fdmindex:fdmindex+nfdms]

                zindex += nfdms

        if 'jitter' in self.gibbsmodel:
            if which in ['all', 'U']:
                ind = range(zindex, zindex+npus)
                Sigma[ind, ind] += 1.0 / psr.Jvec

                zindex += npus

        if 'correx' in self.gibbsmodel:
            if self.have_gibbs_corr and which in ['all', 'F', 'N']:
                (pSinv_vec, pPvec) = gibbs_psr_corrs_ex(self, pp, b)

                ind = range(zindex, zindex + nfs)
                Sigma[ind, ind] += pSinv_vec
                ENx[ind] -= pPvec

                zindex += nfs

        try:
            # Use a QR decomposition for the inversions
            Qs,Rs = sl.qr(Sigma) 

            #Qsb = np.dot(Qs.T, np.eye(Sigma.shape[0])) # computing Q^T*b (project b onto the range of A)
            #Sigi = sl.solve(Rs,Qsb) # solving R*x = Q^T*b
            Sigi = sl.solve(Rs,Qs.T) # solving R*x = Q^T*b
            
            # Ok, we've got the inverse... now what? Do SVD?
            U, s, Vt = sl.svd(Sigi)
            Li = U * np.sqrt(s)

            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")

            ahat = np.dot(Sigi, ENx)

            # Test the coefficients for nans and infs
            nonan = np.all(np.logical_not(np.isnan(ahat)))
            noinf = np.all(np.logical_not(np.isinf(ahat)))
            if not (nonan and noinf):
                np.savetxt('ahat.txt', ahat)
                raise ValueError("Have inf or nan in solution (QR)")
        except (np.linalg.LinAlgError, ValueError):
            try:
                print "ERROR in QR. Doing SVD"

                U, s, Vt = sl.svd(Sigma)
                if not np.all(s > 0):
                    raise np.linalg.LinAlgError
                    #raise ValueError("ERROR: Sigma singular according to SVD")
                Sigi = np.dot(U * (1.0/s), Vt)
                Li = U * (1.0 / np.sqrt(s))

                ahat = np.dot(Sigi, ENx)
            except np.linalg.LinAlgError:
                try:
                    print "ERROR in SVD. Doing Cholesky"

                    cfL = sl.cholesky(Sigma, lower=True)
                    cf = (cfL, True)

                    # Calculate the inverse Cholesky factor (can we do this faster?)
                    cfLi = sl.cho_factor(cfL, lower=True)
                    Li = sl.cho_solve(cfLi, np.eye(Sigma.shape[0]))

                    ahat = sl.cho_solve(cf, ENx)
                except np.linalg.LinAlgError:
                    # Come up with some better exception handling
                    print "ERROR in Cholesky. Help!"
                    raise
        except ValueError:
            print "WTF?"
            print "Look in sigma.txt for the Sigma matrix"
            np.savetxt("sigma.txt", Sigma)
            np.savetxt("phivec.txt", self.Phivec[findex:findex+nfs])
            np.savetxt('nvec.txt', psr.Nvec)

            np.savetxt("znz.txt", ZNZ)
            np.savetxt("ENx.txt", ENx)

            raise

        # Get a sample from the coefficient distribution
        aadd = np.dot(Li, np.random.randn(Li.shape[0]))

        if ml:
            addcoefficients = ahat
        else:
            addcoefficients = ahat + aadd

        # Test the coefficients for nans and infs
        nonan = np.all(np.logical_not(np.isnan(addcoefficients)))
        noinf = np.all(np.logical_not(np.isinf(addcoefficients)))
        if not (nonan and noinf):
            np.savetxt('ahat.txt', ahat)
            np.savetxt('aadd.txt', aadd)
            raise ValueError("Have inf or nan in solution")

        b[pp][zmask] = addcoefficients.copy()
        psr.gibbscoefficients = b[pp]

        a = b[pp].copy()
        if 'design' in self.gibbsmodel:
            a[:psr.Mmat.shape[1]] = np.dot(psr.tmpConv, b[pp][:psr.Mmat.shape[1]])

        # This is mostly for the mark1 Gibbs sampler. RvH: can we get rid of
        # this, by just using:
        # > psr.gibbsresiduals = self.gibbs_get_custom_subresiduals(ii, psr.Zmask_N)
        # > psr.gibbssubresiduals = psr.detresiduals - psr.gibbsresiduals
        # in the mark1 Gibbs sampler, instead of insisting that the 'jitter
        # block' is directly after the quadratic sample block? Note that all the
        # individual samplers also set their data though gibbsresiduals, so it
        # is not straightforward.
        if which == 'N':
            # When which == 'N', we are doing this as part of the joint N-J
            # analysis. So we are not subtracting the jitter/ecorr just yet.
            # So this assumes we will update these values later, right after the
            # N-J conditional
            psr.gibbssubresiduals = np.dot(Zmat, addcoefficients)
            psr.gibbsresiduals = psr.detresiduals - psr.gibbssubresiduals
        else:
            psr.gibbssubresiduals = np.dot(psr.Zmat, psr.gibbscoefficients)
            psr.gibbsresiduals = psr.detresiduals - psr.gibbssubresiduals

        #xi2[ii] = np.sum(psr.gibbsresiduals**2 / psr.Nvec)
        # ZNZ = python_block_shermor_2d(psr.Zmat, psr.Nvec, psr.Jvec, psr.Uinds)
        tmp, xi2 = cython_block_shermor_1D(psr.gibbsresiduals, \
                psr.Nvec, psr.Jvec, psr.Uinds)

        return a, b, xi2



    def gibbs_sample_Phi_quadratics(self, a, ml=False, \
            sigma_precalc=False):
        """
        Assume that the covariances have been set by the hyper-parameters
        already. Generate new quadratic parameters for the full array. This
        function will keep the quadratic parameters for the Jitter and DM
        variation fixed

        @param a:               List of all quadratic parameters (overwritten)
        @param ml:              Whether to provide ML estimates, or to sample
        @param sigma_precalc:   When True, do not recalculate Sigma and
                                Sigma_inv

        @return:                a
        """
        npsrs = len(self.ptapsrs)
        xi2 = 0                     # Xi2 for this pulsar
        beSlow = False       # Do not use frequency optimisation (yet)

        # Create the quadratic mask list (which quadratics of which pulsar are
        # included
        l_amask = []

        for ii, psr in enumerate(self.ptapsrs):
            l_amask.append(psr.Zmask_F)

        if not sigma_precalc:
            # Set the FNF matrix
            for ii, psr in enumerate(self.ptapsrs):
                residuals = self.gibbs_get_custom_subresiduals(ii, \
                        np.logical_not(l_amask[ii]))

                zindex = np.sum(self.npz_f[:ii])
                npz = int(self.npz_f[ii])

                Jldet, self.FNF[zindex:zindex+npz, zindex:zindex+npz] = \
                        cython_block_shermor_2D(psr.Zmat_F, psr.Nvec, psr.Jvec, psr.Uinds)
                #self.FNF[zindex:zindex+npz, zindex:zindex+npz] = \
                #        np.dot(psr.Zmat_F.T, ((1.0/psr.Nvec) * psr.Zmat_F.T).T)

                Nx = cython_block_shermor_0D(residuals, \
                        psr.Nvec, psr.Jvec, psr.Uinds)
                self.rGZ_F[zindex:zindex+npz] = np.dot(Nx, psr.Zmat_F)
                #self.rGZ_F[zindex:zindex+npz] = np.dot(residuals / psr.Nvec, psr.Zmat_F)

                self.GNGldet[ii], self.rGr[ii] = cython_block_shermor_1D(\
                        residuals, psr.Nvec, psr.Jvec, psr.Uinds)
                #self.GNGldet[ii] = np.sum(np.log(psr.Nvec))
                #self.rGr[ii] = np.sum(residuals ** 2 / psr.Nvec)

            # Using that, build Sigma
            if len(self.ptapsrs) == 1:
                psr = self.ptapsrs[0]

                # No fancy tricks required
                Phiinv = 1.0 / (self.Phivec + self.Svec)

                Zmat = psr.Zmat_F

                PhiLD = np.sum(np.log(self.Phivec + self.Svec))

                #Sigma = np.dot(Zmat.T * (1.0 / psr.Nvec), Zmat)
                Sigma = self.FNF.copy()

                inds = range(Zmat.shape[1] - psr.Fmat.shape[1], \
                        Zmat.shape[1])
                Sigma[inds, inds] += Phiinv
            else:
                # We need to do some index magic in this section with masks and such
                # These are just the indices of the frequency matrix
                msk_ind = np.zeros(self.freqmask.shape, dtype=np.int)
                msk_ind[self.freqmask] = np.arange(np.sum(self.freqmask))

                # Transform these indices to the full Z-matrix (no npff here?)
                # This includes the design matrix
                # 
                # Z = (M1  F1   0   0 ... 0   0  )
                #     ( 0   0  M2  F2 ... 0   0  )
                #     ( .   .   .   . ... .   .  )
                #     ( 0   0   0   0 ... Mn  Fn )
                moffset = np.repeat(np.cumsum(self.npm_f), self.npf, axis=0)
                msk_zind = np.arange(np.sum(self.npf)) + moffset

                Sigma = self.FNF.copy()

                if beSlow:
                    try:
                        cf = sl.cho_factor(self.Phi)
                        Phiinv = sl.cho_solve(cf, np.eye(self.Phi.shape[0]))
                        PhiLD = 2*np.sum(np.log(np.diag(cf[0])))

                        Sigma[np.array([msk_zind]).T, msk_zind] += Phiinv
                    except np.linalg.LinAlgError:
                        raise
                else:
                    # Sigma = np.dot(Zmat.T * (1.0 / psr.Nvec), Zmat)


                    # Perform the inversion of Phi per frequency. Is much much faster
                    PhiLD = 0
                    for mode in range(0, self.freqmask.shape[1], 2):
                        freq = int(mode/2)

                        # We had pre-calculated the Cholesky factor and the inverse
                        rncov_inv = self.Scor_im_inv[freq]
                        cf = self.Scor_im_cf[freq]
                        PhiLD += 4 * np.sum(np.log(np.diag(cf[0])))

                        # Ok, we have the inverse for the individual modes. Now add them
                        # to the full sigma matrix

                        # Firstly the Cosine mode
                        newmsk = np.zeros(self.freqmask.shape, dtype=np.bool)
                        newmsk[:, mode] = self.freqmask[:, mode]
                        mode_ind = msk_ind[newmsk]
                        z_ind = msk_zind[mode_ind]
                        Sigma[np.array([z_ind]).T, z_ind] += rncov_inv

                        # Secondly the Sine mode
                        newmsk[:] = False
                        newmsk[:, mode+1] = self.freqmask[:, mode+1]
                        mode_ind = msk_ind[newmsk]
                        z_ind = msk_zind[mode_ind]
                        Sigma[np.array([z_ind]).T, z_ind] += rncov_inv

            # Aight, we have Sigma now. Do the decomposition
            try:
                # Use a QR decomposition for the inversions
                Qs,Rs = sl.qr(Sigma) 

                #Qsb = np.dot(Qs.T, np.eye(Sigma.shape[0])) # computing Q^T*b (project b onto the range of A)
                #Sigi = sl.solve(Rs,Qsb) # solving R*x = Q^T*b
                Sigi = sl.solve(Rs,Qs.T) # solving R*x = Q^T*b
                
                # Ok, we've got the inverse... now what? Do SVD?
                U, s, Vt = sl.svd(Sigi)
                Li = U * np.sqrt(s)

                if not np.all(s > 0):
                    raise ValueError("ERROR: Sigma singular according to SVD")

                ahat = np.dot(Sigi, self.rGZ_F)

                # Test the coefficients for nans and infs
                nonan = np.all(np.logical_not(np.isnan(ahat)))
                noinf = np.all(np.logical_not(np.isinf(ahat)))
                if not (nonan and noinf):
                    np.savetxt('ahat.txt', ahat)
                    raise ValueError("Have inf or nan in solution (QR)")
            except (np.linalg.LinAlgError, ValueError):
                try:
                    print "ERROR in QR. Doing SVD"

                    U, s, Vt = sl.svd(Sigma)
                    if not np.all(s > 0):
                        raise np.linalg.LinAlgError
                        #raise ValueError("ERROR: Sigma singular according to SVD")
                    Sigi = np.dot(U * (1.0/s), Vt)
                    Li = U * (1.0 / np.sqrt(s))

                    ahat = np.dot(Sigi, self.rGZ_F)
                except np.linalg.LinAlgError:
                    try:
                        print "ERROR in SVD. Doing Cholesky"

                        cfL = sl.cholesky(Sigma, lower=True)
                        cf = (cfL, True)

                        # Calculate the inverse Cholesky factor (can we do this faster?)
                        cfLi = sl.cho_factor(cfL, lower=True)
                        Li = sl.cho_solve(cfLi, np.eye(Sigma.shape[0]))

                        ahat = sl.cho_solve(cf, self.rGZ_F)
                    except np.linalg.LinAlgError:
                        # Come up with some better exception handling
                        print "ERROR in Cholesky. Help!"
                        raise
            except ValueError:
                print "WTF?"
                print "Look in sigma.txt for the Sigma matrix"
                np.savetxt("sigma.txt", Sigma)

                raise
        else:
            # We do have Sigma_F_cf pre-calculated
            Li = sl.lapack.clapack.dtrtri(self.Sigma_F_cf[0])[0]
            ahat = sl.cho_solve(self.Sigma_F_cf, self.rGZ_F)

        # Get a sample from the coefficient distribution
        aadd = np.dot(Li, np.random.randn(Li.shape[0]))
        # See what happens if we use numpy
        # aadd = np.random.multivariate_normal(np.zeros(Sigi.shape[0]), \
        #        Sigi)
        #numpy.random.multivariate_normal(mean, cov[, size])

        if ml:
            addcoefficients = ahat
        else:
            addcoefficients = ahat + aadd

        # Test the coefficients for nans and infs
        nonan = np.all(np.logical_not(np.isnan(addcoefficients)))
        noinf = np.all(np.logical_not(np.isinf(addcoefficients)))
        if not (nonan and noinf):
            np.savetxt('ahat.txt', ahat)
            np.savetxt('aadd.txt', aadd)
            raise ValueError("Have inf or nan in solution")


        # Place back the quadratic parameters
        #l_amask.append(np.zeros(nzs, dtype=np.bool))
        pindex = 0
        for pp, psr in enumerate(self.ptapsrs):
            npz = self.npz[pp]
            a[pp][l_amask[pp]] = addcoefficients[pindex:pindex+self.npz_f[pp]]

            psr.gibbscoefficients[l_amask[pp]] = a[pp][l_amask[pp]]

            pindex += self.npz_f[pp]

        return a


    def gibbs_sample_Theta_quadratics(self, a, pp, ml=False):
        """
        RvH:    why do we not just use the regular sampler for this? I recall
                only using this one for debugging purposes.
        """

        psr = self.ptapsrs[pp]
        Zmat = psr.Zmat_D.copy()
        zmask = psr.Zmask_D.copy()
        dmask = psr.getMmask('D').copy()

        ###
        #Zmat = psr.Zmat.copy()
        #zmask[:] = True
        #dmask[:] = True
        ###

        residuals = self.gibbs_get_custom_subresiduals(pp, np.logical_not(zmask))

        Jldet, ZNZ = cython_block_shermor_2D(Zmat, \
                psr.Nvec, psr.Jvec, psr.Uinds)
        #ZNZ = np.dot(Zmat.T, np.dot(np.diag(1.0/psr.Nvec), Zmat))

        inds = range(np.sum(dmask), Zmat.shape[1])
        ZNZ[inds,inds] += 1.0 / self.Thetavec

        # RvH: Why was 'residuals' actuall detresiduals????
        Nx = cython_block_shermor_0D(residuals, psr.Nvec, \
                psr.Jvec, psr.Uinds)
        rGZ_F = np.dot(Nx, Zmat)
        #rGZ_F = np.dot(psr.detresiduals / psr.Nvec, Zmat)

        try:
            cf = (sl.cholesky(ZNZ), False)
            Li = sl.lapack.clapack.dtrtri(cf[0])[0]

            ahat = sl.cho_solve(cf, rGZ_F)
        except LinAlgError:
            U, s, Vt = sl.svd(ZNZ)
            Li = np.dot(Vt.T * (1.0/np.sqrt(s)), U.T)
            ahat = np.dot(Vt.T * (1.0/s), np.dot(U.T, rGZ_F))
        aadd = np.dot(Li, np.random.randn(Li.shape[0]))
        coeffs = ahat + 1.0*aadd

        a[pp][zmask] = coeffs

        return a


    def gibbs_sample_M_quadratics(self, a, pp, ml=False):
        """
        RvH:    why do we not just use the regular sampler for this? I recall
                only using this one for debugging purposes.
        """

        psr = self.ptapsrs[pp]
        Zmat = psr.Zmat_M.copy()
        zmask = psr.Zmask_M.copy()

        residuals = self.gibbs_get_custom_subresiduals(pp, np.logical_not(zmask))

        Jldet, ZNZ = cython_block_shermor_2D(Zmat, psr.Nvec, \
                psr.Jvec, psr.Uinds)
        #ZNZ = np.dot(Zmat.T, np.dot(np.diag(1.0/psr.Nvec), Zmat))

        # RvH: Why was 'residuals' actuall detresiduals????
        Nx = cython_block_shermor_0D(residuals, psr.Nvec, \
                psr.Jvec, psr.Uinds)
        rGZ_F = np.dot(Nx, Zmat)
        #rGZ_F = np.dot(psr.detresiduals / psr.Nvec, Zmat)

        try:
            cf = (sl.cholesky(ZNZ), False)
            Li = sl.lapack.clapack.dtrtri(cf[0])[0]

            ahat = sl.cho_solve(cf, rGZ_F)
        except LinAlgError:
            U, s, Vt = sl.svd(ZNZ)
            Li = np.dot(Vt.T * (1.0/np.sqrt(s)), U.T)
            ahat = np.dot(Vt.T * (1.0/s), np.dot(U.T, rGZ_F))

        aadd = np.dot(Li, np.random.randn(Li.shape[0]))
        coeffs = ahat + aadd

        a[pp][zmask] = coeffs

        return a







    def gibbs_full_loglikelihood(self, aparameters, resetCorInv=True, \
            which='all', pp=-1, updateResiduals=True):
        """
        Within the Gibbs sampler, we would still like to have access to the
        loglikelihood value, even though it is not necessarily used for the sampling
        itself. This function evaluates the ll.

        This function does not set any of the noise/correlation auxiliaries. It
        assumes that has been done earlier in the Gibbs step. Also, it assumes the
        Gibbsresiduals have been set properly.

        NOTE:   the timing model parameters are assumed to be in the basis of Gcmat,
                not Mmat. This for numerical stability (really doesn't work
                otherwise).

        @param aparameters: All the model parameters, including the quadratic pars
        @param coeffs:      List of all the Gibbs coefficients per pulsar
        @param resetCorInv: Re-evaluate the Scor_im_cf and Scor_im_inv lists
        @param which:       Which components of the likelihood need to be
                            evaluated (all, F, D, N, U)
        @param pp:          If D/N/U, for which pulsar do we evaluate it?
        @param updateResiduals:     Update the subtracted residuals (e.g.
                                    gibbssubresiduals_F)

        @return:            The log-likelihood
        """

        ndim = self.dimensions     # This does not include the quadratic parameters
        quadparind = ndim + 0       # Index of quadratic parameters

        # Now we also know the position of the hyper parameters
        allparameters = aparameters.copy()
        parameters = allparameters[:ndim]

        # Set the white noise
        self.setPsrNoise(parameters)

        # Set the red noise / DM correction quantities. Use the Gibbs expansion to
        # build the per-frequency Phi-matrix
        if which == 'all':
            self.constructPhiAndTheta(parameters, make_matrix=False, \
                    noise_vec=True, gibbs_expansion=True)
        elif which == 'N':
            pass
        elif which == 'U':
            pass
        elif which == 'F':
            self.setPhi(parameters, gibbs_expansion=True)
        elif which == 'D':
            if pp > 0:
                self.setTheta(parameters, pp=pp)
            else:
                # This can be optimized!
                self.constructPhiAndTheta(parameters, make_matrix=False, \
                        noise_vec=True, gibbs_expansion=True)

        if self.haveDetSources:
            # Provide a switch for this
            self.updateDetSources(parameters)

        if resetCorInv and (which == 'all' or which == 'F') and \
                'rednoise' in self.gibbsmodel:
            self.gibbs_construct_all_freqcov()

        ksi = []        # Timing model parameters
        a = []          # Red noise / GWB Fourier modes
        d = []          # DM variation Fourier modes
        j = []          # Jitter/epochave residuals

        for ii, psr in enumerate(self.ptapsrs):
            nzs = self.npz[ii]
            nms = self.npm[ii]
            findex = np.sum(self.npf[:ii])
            nfs = self.npf[ii]
            fdmindex = np.sum(self.npfdm[:ii])
            nfdms = self.npfdm[ii]
            npus = self.npu[ii]

            ntot = 0
            nqind = quadparind + 0
            nqind_m = -1
            nqind_f = -1
            nqind_dm = -1
            nqind_u = -1
            if 'design' in self.gibbsmodel:
                #allparameters[nqind:nqind+nms] = np.dot(psr.tmpConvi, allparameters[nqind:nqind+nms])
                ksi.append(allparameters[nqind:nqind+nms])
                ntot += nms
                nqind_m = nqind
                nqind += nms
            if 'rednoise' in self.gibbsmodel:
                a.append(allparameters[nqind:nqind+nfs])
                ntot += nfs
                nqind_f = nqind
                nqind += nfs
            if 'dm' in self.gibbsmodel:
                d.append(allparameters[nqind:nqind+nfdms])
                ntot += nfdms
                nqind_dm = nqind
                nqind += nfdms
            if 'jitter' in self.gibbsmodel:
                j.append(allparameters[nqind:nqind+npus])
                ntot += npus
                nqind_u = nqind
                nqind += npus

            # Calculate the quadratic parameter subtracted residuals
            #zmask = np.array([1]*psr.Zmat.shape[1], dtype=np.bool)
            #self.gibbsresiduals = self.gibbs_get_custom_subresiduals(ii, zmask)
            self.gibbsresiduals = psr.detresiduals - \
                    np.dot(psr.Zmat, self.gibbs_current_a[ii])

            quadparind += ntot

        # Now evaluate the various quadratic forms
        xi2 = 0
        ldet = 0
        for ii, psr in enumerate(self.ptapsrs):
            # The quadratic form of the residuals
            jldet, jxi2 = cython_block_shermor_1D(psr.gibbsresiduals, \
                    psr.Nvec, psr.Jvec, psr.Uinds)
            xi2 += jxi2
            ldet += jldet
            self.gibbs_ll_N[ii] = 0.5*(jxi2 + jldet)

            #nx2 = np.sum(psr.gibbsresiduals ** 2 / psr.Nvec)
            #nld = np.sum(np.log(psr.Nvec))
            #self.gibbs_ll_N[ii] = 0.5*(nx2 + nld)
            #xi2 += nx2
            #ldet += nld

            # Jitter is done per pulsar
            if 'jitter' in self.gibbsmodel:
                if which in ['all', 'U']:
                    jx2 = np.sum(j[ii] ** 2 / psr.Jvec)
                    jld = np.sum(np.log(psr.Jvec))
                    self.gibbs_ll_U[ii] = 0.5*(jx2 + jld)
                    xi2 += jx2
                    ldet += jld
                else:
                    xi2 += 2 * self.gibbs_ll_U[ii]

            # Quadratic form of DM variations, for full array
            if 'dm' in self.gibbsmodel:
                if which in ['all', 'D']:
                    #dx2 = np.sum(np.hstack(d)**2 / self.Thetavec)
                    #dld = np.sum(np.log(self.Thetavec))
                    inds = np.sum(self.npfdm[:ii])
                    inde = inds + self.npfdm[ii]
                    dx2 = np.sum(d[ii]**2 / self.Thetavec[inds:inde])
                    dld = np.sum(np.log(self.Thetavec[inds:inde]))
                    xi2 += dx2
                    ldet += dld
                    self.gibbs_ll_D[ii] = 0.5*(dx2 + dld)
                else:
                    xi2 += 2*self.gibbs_ll_D[ii]


        # Quadratic form of red noise, done for full array
        if 'rednoise' in self.gibbsmodel:
            if which in ['all', 'F']:
                fx2 = 0
                fld = 0

                # Do some fancy stuff here per frequency
                if len(self.ptapsrs) > 1:
                    # Loop over all frequencies
                    for ii in range(0, len(self.Svec), 2):
                        msk = self.freqmask[:, ii]

                        # The covariance between sin/cos modes is identical
                        #cov = self.gibbs_construct_mode_covariance(ii)
                        #cf = sl.cho_factor(cov)
                        cf = self.Scor_im_cf[int(ii / 2)]

                        # Cosine mode
                        bc = self.freqb[msk, ii]
                        Lx = sl.cho_solve(cf, bc)
                        xi2 += np.sum(Lx**2)
                        ldet += 2*np.sum(np.log(np.diag(cf[0])))

                        # Sine mode
                        bs = self.freqb[msk, ii+1]
                        Lx = sl.cho_solve(cf, bs)
                        fx2 += np.sum(Lx**2)
                        fld += 2*np.sum(np.log(np.diag(cf[0])))
                else:
                    # Single pulsar. Just combine the correlated and noise frequencies
                    pcd = self.Phivec + self.Svec
                    fx2 += np.sum(np.hstack(a)**2 / pcd)
                    fld += np.sum(np.log(pcd))

                self.gibbs_ll_F = 0.5*(fx2 + fld)
                xi2 += fx2
                ldet += fld
            else:
                xi2 += 2 * self.gibbs_ll_F

        return -0.5*np.sum(self.npobs)*np.log(2*np.pi) - 0.5*xi2 - 0.5*ldet





    def gibbs_psr_noise_loglikelihood(self, parameters, pp, mask, allpars, \
            joinNJ=True, gibbs_iter=-1):
        """
        The conditional loglikelihood for the subset of white noise parameters
        (EFAC and EQUAD, and possibly jitter later on).

        @param parameters:      The hyper-parameter array
        @param pp:              Index of the pulsar we are treating
        @param mask:            The mask to use for the full set of parameters
        @param allpars:         The vector of all hyper paramertes parmaeters
        @param joinNJ:          If True, don't subtract jitter from residuals
        @param gibbs_iter:      Which iteration in Gibbs (so we don't have to
                                re-calculate the resiudals every time)
        """
        psr = self.ptapsrs[pp]

        # Set the parameters
        apars = allpars.copy()
        apars[mask] = parameters
        self.setSinglePsrNoise_fast(apars, pp=pp, joinNJ=joinNJ)

        if psr.gibbs_N_iter != gibbs_iter:
            # Decide which signals to subtract from the 'detresiduals'
            if joinNJ:
                zmask = np.logical_not(psr.Zmask_U)
            else:
                zmask = np.array([1]*Zmat.shape[1], dtype=np.bool)

            # Subtract the signals from the residuals
            residuals = self.gibbs_get_custom_subresiduals(pp, zmask)
            psr.gibbs_N_residuals = residuals
            psr.gibbs_N_iter = gibbs_iter
        else:
            residuals = psr.gibbs_N_residuals

        # Calculate the block-inverse in the Cython jitter extension module
        jldet, jxi2 = cython_block_shermor_1D(residuals, \
                psr.Nvec, psr.Jvec * np.float(joinNJ), psr.Uinds)

        return -0.5 * jxi2 - 0.5 * jldet


    def gibbs_psr_DM_loglikelihood_mar(self, parameters, pp, mask, allpars):
        """
        The conditional loglikelihood for the subset of DM hyper-parameters. It
        assumes the red noise and quadratic parameters are kept fixed. The DM
        coefficients and the timing model parameters are analytically
        marginalised over. The rest of the coefficients are subtracted from the
        residuals already.

        @param parameters:      The hyper-parameter array
        @param pp:              Index of the pulsar we are treating
        @param mask:            The mask to use for the full set of parameters
        @param allpars:         The vector of all hyper parameters parmaeters
        """
        psr = self.ptapsrs[pp]

        # Obtain the Thetavec indices we need
        inds = np.sum(self.npfdm[:pp])
        inde = inds + self.npfdm[pp]

        # Set the parameters
        apars = allpars.copy()
        apars[mask] = parameters
        self.setTheta(apars, pp=pp)

        # Do not subtract ecorr/jitter, since it's included in the jitter
        # extension
        submask = np.logical_not(psr.Zmask_D)
        submask = np.logical_and(submask, np.logical_not(psr.Zmask_U))

        # Form the sub-residuals (only DM variation residuals)
        residuals = self.gibbs_get_custom_subresiduals(pp, submask)

        # Set the DND matrix (Likelihood)
        Jldet, DND = cython_block_shermor_2D(psr.Zmat_D, psr.Nvec, psr.Jvec, psr.Uinds)

        Nx = cython_block_shermor_0D(residuals, \
                psr.Nvec, psr.Jvec, psr.Uinds)
        rGZ_D = np.dot(Nx, psr.Zmat_D)

        Jldet, rGr = cython_block_shermor_1D(residuals, psr.Nvec, \
                psr.Jvec, psr.Uinds)

        # Set the Sigma matrix (prior)
        indss = range(psr.Zmat_D.shape[1] - psr.DF.shape[1], \
                psr.Zmat_D.shape[1])
        ThetaLD = np.sum(np.log(self.Thetavec[inds:inde]))
        Sigma = DND.copy()
        Sigma[indss, indss] += 1.0 / self.Thetavec[inds:inde]

        # Decompose Sigma for Woodbury, and return the likelihood
        try:
            cf = sl.cho_factor(Sigma)
            SigmaLD = 2*np.sum(np.log(np.diag(cf[0])))
            rGSigmaGr = np.dot(rGZ_D, sl.cho_solve(cf, rGZ_D))
        except np.linalg.LinAlgError:
            U, s, Vh = sl.svd(Sigma)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(rGZ_D, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, rGZ_D))))

        # Return the conditional marginalised log-likelihood
        return -0.5*rGr - 0.5*Jldet \
                + 0.5*rGSigmaGr - 0.5*SigmaLD - 0.5*ThetaLD


    def gibbs_Phi_loglikelihood_mar(self, parameters, mask, allpars):
        """
        The conditional loglikelihood for the subset of red-noise/GWB
        hyper-parameters. It assumes the DMV parameters are kept fixed. The red
        noise/GWB coefficients and the timing model parameters are analytically
        marginalised over. The rest of the coefficients are subtracted from the
        residuals already.

        @param parameters:      The hyper-parameter array
        @param mask:            The mask to use for the full set of parameters
        @param allpars:         The vector of all hyper parameters parmaeters
        """
        beSlow = False       # Do not use frequency optimisation (True if testing)

        # Set the parameters
        apars = allpars.copy()[:self.dimensions]
        apars[mask] = parameters

        if beSlow:
            self.constructPhiAndTheta(apars, make_matrix=True)
        else:
            self.setPhi(apars, gibbs_expansion=True) 
            self.gibbs_construct_all_freqcov()

        # Set the FNF matrix
        for ii, psr in enumerate(self.ptapsrs):
            # Do not subtract ecorr/jitter, since it's included in the jitter
            # extension
            submask = np.logical_not(psr.Zmask_F)
            submask = np.logical_and(submask, np.logical_not(psr.Zmask_U))

            # Form the sub-residuals (only red-noise/signal residuals)
            residuals = self.gibbs_get_custom_subresiduals(ii, submask)

            zindex = np.sum(self.npz_f[:ii])
            npz = int(self.npz_f[ii])
            self.GNGldet[ii], self.FNF[zindex:zindex+npz, zindex:zindex+npz] = \
                    cython_block_shermor_2D(psr.Zmat_F, \
                    psr.Nvec, psr.Jvec, psr.Uinds)
            #self.FNF[zindex:zindex+npz, zindex:zindex+npz] = \
            #        np.dot(psr.Zmat_F.T, ((1.0/psr.Nvec) * psr.Zmat_F.T).T)
            Nx = cython_block_shermor_0D(residuals, psr.Nvec, \
                    psr.Jvec, psr.Uinds)
            self.rGZ_F[zindex:zindex+npz] = np.dot(Nx, psr.Zmat_F)
            # self.rGZ_F[zindex:zindex+npz] = np.dot(residuals / psr.Nvec, psr.Zmat_F)

            self.GNGldet[ii], self.rGr[ii] = cython_block_shermor_1D(\
                    residuals, psr.Nvec, psr.Jvec, psr.Uinds)
            #self.GNGldet[ii] = np.sum(np.log(psr.Nvec))
            #self.rGr[ii] = np.sum(residuals ** 2 / psr.Nvec)
            

        if len(self.ptapsrs) == 1:
            psr = self.ptapsrs[0]

            # No fancy tricks required
            Phivec = self.Phivec + self.Svec
            Phiinv = 1.0 / Phivec

            Zmat = psr.Zmat_F

            PhiLD = np.sum(np.log(self.Phivec + self.Svec))

            self.Sigma_F = self.FNF.copy()
            inds = range(Zmat.shape[1] - psr.Fmat.shape[1], \
                    Zmat.shape[1])
            self.Sigma_F[inds, inds] += Phiinv
        else:
            # We need to do some index magic in this section with masks and such
            # These are just the indices of the frequency matrix
            msk_ind = np.zeros(self.freqmask.shape, dtype=np.int)
            msk_ind[self.freqmask] = np.arange(np.sum(self.freqmask))

            # Transform these indices to the full Z-matrix (no npff here?)
            # This includes the design matrix
            # 
            # Z = (M1  F1   0   0 ... 0   0  )
            #     ( 0   0  M2  F2 ... 0   0  )
            #     ( .   .   .   . ... .   .  )
            #     ( 0   0   0   0 ... Mn  Fn )
            moffset = np.repeat(np.cumsum(self.npm_f), self.npf, axis=0)
            msk_zind = np.arange(np.sum(self.npf)) + moffset

            self.Sigma_F = self.FNF.copy()

            if beSlow:
                try:
                    cf = sl.cho_factor(self.Phi)
                    Phiinv = sl.cho_solve(cf, np.eye(self.Phi.shape[0]))
                    PhiLD = 2*np.sum(np.log(np.diag(cf[0])))

                    self.Sigma_F[np.array([msk_zind]).T, msk_zind] += Phiinv
                except np.linalg.LinAlgError:
                    raise
                
            else:
                # Perform the inversion of Phi per frequency. Is much much faster
                PhiLD = 0.0
                for mode in range(0, self.freqmask.shape[1], 2):
                    freq = int(mode/2)

                    # We had pre-calculated the Cholesky factor and the inverse
                    rncov_inv = self.Scor_im_inv[freq]
                    cf = self.Scor_im_cf[freq]
                    PhiLD += 4 * np.sum(np.log(np.diag(cf[0])))

                    # Ok, we have the inverse for the individual modes. Now add them
                    # to the full sigma matrix

                    # Firstly the Cosine mode
                    newmsk = np.zeros(self.freqmask.shape, dtype=np.bool)
                    newmsk[:, mode] = self.freqmask[:, mode]
                    mode_ind = msk_ind[newmsk]
                    z_ind = msk_zind[mode_ind]
                    self.Sigma_F[np.array([z_ind]).T, z_ind] += rncov_inv

                    # Secondly the Sine mode
                    newmsk[:] = False
                    newmsk[:, mode+1] = self.freqmask[:, mode+1]
                    mode_ind = msk_ind[newmsk]
                    z_ind = msk_zind[mode_ind]
                    self.Sigma_F[np.array([z_ind]).T, z_ind] += rncov_inv

        # With Sigma constructed, we can invert it
        try:
            self.Sigma_F_cf = (sl.cholesky(self.Sigma_F), False)
            SigmaLD = 2*np.sum(np.log(np.diag(self.Sigma_F_cf[0])))
            rGSigmaGr = np.dot(self.rGZ_F, sl.cho_solve(self.Sigma_F_cf, self.rGZ_F))
        except np.linalg.LinAlgError:
            print "Using SVD... return -inf"
            return -np.inf
            #raise RuntimeError("Using SVD")

            U, s, Vh = sl.svd(self.Sigma_F)
            if not np.all(s > 0):
                raise ValueError("ERROR: Sigma singular according to SVD")
            SigmaLD = np.sum(np.log(s))
            rGSigmaGr = np.dot(self.rGZ_F, np.dot(Vh.T, np.dot(np.diag(1.0/s), np.dot(U.T, self.rGZ_F))))
        except ValueError:
            print self.Sigma_F, self.Phivec, self.Svec, parameters
            print "prior:", self.gibbs_Phi_logprior(parameters, mask, allpars),\
                    "for: ", apars
            np.savetxt("Sigma.txt", self.Sigma_F)
            np.savetxt("Phivec.txt", self.Phivec)
            np.savetxt("Svec.txt", self.Svec)
            raise

        #print "rGr = ", np.sum(self.rGr)
        #print "GNGldet = ", np.sum(self.GNGldet)
        #print "rGSigmaGr = ", rGSigmaGr
        #print "SigmaLD = ", SigmaLD
        #print "PhiLD = ", PhiLD

        #print "ll = ", ll

        # Return the conditional marginalised log-likelihood
        return -0.5*np.sum(self.rGr) - 0.5*np.sum(self.GNGldet) \
                + 0.5*rGSigmaGr - 0.5*SigmaLD - 0.5*PhiLD




    def gibbs_psr_noise_logprior(self, parameters, pp, mask, allpars, \
            joinNJ=True, gibbs_iter=-1):
        apars = allpars.copy()
        apars[mask] = parameters
        return self.logprior(apars[:self.dimensions])

    def gibbs_psr_DM_logprior(self, parameters, pp, mask, allpars):
        apars = allpars.copy()
        apars[mask] = parameters
        return self.logprior(apars[:self.dimensions])

    def gibbs_psr_J_logprior(self, parameters, pp, mask, allpars):
        apars = allpars.copy()
        apars[mask] = parameters
        return self.logprior(apars[:self.dimensions])


    def gibbs_Phi_logprior(self, parameters, mask, allpars):
        apars = allpars.copy()
        apars[mask] = parameters
        return self.logprior(apars[:self.dimensions])



    def loglikelihood(self, parameters):
        ll = 0.0

        if(np.all(self.pmin <= parameters) and np.all(parameters <= self.pmax)):
            if self.likfunc == 'mark1':
                ll = self.mark1loglikelihood(parameters)
            elif self.likfunc == 'mark2':
                ll = self.mark2loglikelihood(parameters)
            elif self.likfunc in ['mark3', 'mark3nc']:
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
            elif self.likfunc == 'mark12':
                ll = self.mark12loglikelihood(parameters)

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
            if "prior" in m2signal:
                if m2signal["prior"] in ['linear', 'flat'] and m2signal['stype'] != 'spectrum':
                    # The prior has been set
                    lp += np.log(10)*parameters[m2signal['parindex']]
                elif m2signal["prior"] in ['linear', 'flat']:
                    lp += np.log(10)*np.sum(parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']])
            elif m2signal['stype'] == 'powerlaw' and m2signal['corr'] == 'anisotropicgwb':
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
            #elif m2signal['stype'] == 'powerlaw' and m2signal['corr'] != 'single':
            #    lp += parameters[m2signal['parindex']]
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
            #elif m2signal['stype'] == 'spectrum' and m2signal['corr'] != 'single':
            #    lp += np.sum(parameters[m2signal['parindex']:m2signal['parindex']+m2signal['npars']])

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
            elif self.likfunc in ['mark3', 'mark3nc']:
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
            elif self.likfunc == 'mark12':
                lp = self.mark4logprior(parameters)
            elif self.likfunc == 'gibbs':
                lp = self.mark4logprior(parameters)
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
                h5df.addData(psr.name, 'prefitRes', psr.residuals[psr.iisort], overwrite=True)
                h5df.addData(psr.name, 'postfitRes', psr.residuals[psr.iisort], overwrite=True)

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
                h5df.addData(psr.name, 'prefitRes', psr.residuals[psr.iisort], overwrite=True)
                h5df.addData(psr.name, 'postfitRes', psr.residuals[psr.iisort], overwrite=True)


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
            tmpdelta = chi[index:index+np.sum(ind)]

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
    Same as the original, but also calculates the TMPs

    TODO: Gr does not include detresiduals!!!! FIX THIS
    """
    def mlPredictionFilter3(self, mlparameters, signum=None, selection=None):
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
        Cfull = np.zeros((np.sum(self.npobs), np.sum(self.npobs)))
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

            Cfull[nindex:nindex+nobs,nindex:nindex+nobs] = \
                    np.diag(psr.Nvec)

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

            Cfull[nindex:nindex+nobs,nindex:nindex+nobs] += \
                    np.dot(DF, (allThetavec[fdmindex:fdmindex+2*nfreqdm] * DF).T)

        # Now add the red signals, too
        GCGfull += np.dot(totGF, np.dot(allPhi, totGF.T))
        Cpred += np.dot(totF, np.dot(predPhi, totF.T))
        GtCpred = np.dot(totG.T, Cpred)
        Cfull += np.dot(totF, np.dot(allPhi, totF.T))

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

        # Calculate the ML TMPs
        try:
            if len(self.ptapsrs) > 1:
                raise ValueError("ERROR: too many pulsars!")
            psr = self.ptapsrs[0]

            cf = sl.cho_factor(Cfull)
            Ci = sl.cho_solve(cf, np.eye(Cfull.shape[0]))
            MCM = np.dot(psr.Mmat.T, np.dot(Ci, psr.Mmat))
            cf = sl.cho_factor(MCM)

            MCx = np.dot(psr.Mmat.T, np.dot(Ci, psr.prefitresiduals))
            MCMi = sl.cho_solve(cf, np.eye(MCM.shape[0]))
            chiML = np.dot(MCMi, MCx)
            chiMLerr = np.sqrt(np.diag(MCMi))
        except np.linalg.LinAlgError:
            raise ValueError("ERROR: C not Cholesky-factorisable")

        Cti = np.dot(totG, GCGr)
        recsig = np.dot(Cpred, Cti)

        CtGCp = np.dot(Cpred, np.dot(totG, GCGCp))
        recsigCov = Cpred - CtGCp

        return recsig, np.sqrt(np.diag(recsigCov)), chiML, chiMLerr






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
