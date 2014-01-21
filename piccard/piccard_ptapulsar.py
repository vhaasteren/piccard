#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
piccard_ptapulsar.py

"""


from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss

from .piccard_datafile import DataFile
from .piccard_ptafuncs import *
from .piccard_constants import *


class ptaPulsar(object):
    """ Represents a pulsar, the pulsar observations, and auxiliary quantities

    The ptaPulsar class is a basic quantity in Piccard that completely describes
    a pulsar dataset. It contains many auxiliary quantities that are necessary
    for teh evaluation of the various likelihood functions.

    All the attributes will be written as properties. Initially the ptaPulsar
    object will be initialised from an HDF5/othertype datafile, which may
    contain the auxiliary quantities pre-computed. The attributes will then,
    when the quantity is not pre-loaded first check the HDF5 file if it exists.
    If it does exist, it will be loaded. If it doesn't, it will be computed.

    """
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
    freqs = None
    unitconversion = None
    Gmat = None
    Gcmat = None
    Mmat = None
    ptmpars = []
    ptmparerrs = []
    ptmdescription = []
    flags = None
    name = "J0000+0000"

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
    avetoas = None          
    SFdmmat = None         # Fdmmatrix for the dm frequency lines
    #FFdmmat = None         # Total of SFdmmatrix and Fdmmat
    Dmat = None
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
    Gr = None
    GGr = None
    GtF = None
    GtD = None
    GGtD = None
    AGr = None      # Replaces GGr in 2-component noise model
    AoGr = None     #   Same but for orthogonal basis (when compressing)
    AGF = None      # Replaces GGtF in 2-component noise model
    AoGF = None     #   Same but for orthogonal basis (when compressing)
    AGD = None      # Replaces GGtD in 2-component noise model
    AoGD = None     #   Same but for orthogonal basis (when compressing)
    AGE = None      # Replaces GGtE in 2-component noise model
    AoGE = None     #   Same but for orthogonal basis (when compressing)
    AGU = None      # Replace GGtU in 2-component noise model
    AoGU = None     #   Same .... you got it

    # Auxiliaries used in the likelihood
    twoComponentNoise = False       # Whether we use the 2-component noise model
    Nvec = None             # The total white noise (eq^2 + ef^2*err)
    Wvec = None             # The weights in 2-component noise
    Wovec = None            # The weights in 2-component orthogonal noise
    Nwvec = None            # Total noise in 2-component basis (eq^2 + ef^2*Wvec)
    Nwovec = None           # Total noise in 2-component orthogonal basis

    # To select the number of Frequency modes
    bfinc = None        # Number of modes of all internal matrices
    bfdminc = None      # Number of modes of all internal matrices (DM)
    bcurfinc = None     # Current number of modes in RJMCMC
    bcurfdminc = None   # Current number of modes in RJMCMC

    Qam = 0.0           # The pulse Jitter amplitude (if we use it)

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
        self.unitconversion = None
        self.Gmat = None
        self.Gcmat = None
        self.Mmat = None
        self.ptmpars = []
        self.ptmparerrs = []
        self.ptmdescription = []
        self.flags = None
        self.name = "J0000+0000"

        self.Fmat = None
        self.SFmat = None
        self.FFmat = None
        self.Fdmmat = None
        self.Hmat = None
        self.Homat = None
        self.Hcmat = None
        self.Hocmat = None
        self.Umat = None
        self.avetoas = None
        self.Dmat = None
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
        self.GGtD = None

        self.bfinc = None
        self.bfdminc = None
        self.bprevfinc = None
        self.bprevfdminc = None

        self.Qam = 0.0

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
            oldunitconversion=None, noWarning=False):
        if oldMmat is None:
            oldMmat = self.Mmat
        if oldptmdescription is None:
            oldptmdescription = self.ptmdescription
        if oldptmpars is None:
            oldptmpars = self.ptmpars
        if oldunitconversion is None:
            oldunitconversion = self.unitconversion
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
            newunitconversion = np.append(oldunitconversion, addunitvals)

            # Construct the G-matrices
            U, s, Vh = sl.svd(newM)
            newG = U[:, (newM.shape[1]):].copy()
            newGc = U[:, :(newM.shape[1])].copy()
        else:
            newM = oldMmat.copy()
            newptmdescription = np.array(oldptmdescription)
            newunitconversion = np.array(oldunitconversion)
            newptmpars = oldptmpars.copy()
            newG = oldGmat.copy()

            newGc = oldGcmat.copy()

        return newM, newG, newGc, newptmpars, map(str, newptmdescription), newunitconversion

    """
    Consgtructs a new modified design matrix by deleting some columns from it.
    Returns a list of new objects that represent the new timing model

    @param delpars:     Names of the parameters/columns that need to be deleted.
    
    @return (list):     Return the elements: (newM, newG, newGc,
                        newptmpars, newptmdescription, newunitconversion)
                        in order: the new design matrix, the new G-matrix, the
                        new co-Gmatrix (orthogonal complement), the new values
                        of the timing model parameters, the new descriptions of
                        the timing model parameters. Note that the timing model
                        parameters are not really 'new', just re-selected
    """
    def delFromDesignMatrix(self, delpars, \
            oldMmat=None, oldGmat=None, oldGcmat=None, \
            oldptmpars=None, oldptmdescription=None, \
            oldunitconversion=None):
        if oldMmat is None:
            oldMmat = self.Mmat
        if oldptmdescription is None:
            oldptmdescription = self.ptmdescription
        if oldunitconversion is None:
            oldunitconversion = self.unitconversion
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
            newunitconversion = np.array(oldunitconversion)[indkeep]
            newptmpars = oldptmpars[indkeep]

            # Construct the G-matrices
            U, s, Vh = sl.svd(newM)
            newG = U[:, (newM.shape[1]):].copy()
            newGc = U[:, :(newM.shape[1])].copy()
        else:
            newM = oldMmat.copy()
            newptmdescription = np.array(oldptmdescription)
            newunitconversion = oldunitconversion.copy()
            newptmpars = oldptmpars.copy()
            newG = oldGmat.copy()
            newGc = oldGcmat.copy()

        return newM, newG, newGc, newptmpars, map(str, newptmdescription), newunitconversion


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
        
        (newM, newG, newGc, newptmpars, newptmdescription, newunitconversion) = \
                (self.Mmat, self.Gmat, self.Gcmat, self.ptmpars, \
                self.ptmdescription, self.unitconversion)

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
            (newM, newG, newGc, newptmpars, newptmdescription, newunitconversion) = \
                    self.addToDesignMatrix(addpar, newM, newG, newGc, \
                    newptmpars, newptmdescription, newunitconversion, \
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
            (newM, newG, newGc, newptmpars, newptmdescription, newunitconversion) = \
                    self.delFromDesignMatrix(delpar, newM, newG, newGc, \
                    newptmpars, newptmdescription, newunitconversion)

        return newM, newG, newGc, newptmpars, newptmdescription, newunitconversion




    # Modify the design matrix to include fitting for a quadratic in the DM
    # signal.
    # TODO: Check if the DM is fit for in the design matrix. Use ptmdescription
    #       for that. It should have a field with 'DM' in it.
    def addDMQuadratic(self):
        self.Mmat, self.Gmat, self.Gcmat, self.ptmpars, \
                self.ptmdescription, self.unitconversion = \
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

            # We will do a weighted fit
            w = 1.0/self.toaerrs**0
            # Create the weighted projection matrix (oblique projection)
            UWU = np.dot(self.Umat.T, (w * self.Umat.T).T)
            cf = sl.cho_factor(UWU)
            UWUi = sl.cho_solve(cf, np.eye(UWU.shape[0]))
            P = np.dot(self.Umat, np.dot(UWUi, self.Umat.T * w))
            #PuG = self.Gmat
            PuG = np.dot(P, self.Gmat)
            GU = np.dot(PuG.T, self.Umat)
            GUUG = np.dot(GU, GU.T)

            """
            # Build a projection matrix for U
            Pu = np.dot(self.Umat, Ui)
            PuG = np.dot(Pu, self.Gmat)
            Vmat, svec, Vhsvd = sl.svd(np.dot(self.Gmat.T, PuG))

            UU = np.dot(self.Umat.T, self.Umat)
            cf = sl.cho_factor(UU)
            UUi = sl.cho_solve(cf, np.eye(UU.shape[0]))
            Pu = np.dot(self.Umat, np.dot(UUi, self.Umat.T))

            # This assumes that self.Umat has already been set
            #GU = np.dot(self.Gmat.T, self.Umat)
            #GUUG = np.dot(GU, GU.T)
            GUUG = np.dot(self.Gmat.T, np.dot(Pu, self.Gmat))
            """

            """
            GU = np.dot(self.Gmat.T, self.Umat)
            GUUG = np.dot(GU, GU.T)
            """

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
            (newM, newG, newGc, newptmpars, newptmdescription, newunitconversion) = \
                    self.getModifiedDesignMatrix(removeAll=True)
            self.Hmat = newG
            self.Hcmat = newGc
            self.Homat = np.zeros((self.Hmat.shape[0], 0))      # There is no complement
            self.Hocmat = np.zeros((self.Hmat.shape[0], 0))
        elif compression == 'timingmodel':
            tmpardel = self.getNewTimingModelParameterList(keep=False, tmpars=tmpars)

            (newM, newG, newGc, newptmpars, newptmdescription, newunitconversion) = \
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
            tmsigpars=None):
        # For creating the auxiliaries it does not really matter: we are now
        # creating all quantities per default
        # TODO: set this parameter in another place?
        if twoComponent:
            self.twoComponentNoise = True

        # Before writing anything to file, we need to know right away how many
        # fixed and floating frequencies this model contains.
        nf = 0 ; ndmf = 0 ; nsf = nSingleFreqs ; nsdmf = nSingleDMFreqs
        if nfreqs is not None and nfreqs != 0:
            nf = nfreqs
        if ndmfreqs is not None and ndmfreqs != 0:
            ndmf = ndmfreqs

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
            h5df.addData(self.name, 'pic_Tmax', [Tmax])

        # Create the Fourier design matrices for noise
        if nf > 0:
            (self.Fmat, self.Ffreqs) = fourierdesignmatrix(self.toas, 2*nf, Tmax)
        else:
            self.Fmat = np.zeros((len(self.toas), 0))
            self.Ffreqs = np.zeros(0)

        # Create the Fourier design matrices for DM variations
        if ndmf > 0:
            (self.Fdmmat, self.Fdmfreqs) = fourierdesignmatrix(self.toas, 2*ndmf, Tmax)
            self.Dmat = np.diag(pic_DMk / (self.freqs**2))
            #self.DF = np.dot(self.Dmat, self.Fdmmat)
            self.DF = (np.diag(self.Dmat) * self.Fdmmat.T).T
        else:
            self.Fdmmat = np.zeros((len(self.freqs), 0))
            self.Fdmfreqs = np.zeros(0)
            self.Dmat = np.diag(pic_DMk / (self.freqs**2))
            self.DF = np.zeros((len(self.freqs), 0))

        # Create the dailay averaged residuals
        (self.avetoas, self.Umat) = dailyaveragequantities(self.toas)

        # Write these quantities to disk
        if write != 'no':
            h5df.addData(self.name, 'pic_Fmat', self.Fmat)
            h5df.addData(self.name, 'pic_Ffreqs', self.Ffreqs)
            h5df.addData(self.name, 'pic_Fdmmat', self.Fdmmat)
            h5df.addData(self.name, 'pic_Fdmfreqs', self.Fdmfreqs)
            h5df.addData(self.name, 'pic_Dmat', self.Dmat)
            h5df.addData(self.name, 'pic_DF', self.DF)

            h5df.addData(self.name, 'pic_avetoas', self.avetoas)
            h5df.addData(self.name, 'pic_Umat', self.Umat)

        # Next we'll need the G-matrices, and the compression matrices.
        U, s, Vh = sl.svd(self.Mmat)
        self.Gmat = U[:, self.Mmat.shape[1]:].copy()
        self.Gcmat = U[:, :self.Mmat.shape[1]].copy()

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
            h5df.addData(self.name, 'pic_Gmat', self.Gmat)
            h5df.addData(self.name, 'pic_Gcmat', self.Gcmat)
            h5df.addData(self.name, 'pic_Hmat', self.Hmat)
            h5df.addData(self.name, 'pic_Hcmat', self.Hcmat)
            h5df.addData(self.name, 'pic_Homat', self.Homat)
            h5df.addData(self.name, 'pic_Hocmat', self.Hocmat)



        # Now, write such quantities on a per-likelihood basis
        if likfunc == 'mark1' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            self.GtD = np.dot(self.Hmat.T, self.DF)

            # For two-component noise
            # Diagonalise GtEfG (HtEfH)
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
                self.AoGF = np.dot(self.Aomat.T, HotF)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGF = np.zeros((0, self.GtF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_GtD', self.GtD)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGF', self.AGF)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGF', self.AoGF)

        if likfunc == 'mark2' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)

            # For two-component noise
            # Diagonalise GtEfG
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
                self.AoGF = np.dot(self.Aomat.T, HotF)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGF = np.zeros((0, self.GtF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGF', self.AGF)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGF', self.AoGF)

        if likfunc == 'mark4' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            GtU = np.dot(self.Hmat.T, self.Umat)

            self.UtF = np.dot(self.Umat.T, self.Fmat)

            # For two-component noise
            # Diagonalise GtEfG
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
                self.AoGU = np.dot(self.Aomat.T, HotU)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGU = np.zeros((0, GtU.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_UtF', self.UtF)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGU', self.AGU)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGU', self.AoGU)


        if likfunc == 'mark4ln' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            GtU = np.dot(self.Hmat.T, self.Umat)

            self.UtF = np.dot(self.Umat.T, self.Fmat)

            # Initialise the single frequency with a frequency of 10 / yr
            self.frequencyLinesAdded = nSingleFreqs
            deltaf = 2.3 / pic_spy      # Just some random number
            sfreqs = np.linspace(deltaf, 5.0*deltaf, nSingleFreqs)
            self.SFmat = singleFreqFourierModes(self.toas, np.log10(sfreqs))
            self.FFmat = np.append(self.Fmat, self.SFmat, axis=1)
            self.SFfreqs = np.log10(np.array([sfreqs, sfreqs]).T.flatten())

            self.UtFF = np.dot(self.Umat.T, self.FFmat)

            # For two-component noise
            # Diagonalise GtEfG
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
                self.AoGU = np.dot(self.Aomat.T, HotU)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGU = np.zeros((0, GtU.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_UtF', self.UtF)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat)
                h5df.addData(self.name, 'pic_FFmat', self.FFmat)
                h5df.addData(self.name, 'pic_SFfreqs', self.SFfreqs)
                h5df.addData(self.name, 'pic_UtFF', self.UtFF)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGU', self.AGU)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGU', self.AoGU)

        if likfunc == 'mark6' or likfunc == 'mark6fa' or write == 'all':
            # Red noise
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)
            #self.GGtF = np.dot(self.Hmat, self.GtF)

            # DM
            GtD = np.dot(self.Hmat.T, self.DF)
            self.GGtD = np.dot(self.Hmat, GtD)

            # DM + Red noise stuff (mark6 needs this)
            self.Emat = np.append(self.Fmat, self.DF, axis=1)
            GtE = np.dot(self.Hmat.T, self.Emat)
            self.GGtE = np.dot(self.Hmat, GtE)

            # For two-component noise
            # Diagonalise GtEfG
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
                HotF = np.dot(self.Homat.T, self.Fmat)
                HotD = np.dot(self.Homat.T, self.DF)
                HotE = np.dot(self.Homat.T, self.Emat)
                self.AoGr = np.dot(self.Aomat.T, Hor)
                self.AoGF = np.dot(self.Aomat.T, HotF)
                self.AoGD = np.dot(self.Aomat.T, HotD)
                self.AoGE = np.dot(self.Aomat.T, HotE)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGF = np.zeros((0, self.GtF.shape[1]))
                self.AoGD = np.zeros((0, GtD.shape[1]))
                self.AoGE = np.zeros((0, GtE.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat)
                h5df.addData(self.name, 'pic_GGtE', self.GGtE)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGF', self.AGF)
                h5df.addData(self.name, 'pic_AGD', self.AGD)
                h5df.addData(self.name, 'pic_AGE', self.AGE)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                h5df.addData(self.name, 'pic_AoGD', self.AoGD)
                h5df.addData(self.name, 'pic_AoGE', self.AoGE)

        if likfunc == 'mark7' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # For two-component noise
            # Diagonalise GtEfG
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
                self.AoGF = np.dot(self.Aomat.T, HotF)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGF = np.zeros((0, self.GtF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGF', self.AGF)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGF', self.AoGF)

        if likfunc == 'mark8' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # For the DM stuff
            GtD = np.dot(self.Hmat.T, self.DF)
            self.GGtD = np.dot(self.Hmat, GtD)

            # DM + Red noise stuff
            self.Emat = np.append(self.Fmat, self.DF, axis=1)
            GtE = np.dot(self.Hmat.T, self.Emat)
            self.GGtE = np.dot(self.Hmat, GtE)

            # For two-component noise
            # Diagonalise GtEfG
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
                HotF = np.dot(self.Homat.T, self.Fmat)
                HotD = np.dot(self.Homat.T, self.DF)
                HotE = np.dot(self.Homat.T, self.Emat)
                self.AoGr = np.dot(self.Aomat.T, Hor)
                self.AoGF = np.dot(self.Aomat.T, HotF)
                self.AoGD = np.dot(self.Aomat.T, HotD)
                self.AoGE = np.dot(self.Aomat.T, HotE)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGF = np.zeros((0, self.GtF.shape[1]))
                self.AoGD = np.zeros((0, GtD.shape[1]))
                self.AoGE = np.zeros((0, GtE.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat)
                h5df.addData(self.name, 'pic_GGtE', self.GGtE)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGF', self.AGF)
                h5df.addData(self.name, 'pic_AGD', self.AGD)
                h5df.addData(self.name, 'pic_AGE', self.AGE)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                h5df.addData(self.name, 'pic_AoGD', self.AoGD)
                h5df.addData(self.name, 'pic_AoGE', self.AoGE)

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
                HotF = np.dot(self.Homat.T, self.Fmat)
                HotFF = np.dot(self.Homat.T, self.FFmat)
                self.AoGr = np.dot(self.Aomat.T, Hor)
                self.AoGF = np.dot(self.Aomat.T, HotF)
                self.AoGFF = np.dot(self.Aomat.T, HotFF)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGF = np.zeros((0, self.GtF.shape[1]))
                self.AoGFF = np.zeros((0, GtFF.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat)
                h5df.addData(self.name, 'pic_FFmat', self.FFmat)
                h5df.addData(self.name, 'pic_SFfreqs', self.SFfreqs)
                h5df.addData(self.name, 'pic_Wvec', self.Wvec)
                h5df.addData(self.name, 'pic_Amat', self.Amat)
                h5df.addData(self.name, 'pic_AGr', self.AGr)
                h5df.addData(self.name, 'pic_AGF', self.AGF)
                h5df.addData(self.name, 'pic_AGFF', self.AGFF)
                h5df.addData(self.name, 'pic_Wovec', self.Wovec)
                h5df.addData(self.name, 'pic_Aomat', self.Aomat)
                h5df.addData(self.name, 'pic_AoGr', self.AoGr)
                h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                h5df.addData(self.name, 'pic_AoGFF', self.AoGFF)

        if likfunc == 'mark10' or write == 'all':
            self.Gr = np.dot(self.Hmat.T, self.residuals)
            self.GGr = np.dot(self.Hmat, self.Gr)
            self.GtF = np.dot(self.Hmat.T, self.Fmat)

            # For the DM stuff
            GtD = np.dot(self.Hmat.T, self.DF)
            self.GGtD = np.dot(self.Hmat, GtD)

            # DM + Red noise stuff (mark6 needs this)
            self.Emat = np.append(self.Fmat, self.DF, axis=1)
            GtE = np.dot(self.Hmat.T, self.Emat)
            self.GGtE = np.dot(self.Hmat, GtE)

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
            self.DSF = np.dot(self.Dmat, self.SFdmmat)
            self.DFF = np.append(self.DF, self.DSF, axis=1)

            GtFF = np.dot(self.Hmat.T, self.FFmat)

            self.EEmat = np.append(self.FFmat, self.DFF, axis=1)
            GtEE = np.dot(self.Hmat.T, self.EEmat)
            self.GGtEE = np.dot(self.Hmat, GtEE)
            
            # For two-component noise
            # Diagonalise GtEfG
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
                HotF = np.dot(self.Homat.T, self.Fmat)
                HotFF = np.dot(self.Homat.T, self.FFmat)
                HotD = np.dot(self.Homat.T, self.DF)
                HotE = np.dot(self.Homat.T, self.Emat)
                HotEE = np.dot(self.Homat.T, self.EEmat)
                self.AoGr = np.dot(self.Aomat.T, Hor)
                self.AoGF = np.dot(self.Aomat.T, HotF)
                self.AoGFF = np.dot(self.Aomat.T, HotFF)
                self.AoGD = np.dot(self.Aomat.T, HotD)
                self.AoGE = np.dot(self.Aomat.T, HotE)
                self.AoGEE = np.dot(self.Aomat.T, HotEE)
            else:
                self.Wovec = np.zeros(0)
                self.Aomat = np.zeros((self.Amat.shape[0], 0))
                self.AoGr = np.zeros((0, self.Gr.shape[0]))
                self.AoGF = np.zeros((0, self.GtF.shape[1]))
                self.AoGFF = np.zeros((0, GtFF.shape[1]))
                self.AoGD = np.zeros((0, GtD.shape[1]))
                self.AoGE = np.zeros((0, GtE.shape[1]))
                self.AoGEE = np.zeros((0, GtEE.shape[1]))

            if write != 'none':
                # Write all these quantities to the HDF5 file
                h5df.addData(self.name, 'pic_Gr', self.Gr)
                h5df.addData(self.name, 'pic_GGr', self.GGr)
                h5df.addData(self.name, 'pic_GtF', self.GtF)
                h5df.addData(self.name, 'pic_GGtD', self.GGtD)
                h5df.addData(self.name, 'pic_Emat', self.Emat)
                h5df.addData(self.name, 'pic_GGtE', self.GGtE)
                h5df.addData(self.name, 'pic_SFmat', self.SFmat)
                h5df.addData(self.name, 'pic_SFdmmat', self.SFdmmat)
                h5df.addData(self.name, 'pic_FFmat', self.FFmat)
                h5df.addData(self.name, 'pic_SFfreqs', self.SFfreqs)
                h5df.addData(self.name, 'pic_DSF', self.DSF)
                h5df.addData(self.name, 'pic_DFF', self.DFF)
                h5df.addData(self.name, 'pic_EEmat', self.EEmat)
                h5df.addData(self.name, 'pic_GGtEE', self.GGtEE)
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
                h5df.addData(self.name, 'pic_AoGF', self.AoGF)
                h5df.addData(self.name, 'pic_AoGFF', self.AoGFF)
                h5df.addData(self.name, 'pic_AoGD', self.AoGD)
                h5df.addData(self.name, 'pic_AoGE', self.AoGE)
                h5df.addData(self.name, 'pic_AoGEE', self.AoGEE)



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
            compression='None', likfunc='mark3', memsave=True):
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
            raise ValueError("File frequencies are not compatible with model frequencies")
        # Ok, this model seems good to go. Let's start

        # G/H compression matrices
        self.Gmat = np.array(h5df.getData(self.name, 'pic_Gmat', dontread=memsave))
        self.Gcmat = np.array(h5df.getData(self.name, 'pic_Gcmat', dontread=memsave))
        self.Hmat = np.array(h5df.getData(self.name, 'pic_Hmat'))
        self.Hcmat = np.array(h5df.getData(self.name, 'pic_Hcmat'))
        self.Homat = np.array(h5df.getData(self.name, 'pic_Homat'))
        self.Hocmat = np.array(h5df.getData(self.name, 'pic_Hocmat'))
        self.Gr = np.array(h5df.getData(self.name, 'pic_Gr', dontread=memsave))
        self.GGr = np.array(h5df.getData(self.name, 'pic_GGr', dontread=memsave))
        self.Wvec = np.array(h5df.getData(self.name, 'pic_Wvec'))
        self.Wovec = np.array(h5df.getData(self.name, 'pic_Wovec'))
        self.Amat = np.array(h5df.getData(self.name, 'pic_Amat', dontread=memsave))
        self.Aomat = np.array(h5df.getData(self.name, 'pic_Aomat', dontread=memsave))
        self.AoGr = np.array(h5df.getData(self.name, 'pic_AoGr'))
        self.Ffreqs = np.array(h5df.getData(self.name, 'pic_Ffreqs'))
        self.Fdmfreqs = np.array(h5df.getData(self.name, 'pic_Fdmfreqs'))

        # If compression is not done, but Hmat represents a compression matrix,
        # we need to re-evaluate the lot. Raise an error
        if (compression == 'None' or compression is None) and \
                h5df.getShape(self.name, 'pic_Gmat')[1] != \
                h5df.getShape(self.name, 'pic_Hmat')[1]:
            raise ValueError("Compressed file detected. Re-calculating all quantities.")
        elif (compression != 'None' and compression != None) and \
                h5df.getShape(self.name, 'pic_Gmat')[1] == \
                h5df.getShape(self.name, 'pic_Hmat')[1]:
            raise ValueError("Uncompressed file detected. Re-calculating all quantities.")

        if likfunc == 'mark1':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF'))
            self.GtD = np.array(h5df.getData(self.name, 'pic_GtD'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr', dontread=memsave))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF', dontread=memsave))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF', dontread=memsave))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat', dontread=memsave))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', dontread=memsave))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark2':
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark3' or likfunc == 'mark3fa':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF'))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF', dontread=memsave))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark4':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.UtF = np.array(h5df.getData(self.name, 'pic_UtF'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGU = np.array(h5df.getData(self.name, 'pic_AGU'))
            self.AoGU = np.array(h5df.getData(self.name, 'pic_AoGU', dontread=memsave))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))
            self.Umat = np.array(h5df.getData(self.name, 'pic_Umat'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))

        if likfunc == 'mark4ln':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.UtF = np.array(h5df.getData(self.name, 'pic_UtF', dontread=memsave))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', dontread=memsave))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', dontread=memsave))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.UtFF = np.array(h5df.getData(self.name, 'pic_UtFF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGU = np.array(h5df.getData(self.name, 'pic_AGU'))
            self.AoGU = np.array(h5df.getData(self.name, 'pic_AoGU', dontread=memsave))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))
            self.Umat = np.array(h5df.getData(self.name, 'pic_Umat'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))

        if likfunc == 'mark6' or likfunc == 'mark6fa':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.GGtD = np.array(h5df.getData(self.name, 'pic_GGtD', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat'))
            self.GGtE = np.array(h5df.getData(self.name, 'pic_GGtE', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF', dontread=memsave))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD', dontread=memsave))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE'))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF', dontread=memsave))
            self.AoGD = np.array(h5df.getData(self.name, 'pic_AoGD', dontread=memsave))
            self.AoGE = np.array(h5df.getData(self.name, 'pic_AoGE', dontread=memsave))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat', dontread=memsave))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF', dontread=memsave))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat', dontread=memsave))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark7':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF', dontread=memsave))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF', dontread=memsave))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark8':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.GGtD = np.array(h5df.getData(self.name, 'pic_GGtD', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat', dontread=memsave))
            self.GGtE = np.array(h5df.getData(self.name, 'pic_GGtE', dontread=memsave))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF', dontread=memsave))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD', dontread=memsave))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE', dontread=memsave))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF', dontread=memsave))
            self.AoGD = np.array(h5df.getData(self.name, 'pic_AoGD', dontread=memsave))
            self.AoGE = np.array(h5df.getData(self.name, 'pic_AoGE', dontread=memsave))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat'))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark9':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', dontread=memsave))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', dontread=memsave))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF', dontread=memsave))
            self.AGFF = np.array(h5df.getData(self.name, 'pic_AGFF'))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF', dontread=memsave))
            self.AoGFF = np.array(h5df.getData(self.name, 'pic_AoGFF', dontread=memsave))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))

        if likfunc == 'mark10':
            self.GtF = np.array(h5df.getData(self.name, 'pic_GtF', dontread=memsave))
            self.GGtD = np.array(h5df.getData(self.name, 'pic_GGtD', dontread=memsave))
            self.Emat = np.array(h5df.getData(self.name, 'pic_Emat'))
            self.GGtE = np.array(h5df.getData(self.name, 'pic_GGtE', dontread=memsave))
            self.SFmat = np.array(h5df.getData(self.name, 'pic_SFmat', dontread=memsave))
            self.SFdmmat = np.array(h5df.getData(self.name, 'pic_SFdmmat', dontread=memsave))
            self.FFmat = np.array(h5df.getData(self.name, 'pic_FFmat', dontread=memsave))
            self.SFfreqs = np.array(h5df.getData(self.name, 'pic_SFfreqs'))
            self.DSF = np.array(h5df.getData(self.name, 'pic_DSF', dontread=memsave))
            self.DFF = np.array(h5df.getData(self.name, 'pic_DFF'))
            self.EEmat = np.array(h5df.getData(self.name, 'pic_EEmat'))
            self.GGtEE = np.array(h5df.getData(self.name, 'pic_GGtEE'))
            self.AGr = np.array(h5df.getData(self.name, 'pic_AGr'))
            self.AGF = np.array(h5df.getData(self.name, 'pic_AGF', dontread=memsave))
            self.AGFF = np.array(h5df.getData(self.name, 'pic_AGFF'))
            self.AGD = np.array(h5df.getData(self.name, 'pic_AGD', dontread=memsave))
            self.AGE = np.array(h5df.getData(self.name, 'pic_AGE'))
            self.AGEE = np.array(h5df.getData(self.name, 'pic_AGEE'))
            self.AoGF = np.array(h5df.getData(self.name, 'pic_AoGF', dontread=memsave))
            self.AoGFF = np.array(h5df.getData(self.name, 'pic_AoGFF', dontread=memsave))
            self.AoGD = np.array(h5df.getData(self.name, 'pic_AoGD', dontread=memsave))
            self.AoGE = np.array(h5df.getData(self.name, 'pic_AoGE', dontread=memsave))
            self.AoGEE = np.array(h5df.getData(self.name, 'pic_AoGEE', dontread=memsave))
            self.Fdmmat = np.array(h5df.getData(self.name, 'pic_Fdmmat', dontread=memsave))
            self.Dmat = np.array(h5df.getData(self.name, 'pic_Dmat'))
            self.DF = np.array(h5df.getData(self.name, 'pic_DF'))
            self.Fmat = np.array(h5df.getData(self.name, 'pic_Fmat'))
            self.avetoas = np.array(h5df.getData(self.name, 'pic_avetoas'))



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


