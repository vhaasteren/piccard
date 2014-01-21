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




Description of the ptasignals dictionary in the docs. The old class object was:

class ptasignal(object):
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


# Internal modules
from . import anisotropygammas as ang
from .piccard_datafile import DataFile
from .piccard_ptapulsar import *
from .piccard_ptafuncs import *
from .piccard_constants import *

try:    # If without libstempo, can still read hdf5 files
    import libstempo
    t2 = libstempo
except ImportError:
    t2 = None

# In order to keep the dictionary in order
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class ptaLikelihood(object):
    """ Basic implementation of the model/likelihood.
    
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
    pamplitudeind = None
    initialised = False
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
    FGGNGGF = None      #        mark3, mark?
    EGGNGGE = None      #                      mark6
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
    def __init__(self, h5filename=None, jsonfilename=None, pulsars='all', auxFromFile=True):
        self.clear()

        if h5filename is not None:
            self.initFromFile(h5filename, pulsars=pulsars)

            if jsonfilename is not None:
                self.initModelFromFile(jsonfilename, auxFromFile=auxFromFile)

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
        self.pamplitudeind = None
        self.initialised = False
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
        if not 'Tmax' in signal:
            Tstart = np.min(self.ptapsrs[0].toas)
            Tfinish = np.max(self.ptapsrs[0].toas)
            for m2psr in self.ptapsrs:
                Tstart = np.min([np.min(m2psr.toas), Tstart])
                Tfinish = np.max([np.max(m2psr.toas), Tfinish])
            Tmax = Tfinish - Tstart

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

        signal['Nvec'] = np.ones(len(self.ptapsrs[signal['pulsarind']].toaerrs))

        if signal['flagname'] != 'pulsarname':
            # This equad only applies to some TOAs, not all of 'm
            ind = np.array(self.ptapsrs[signal['pulsarind']].flags) != signal['flagvalue']
            signal['Nvec'][ind] = 0.0

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
    @param unitconversion:  The unit conversion factor for the timing model
    """
    def addSignalTimingModel(self, signal, linear=True):
        # Assert that all the correct keys are there...
        keys = ['pulsarind', 'stype', 'corr', 'bvary', 'parid', \
                'pmin', 'pmax', 'pwidth', 'pstart', 'parindex', 'unitconversion']
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
        for ii in range(npsrs):
            if not self.likfunc in ['mark2']:
                self.npf[ii] = len(self.ptapsrs[ii].Ffreqs)
                self.npff[ii] = self.npf[ii]

            if self.likfunc in ['mark4ln', 'mark9', 'mark10']:
                self.npff[ii] += len(self.ptapsrs[ii].SFfreqs)

            if self.likfunc in ['mark4', 'mark4ln']:
                self.npu[ii] = len(self.ptapsrs[ii].avetoas)

            if self.likfunc in ['mark1', 'mark4', 'mark4ln', 'mark6', 'mark6fa', 'mark8', 'mark10']:
                self.npfdm[ii] = len(self.ptapsrs[ii].Fdmfreqs)
                self.npffdm[ii] = len(self.ptapsrs[ii].Fdmfreqs)

            if self.likfunc in ['mark10']:
                self.npffdm[ii] += len(self.ptapsrs[ii].SFdmfreqs)

            self.npobs[ii] = len(self.ptapsrs[ii].toas)
            self.npgs[ii] = self.ptapsrs[ii].Hmat.shape[1]
            self.npgos[ii] = self.ptapsrs[ii].Homat.shape[1]
            self.ptapsrs[ii].Nvec = np.zeros(len(self.ptapsrs[ii].toas))
            self.ptapsrs[ii].Nwvec = np.zeros(self.ptapsrs[ii].Hmat.shape[1])
            self.ptapsrs[ii].Nwovec = np.zeros(self.ptapsrs[ii].Homat.shape[1])

        self.Phi = np.zeros((np.sum(self.npf), np.sum(self.npf)))
        self.Thetavec = np.zeros(np.sum(self.npfdm))

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



    """
    Based on somewhat simpler quantities, this function makes a full model
    dictionary. Standard use would be to save that dictionary in a json file, so
    that it does not need to be manually created by the user

    @param nfreqmodes:      blah
    @param dmnfreqmodes:    blah
    """
    def makeModelDict(self,  nfreqs=20, ndmfreqs=None, \
            incRedNoise=False, noiseModel='powerlaw', fc=None, \
            incDM=False, dmModel='powerlaw', \
            incClock=False, clockModel='powerlaw', \
            incGWB=False, gwbModel='powerlaw', \
            incDipole=False, dipoleModel='powerlaw', \
            incAniGWB=False, anigwbModel='powerlaw', lAniGWB=1, \
            incBWM=False, \
            incTimingModel=False, nonLinear=False, \
            varyEfac=True, incEquad=False, separateEfacs=False, \
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

                # If compression 
                if compression != 'average' and compression != 'avefrequencies' \
                        and likfunc[:5] != 'mark4':
                    print "WARNING: Jitter included, but likelihood function will deal with it as an equad."
                    print "         Use an adequate compression level, or a 'mark4' likelihood"

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
                if dmModel=='spectrum':
                    nfreqs = numDMFreqs[ii]
                    bvary = [True]*nfreqs
                    pmin = [-14.0]*nfreqs
                    pmax = [-3.0]*nfreqs
                    pstart = [-7.0]*nfreqs
                    pwidth = [0.1]*nfreqs
                    dmModel = 'dmspectrum'
                elif dmModel=='powerlaw':
                    bvary = [True, True, False]
                    pmin = [-14.0, 0.02, 1.0e-11]
                    pmax = [-6.5, 6.98, 3.0e-9]
                    pstart = [-13.0, 2.01, 1.0e-10]
                    pwidth = [0.1, 0.1, 5.0e-11]
                    dmModel = 'dmpowerlaw'

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

            if incTimingModel:
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
                    tmperrs = np.sqrt(np.diag(Sigma)) * m2psr.unitconversion
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

                # Select the numerical parameters. These are the ones not
                # present in the quantities that getModifiedDesignMatrix
                # returned
                parids=[]
                bvary = []
                pmin = []
                pmax = []
                pwidth = []
                pstart = []
                unitconversion = []
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
                            pmin += [-20.0 * tmperrs[jj] + tmpest[jj]]
                            pmax += [20.0 * tmperrs[jj] + tmpest[jj]]
                            pwidth += [(pmax[-1]-pmin[-1])/20.0]
                        pstart += [tmpest[jj]]
                        unitconversion += [m2psr.unitconversion[jj]]

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
                    "parid":parids,
                    "unitconversion":unitconversion
                    })
                signals.append(newsignal)

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
            if 'unitconversion' in signals[-1]:
                signals[-1]['unitconversion'] = map(float, signals[-1]['unitconversion'])

        modeldict = OrderedDict({
            "file version":2013.12,
            "author":"piccard-makeModel",
            "numpulsars":len(self.ptapsrs),
            "pulsarnames":[self.ptapsrs[ii].name for ii in range(len(self.ptapsrs))],
            "numNoiseFreqs":list(numNoiseFreqs),
            "numDMFreqs":list(numDMFreqs),
            "compression":self.compression,
            "orderFrequencyLines":self.orderFrequencyLines,
            "evalCompressionComplement":self.evallikcomp,
            "likfunc":self.likfunc,
            "signals":signals
            })

        return modeldict


    """
    Initialise a model from a json file

    @param filename:    Filename of the json file with the model
    """
    def initModelFromFile(self, filename, auxFromFile=True):
        with open(filename) as data_file:
            model = OrderedDict(json.load(data_file))
        self.initModel(model, fromFile=auxFromFile)

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


    """
    Initialise the model.
    @param numNoiseFreqs:       Dictionary with the full model
    @param fromFile:            Try to read the necessary Auxiliaries quantities
                                from the HDF5 file
    @param verbose:             Give some extra information about progress
    """
    def initModel(self, fullmodel, fromFile=True, verbose=False):
        numNoiseFreqs = fullmodel['numNoiseFreqs']
        numDMFreqs = fullmodel['numDMFreqs']
        compression = fullmodel['compression']
        evalCompressionComplement = fullmodel['evalCompressionComplement']
        orderFrequencyLines = fullmodel['orderFrequencyLines']
        likfunc = fullmodel['likfunc']
        signals = fullmodel['signals']

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
        Tmax = Tfinish - Tstart

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
        separateEfacs = numEfacs > 1

        # Modify design matrices, and create pulsar Auxiliary quantities
        for pindex, m2psr in enumerate(self.ptapsrs):
            # If we model DM variations, we will need to include QSD
            # marginalisation for DM. Modify design matrix accordingly
            #if dmModel[pindex] != 'None':
            if numDMFreqs[pindex] > 0:
                m2psr.addDMQuadratic()

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
                m2psr.readPulsarAuxiliaries(self.h5df, Tmax, \
                        numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], not separateEfacs[pindex], \
                        nSingleFreqs=numSingleFreqs[pindex], \
                        nSingleDMFreqs=numSingleDMFreqs[pindex], \
                        likfunc=likfunc, compression=compression, \
                        memsave=True)
            except (StandardError, ValueError, KeyError, IOError, RuntimeError) as err:
                # Create the Auxiliaries ourselves

                # For every pulsar, construct the auxiliary quantities like the Fourier
                # design matrix etc
                if verbose:
                    print str(err)
                    print "Creating Auxiliaries for {0}".format(m2psr.name)
                m2psr.createPulsarAuxiliaries(self.h5df, Tmax, numNoiseFreqs[pindex], \
                        numDMFreqs[pindex], not separateEfacs[pindex], \
                                nSingleFreqs=numSingleFreqs[pindex], \
                                nSingleDMFreqs=numSingleDMFreqs[pindex], \
                                likfunc=likfunc, compression=compression, \
                                write='likfunc', tmsigpars=tmsigpars)

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
            self.addSignal(signal, index, Tmax)
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
                    if jj >= len(self.ptapsrs[psrindex].Ffreqs)/2:
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

                    if jj < 3:
                        if sig['corr'] == 'gr':
                            flagvalue = ['GWB-Amplitude', 'GWB-spectral-index', 'low-frequency-cutoff'][jj]
                        elif sig['corr'] == 'uniform':
                            flagvalue = ['CLK-Amplitude', 'CLK-spectral-index', 'low-frequency-cutoff'][jj]
                        elif sig['corr'] == 'dipole':
                            flagvalue = ['DIP-Amplitude', 'DIP-spectral-index', 'low-frequency-cutoff'][jj]
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
                else:
                    flagname = 'none'
                    flagvalue = 'none'

                pardes.append(\
                        {'index': index, 'pulsar': psrindex, 'sigindex': ii, \
                            'sigtype': sig['stype'], 'correlation': sig['corr'], \
                            'name': flagname, 'id': flagvalue})

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
            fil.write("{0:d} \t{1:d} \t{2:s} \t{3:s} \t{4:s} \t{5:s}\n".format(\
                    self.pardes[ii]['index'],
                    self.pardes[ii]['pulsar'],
                    self.pardes[ii]['sigtype'],
                    self.pardes[ii]['correlation'],
                    self.pardes[ii]['name'],
                    self.pardes[ii]['id']))

        fil.close

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
    def getEfacNumbers(self):
        parind = []
        psrind = []
        names = []

        for ii, m2signal in enumerate(self.ptasignals):
            if m2signal['stype'] == 'efac' and m2signal['bvary'][0]:
                parind.append(m2signal['parindex'])
                psrind.append(m2signal['pulsarind'])
                names.append(m2signal['flagvalue'])

        return (parind, psrind, names)

    """
    Return a list of all spectrum signals: signal name, start-par, stop-par, and
    the actual frequencies

    TODO: parameters can be non-varying. Take that into accoutn as well
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
    def setPsrNoise(self, parameters, selection=None, incJitter=True):
        # For every pulsar, set the noise vector to zero
        for m2psr in self.ptapsrs:
            if m2psr.twoComponentNoise:
                m2psr.Nwvec[:] = 0
                m2psr.Nwovec[:] = 0
            #else:
            m2psr.Nvec[:] = 0
            m2psr.Qamp = 0

        if selection is None:
            selection = np.array([1]*len(self.ptasignals), dtype=np.bool)

        # Loop over all white noise signals, and fill the pulsar Nvec
        for ss, m2signal in enumerate(self.ptasignals):
            m2signal = self.ptasignals[ss]
            if selection[ss]:
                if m2signal['stype'] == 'efac':
                    if m2signal['npars'] == 1:
                        pefac = parameters[m2signal['parindex']]
                    else:
                        pefac = m2signal['pstart'][0]

                    if self.ptapsrs[m2signal['pulsarind']].twoComponentNoise:
                        self.ptapsrs[m2signal['pulsarind']].Nwvec += \
                                self.ptapsrs[m2signal['pulsarind']].Wvec * pefac**2
                        self.ptapsrs[m2signal['pulsarind']].Nwovec += \
                                self.ptapsrs[m2signal['pulsarind']].Wovec * pefac**2

                    self.ptapsrs[m2signal['pulsarind']].Nvec += m2signal['Nvec'] * pefac**2

                elif m2signal['stype'] == 'equad':
                    if m2signal['npars'] == 1:
                        pequadsqr = 10**(2*parameters[m2signal['parindex']])
                    else:
                        pequadsqr = 10**(2*m2signal['pstart'][0])

                    if self.ptapsrs[m2signal['pulsarind']].twoComponentNoise:
                        self.ptapsrs[m2signal['pulsarind']].Nwvec += pequadsqr
                        self.ptapsrs[m2signal['pulsarind']].Nwovec += pequadsqr

                    self.ptapsrs[m2signal['pulsarind']].Nvec += m2signal['Nvec'] * pequadsqr
                elif m2signal['stype'] == 'jitter':
                    if m2signal['npars'] == 1:
                        pequadsqr = 10**(2*parameters[m2signal['parindex']])
                    else:
                        pequadsqr = 10**(2*m2signal['pstart'][0])

                    self.ptapsrs[m2signal['pulsarind']].Qamp = pequadsqr

                    if incJitter:
                        # Need to include it just like the equad (for compresison)
                        self.ptapsrs[m2signal['pulsarind']].Nvec += m2signal['Nvec'] * pequadsqr

                        if self.ptapsrs[m2signal['pulsarind']].twoComponentNoise:
                            self.ptapsrs[m2signal['pulsarind']].Nwvec += pequadsqr
                            self.ptapsrs[m2signal['pulsarind']].Nwovec += pequadsqr


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
    def constructPhiAndTheta(self, parameters, selection=None):
        self.Phi[:] = 0         # Start with a fresh matrix
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
                        self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += 10**pcdoubled
                    elif m2signal['corr'] in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                        nfreq = m2signal['npars']

                        if m2signal['corr'] in ['gr', 'uniform', 'dipole']:
                            pcdoubled = np.array([sparameters, sparameters]).T.flatten()
                            corrmat = m2signal['corrmat']
                        elif m2signal['corr'] == 'anisotropicgwb':
                            nclm = m2signal['aniCorr'].clmlength()
                            pcdoubled = np.array([\
                                    sparameters[:-nclm],\
                                    sparameters[:-nclm]]).T.flatten()
                            clm = sparameters[-nclm:]
                            corrmat = m2signal['aniCorr'].corrmat(clm)

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
                        self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                    elif m2signal['corr'] in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                        freqpy = m2signal['Ffreqs'] * pic_spy
                        pcdoubled = (Amp**2 * pic_spy**3 / (12*np.pi*np.pi * m2signal['Tmax'])) * freqpy ** (-Si)
                        nfreq = len(freqpy)

                        if m2signal['corr'] in ['gr', 'uniform', 'dipole']:
                            corrmat = m2signal['corrmat']
                        elif m2signal['corr'] == 'anisotropicgwb':
                            nclm = m2signal['aniCorr'].clmlength()
                            clm = sparameters[-nclm:]
                            corrmat = m2signal['aniCorr'].corrmat(clm)

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
                        self.Phi[findex:findex+2*nfreq, findex:findex+2*nfreq][di] += pcdoubled
                    elif m2signal['corr'] in ['gr', 'uniform', 'dipole', 'anisotropicgwb']:
                        freqpy = self.ptapsrs[0].Ffreqs * pic_spy
                        pcdoubled = (Amp * pic_spy**3 / m2signal['Tmax']) / \
                                ((1 + (freqpy/fc)**2)**(-0.5*alpha))
                        nfreq = len(freqpy)

                        if m2signal['corr'] in ['gr', 'uniform', 'dipole']:
                            corrmat = m2signal['corrmat']
                        elif m2signal['corr'] == 'anisotropicgwb':
                            nclm = m2signal['aniCorr'].clmlength()
                            clm = sparameters[-nclm:m2signal['parindex']+m2signal['npars']]
                            corrmat = m2signal['aniCorr'].corrmat(clm)

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

                    self.Phi[findex:findex+2, findex:findex+2][di] += 10**pcdoubled
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

                    # Create slicing vector (select parameters actually in signal)
                    for jj, parid in enumerate(self.ptapsrs[pp].ptmdescription):
                        if parid in newdes:
                            ind += [True]
                        else:
                            ind += [False]
                    ind = np.array(ind, dtype=np.bool)

                    # residuals = M * pars
                    self.ptapsrs[pp].detresiduals -= \
                            np.dot(self.ptapsrs[pp].Mmat[:,ind], \
                            (sparameters-m2signal['pstart']) / m2signal['unitconversion'])

                elif m2signal['stype'] == 'nonlineartimingmodel':
                    # The t2psr libstempo object has to be set. Assume it is.
                    pp = m2signal['pulsarind']

                    # For each varying parameter, update the libstempo object
                    # parameter with the new value
                    pindex = 0
                    for jj in range(m2signal['ntotpars']):
                        if m2signal['bvary'][jj]:
                            # If this parameter varies, update the parameter
                            self.ptapsrs[pp].t2psr[m2signal['parid'][jj]].val = \
                                    sparameters[pindex]
                            pindex += 1

                    # Generate the new residuals
                    # QUESTION: why do I have to update the BATs for good results?
                    self.ptapsrs[pp].detresiduals = np.array(self.ptapsrs[pp].t2psr.residuals(updatebats=True), dtype=np.double)

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

                            self.ptapsrs[pp].detresiduals -= - bwmsig


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

                    m2psr.UtFF = np.dot(m2psr.Umat.T, m2psr.FFmat)
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
                m2psr.DSF = np.dot(m2psr.Dmat, m2psr.SFdmmat)
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

        self.setPsrNoise(parameters, incJitter=False)

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
        for ii in range(npsrs):
            if self.npgos[ii] > 0:
                if self.ptapsrs[ii].twoComponentNoise:
                    self.rGr[ii] = np.sum(self.ptapsrs[ii].AoGr ** 2 / self.ptapsrs[ii].Nwovec)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwovec))
                else:
                    Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                    NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hocmat.T).T
                    GcNiGc = np.dot(self.ptapsrs[ii].Hocmat.T, NiGc)
                    GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)

                    try:
                        cf = sl.cho_factor(GcNiGc)
                        self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
                                2*np.sum(np.log(np.diag(cf[0])))
                        GcNiGcr = sl.cho_solve(cf, GcNir)
                    except np.linalg.LinAlgError:
                        print "comploglikelihood: GcNiGc singular"

                    self.rGr[ii] = np.dot(self.ptapsrs[ii].detresiduals, Nir) \
                            - np.dot(GcNir, GcNiGcr)
            else:
                self.rGr[ii] = 0
                self.GNGldet[ii] = 0

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
        self.GCG[:] = 0
        for ii in range(npsrs):
            gindex = np.sum(self.npgs[:ii])
            ng = self.npgs[ii]

            # Two-component noise or not does not matter
            self.GCG[gindex:gindex+ng, gindex:gindex+ng] = \
                    np.dot(self.ptapsrs[ii].Hmat.T, (self.ptapsrs[ii].Nvec * self.ptapsrs[ii].Hmat.T).T)

            # Create the total GtF and GtD lists for addition of Red(DM) noise
            GtFtot.append(self.ptapsrs[ii].GtF)
            GtDtot.append(self.ptapsrs[ii].GtD)

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
        #self.GCG += np.dot(GtD, (self.Thetavec * GtD).T)
        self.GCG += blockmul(np.diag(self.Thetavec), GtD.T, self.npfdm, self.npgs)

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
        for ii in range(npsrs):
            if self.ptapsrs[ii].twoComponentNoise:
                # This is equivalent to np.dot(np.diag(1.0/Nwvec, AGF))
                self.rGr[ii] = np.sum(self.ptapsrs[ii].AGr ** 2 / self.ptapsrs[ii].Nwvec)
                self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nwvec))
            else:
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)

                try:
                    cf = sl.cho_factor(GcNiGc)
                    self.GNGldet[ii] = np.sum(np.log(self.ptapsrs[ii].Nvec)) + \
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
        for ii in range(npsrs):
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
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Fmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].Fmat)

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
                        np.dot(NiF.T, self.ptapsrs[ii].Fmat) - np.dot(GcNiF.T, GcNiGcF)



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
        for ii in range(npsrs):
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
                Nir = self.ptapsrs[ii].detresiduals / self.ptapsrs[ii].Nvec
                NiGc = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Hcmat.T).T
                GcNiGc = np.dot(self.ptapsrs[ii].Hcmat.T, NiGc)
                NiF = ((1.0/self.ptapsrs[ii].Nvec) * self.ptapsrs[ii].Fmat.T).T
                GcNir = np.dot(NiGc.T, self.ptapsrs[ii].detresiduals)
                GcNiF = np.dot(NiGc.T, self.ptapsrs[ii].Fmat)

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
        self.setPsrNoise(parameters, incJitter=False)

        # MARK B

        self.constructPhiAndTheta(parameters)

        if self.haveDetSources:
            self.updateDetSources(parameters)

        # MARK C

        # Armed with the Noise (and it's inverse), we'll construct the
        # auxiliaries for all pulsars
        for ii in range(npsrs):
            findex = np.sum(self.npf[:ii])
            nfreq = int(self.npf[ii]/2)
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

            UPhiU = np.dot(self.ptapsrs[0].UtF, np.dot(self.Phi, self.ptapsrs[0].UtF.T))
            Phi = UPhiU + self.ptapsrs[0].Qamp * np.eye(len(self.ptapsrs[0].avetoas))

            #PhiLD = np.sum(np.log(np.diag(Phi)))
            #Phiinv = np.diag(1.0 / np.diag(Phi))
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

                #print "Fallback to SVD for Phi", parameters

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

            #print "Fallback to SVD for Sigma", parameters
        # Mark F

        # Now we are ready to return the log-likelihood
        #print rGSigmaGr, PhiLD, SigmaLD, np.trace(self.UGGNGGU), \
        #        np.trace(Phiinv), np.trace(self.Phi)
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
        for ii in range(npsrs):
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
            Phi = UPhiU + self.ptapsrs[0].Qamp * np.eye(len(self.ptapsrs[0].avetoas))

            #PhiLD = np.sum(np.log(np.diag(Phi)))
            #Phiinv = np.diag(1.0 / np.diag(Phi))
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
        for ii in range(npsrs):
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

        # Construct Sigma (needs to be written out because of the ordering of
        # the phi/theta matrices)
        self.Sigma = self.EGGNGGE
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
        for ii in range(npsrs):
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
        for ii in range(npsrs):
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
        for ii in range(npsrs):
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
        for ii in range(npsrs):
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
        for ii in range(npsrs):
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

            if psr.DF.shape[1] == nppfdm:
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
                qvec = np.append(qvec, [psr.Qamp]*len(psr.avetoas))

            #if totU.shape[1] == len(qvec):
            #    Cov += np.dot(totU, (qvec * totU).T)

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

        self.setPsrNoise(parameters, incJitter=False)

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
                totDmat[nindex:nindex+npobs, nindex:nindex+npobs] = self.ptapsrs[ii].Dmat

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
            Cov += np.dot(totDmat, np.dot(Cdm, totDmat))
        else:
            # Construct them from Phi/Theta
            Cov += np.dot(totFmat, np.dot(self.Phi, totFmat.T))
            if self.Thetavec is not None and len(self.Thetavec) == totDFmat.shape[1]:
                Cov += np.dot(totDFmat, np.dot(np.diag(self.Thetavec), totDFmat.T))

            # Include jitter
            qvec = np.array([])
            for pp, psr in enumerate(self.ptapsrs):
                qvec = np.append(qvec, [psr.Qamp]*len(psr.avetoas))
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

        self.setPsrNoise(mlparameters, incJitter=False)

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
                        np.dot(psr.Umat, np.dot(psr.Qamp * np.eye(psr.Umat.shape[1]), \
                        psr.Umat.T))

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
            tmpdelta = chi[index:index+np.sum(ind)] * psr.unitconversion[ind]

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
        for ii in range(npsrs):
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
            totDvec[nindex:nindex+nobs] = np.diag(self.ptapsrs[ii].Dmat)

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
            totDvec[nindex:nindex+nobs] = np.diag(self.ptapsrs[ii].Dmat)

            totGFp[gindexp:gindexp+ngsp, findexp:findexp+2*nfreqp] = \
                    np.dot(predlikob.ptapsrs[ii].Gmat.T, predlikob.ptapsrs[ii].Fmat)
            totFp[nindexp:nindexp+nobsp, findexp:findexp+2*nfreqp] = \
                    predlikob.ptapsrs[ii].Fmat
            totGp[nindexp:nindexp+nobsp, gindexp:gindexp+ngsp] = predlikob.ptapsrs[ii].Gmat
            totGrp[gindexp:gindexp+ngsp] = predlikob.ptapsrs[ii].Gr
            totDvecp[nindexp:nindexp+nobsp] = np.diag(predlikob.ptapsrs[ii].Dmat)

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

