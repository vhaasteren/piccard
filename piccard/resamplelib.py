from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import scipy.stats as stats
import scipy.special as ss
import scipy.ndimage.filters as filters
import scipy.interpolate as interp
from piccard import *
import glob, os, sys


class Bounded_kde(stats.gaussian_kde):
    r"""Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, low=None, high=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param low: The lower domain boundary.

        :param high: The upper domain boundary."""
        pts = np.atleast_1d(pts)

        assert pts.ndim == 1, 'Bounded_kde can only be one-dimensional'
        
        super(Bounded_kde, self).__init__(pts, *args, **kwargs)

        self._low = low
        self._high = high

    @property
    def low(self):
        """The lower bound of the domain."""
        return self._low

    @property
    def high(self):
        """The upper bound of the domain."""
        return self._high

    def evaluate(self, xs):
        """Return an estimate of the density evaluated at the given
        points."""
        xs = np.atleast_1d(xs)
        assert xs.ndim == 1, 'points must be one-dimensional'

        pdf = super(Bounded_kde, self).evaluate(xs)

        if self.low is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.low - xs)

        if self.high is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.high - xs)

        return pdf

    __call__ = evaluate



def getPlSpectrum(Apl, gpl, Tmax, ufreqs):
    """
    Convert power-law parameters to residual-power
    
    @param Apl:     Power-law amplitude
    @param gpl:     Power-law spectral index
    @param Tmax:    Duration experiment
    @param ufreqs:  Log10 of frequencies
    """
    
    pfreqs = 10**ufreqs
    Apl = 10**Apl
    ypl = (Apl**2 * pic_spy**3 / (12*np.pi*np.pi * (Tmax))) * ((pfreqs * pic_spy) ** (-gpl))
    
    return np.log10(ypl)

def getWnSpectrum(Wn, ufreqs):
    """
    Get the PSD for white-noise
    
    @param Wn:      RMS white-noise leven in sec
    @param ufreqs:  Log10 of frequencies
    """
    psd = Wn**2 * np.ones(len(ufreqs))
    return np.log10(psd)

def spectrumToHc(ufreqs, rho, Tmax):
    """
    Convert residual power per frequency to h_c
    
    @param ufreqs:  Log10 of frequencies
    @param rho:     Residual power
    @param Tmax:    Duration of experiment
    """
    
    pfreqs = 10**ufreqs
    norm = 12*Tmax*np.pi**2 / (pic_spy**3)
    return np.log10(np.sqrt(norm * (10**rho) * (pfreqs * pic_spy)**3))

def gw_turnover_spectrum(freqs, Tmax, pars):
    # get Amplitude and spectral index
    Amp = 10 ** pars[0]
    gamma = pars[1]
    f0 = 10 ** pars[2]
    kappa = pars[3]

    freqpy = freqs
    f1yr = 1.0 / pic_spy
    hcf = Amp * (freqpy / f1yr) ** ((3 - gamma) / 2) / \
        (1 + (f0 / freqpy) ** kappa) ** (1 / 2)
    return hcf ** 2 / 12 / np.pi ** 2 / freqpy ** 3 / Tmax


def gw_pl_spectrum(freqs, lh_c=0.0, si=4.33, Tmax=None):
    """
    Given an array of frequencies, return the power-law spectrum for h_c and si
    """
    spy = 31557600.0
    fr = freqs * spy
    if Tmax is None:
        Tmax = 1.0 / freqs[0]
    
    amp = 10**lh_c
    
    return ((amp**2) * spy**3 / (12*np.pi*np.pi * Tmax)) * fr ** (-si)

def get_kde(samples):
    """
    Given 1D samples 'samples', return a function
    that returns a kde estimate of the distribution
    
    hmm, seems too trivial actually :)
    """
    return bounded_kde(samples)

def transform_xy(x, y):
    """
    Make the transformation from (x,y) to z
    """
    return 0.5 * np.log10(10**(2*x) + 10**(2*y))

def smooth_hist(hist, xedges, sigma=0.75, kind='cubic'):
    """
    Create a smoothed version of the histogram
    """
    ghist = filters.gaussian_filter(hist, sigma=sigma)
    return interp.interp1d(xedges, ghist, kind=kind)

def kde_gwprior(low=-18.0, high=-10.0, bins=250, kind='cubic'):
    """
    The GW parameter is log(amp). We need a prior flat in amp.
    Return the KDE of that prior
    """
    gwamp = np.linspace(low, high, bins)
    prior = 10**gwamp
    return interp.interp1d(gwamp, prior, kind=kind)

def kde_multiply(kde_list, low=-18.0, high=-10.0, bins=250, kind='cubic'):
    """
    Multiply a bunch of KDEs, and return the resulting KDE
    """
    pdf = np.ones(bins)
    x = np.linspace(low, high, bins)
    for kde in kde_list:
        pdf *= kde(x)
    
    pdf *= bins / (np.sum(pdf) * (high-low))
    return smooth_hist(pdf, x, kind=kind)

def make_ul_kde(kde, minz, maxz, bins=250, kind='cubic'):
    """
    Given a kernel density estimator on the logarithmic interval [minz, maxz],
    return the kde of the upper-limit estimate, produced by changing the model
    in two components:
    10**(2*z) = 10**(2*x) + 10**(2*y)
    
    Marginalize over y, and get the kde for x
    """
    # Obtain the 2D distribution
    X, Y = np.mgrid[minz:maxz:bins*1j, minz:maxz:bins*1j]
    Z = transform_xy(X.ravel(), Y.ravel())
    dist_xy = np.reshape(kde(Z), X.shape)
    
    # Marginalize
    X1D = np.mgrid[minz:maxz:bins*1j]
    dx = X1D[1]-X1D[0]
    xedges = np.append(X1D+0.5*dx, [X1D[0]-0.5*dx])
    dist_x = np.sum(dist_xy, axis=1)
    dist_x *= bins / (np.sum(dist_x) * (maxz-minz))
    
    return smooth_hist(dist_x, X1D, kind=kind)

def single_psr_sp_ul(psrdir, burnin=5000, low=-18.0, high=-10.0, bins=250, numfreqs=None, kind='cubic'):
    """
    For a single pulsar (chain directory), create the kernel density estimate distributions
    of the per-frequency posteriors. Also create the upper-limit equivalent
    """
    lp, ll, chain, labels, pulsars, pulsarnames, stypes, mlpso, mlpsopars = ReadMCMCFile(psrdir, incextra=True)
    inds = np.where(np.array(stypes) == 'spectrum')[0]
    
    if numfreqs is not None:
        inds = inds[:numfreqs]

    freqs = np.float64(np.array(labels)[inds])
    
    kdes = []
    kdes_ul = []
    for ii in inds:
        samples = chain[burnin:,ii]
        kde = bounded_kde.Bounded_kde(samples, low=low)
        kdes.append(kde)
        kde_ul = make_ul_kde(kde, low, high, bins=bins, kind=kind)
        kdes_ul.append(kde_ul)
    
    return freqs, kdes, kdes_ul

def process_psrdirs(psrdirs, low=-18.0, high=-10.0, bins=100, numfreqs=None,
        verbose=False, kind='cubic'):
    """
    Create the result list of dicts of the psr dictionaries
    """
    resdict = []
    for psrdir in psrdirs:
        fr, kd, kdu = single_psr_sp_ul(psrdir, burnin=5000,
                                       low=low, high=high, bins=bins,
                                       numfreqs=numfreqs, kind=kind)
        resdict.append({'freqs':fr, 'kdes':kd, 'kdes_ul':kdu})
        if verbose:
            print("Done with {0}".format(psrdir))

    return resdict

def gw_ul_spectrum(resdict, confidence=0.95, low=-18.0, high=-10.0, bins=100):
    """
    Using the list of result dictionaries, create the spectrum upper-limits
    """
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    prior_kde = kde_gwprior(low=low, high=high, bins=bins)
    ul = np.zeros_like(gwfreqs)
    gwamps = np.linspace(low, high, bins)

    for ii, freq in enumerate(gwfreqs):
        kdes = [prior_kde]
        for pp, psrres in enumerate(resdict):
            kdes.append(psrres['kdes_ul'][ii])

        fullkde = kde_multiply(kdes, low, high, bins=bins)

        cdf = np.cumsum(fullkde(gwamps)) / np.sum(fullkde(gwamps))
        ul[ii] = gwamps[cdf > confidence][0]
    
    return gwfreqs, ul

def ul_loglikelihood(resdict, logpsd, low=-18.0, high=-10.0):
    """
    For a set of log10(PSD) values ``lpsd'', evaluate the upper-limit likelihood
    of the GW frequencies of all pulsars

    Note that this marginalized likelihood/posterior has a prior flat in
    log(gwamp)
    """
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    lpsd = np.array(logpsd).copy()
    loglik = 0.0

    # Set the practical log10(psd) for this round
    lpsd[np.where(lpsd < low)] = low
    lpsd[np.where(lpsd > high)] = high
    
    # Calculate the posterior, by multiplying the kdes of all pulsars
    for jj, freq in enumerate(gwfreqs):
        for pp, psrres in enumerate(resdict):
            kde = psrres['kdes_ul'][jj]

            loglik += np.log(kde(lpsd[jj]))

    return loglik
        
def gw_ul_powerlaw(resdict, confidence=0.95, low=-18.0, high=-10.0, bins=100,
        gwlow=-20.0, gwhigh=-13.0, si=4.33, gwbins=50000):
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    gwamps = np.linspace(gwlow, gwhigh, bins)
    prior_kde = kde_gwprior(low=gwlow, high=gwhigh, bins=bins)
    lpdf = np.zeros_like(gwamps)

    for ii, gwamp in enumerate(gwamps):
        # Calculate the spectrum for this GW amplitude
        spect = np.log10(gw_pl_spectrum(gwfreqs, lh_c=gwamp, si=si))
        spect[np.where(spect < low)] = low
        spect[np.where(spect > high)] = high

        lpdf[ii] += ul_loglikelihood(resdict, spect, low=low, high=high)
        lpdf[ii] += np.log(prior_kde(gwamp))

    # Smooth the log-pdf
    lpdf_kde = smooth_hist(lpdf, gwamps, sigma=0.75, kind='linear')
    gwamps_l = np.linspace(gwlow, gwhigh, gwbins)
    lpdf_l = lpdf_kde(gwamps_l)
    pdf = np.exp(lpdf_l - np.max(lpdf_l))
    #pdf *= gwbins / (np.sum(pdf) * (gwhigh-gwlow))
    
    # Now calculate the upper-limit
    cdf = np.cumsum(pdf) / np.sum(pdf)

    return gwamps_l[cdf > confidence][0], gwamps_l, pdf

def gw_ul_powerlaw_old(resdict, confidence=0.95, low=-18.0, high=-10.0, bins=100,
        gwlow=-20.0, gwhigh=-13.0, si=4.33, gwbins=50000):
    """
    Using the list of result dictionaries, create the power-law gamma=4.33 upper-limits
    """
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    gwamps = np.linspace(gwlow, gwhigh, bins)
    prior_kde = kde_gwprior(low=gwlow, high=gwhigh, bins=bins)
    lpdf = np.zeros_like(gwamps)

    for ii, gwamp in enumerate(gwamps):
        # Calculate the spectrum for this GW amplitude
        spect = np.log10(gw_pl_spectrum(gwfreqs, lh_c=gwamp, si=si))
        spect[np.where(spect < low)] = low
        spect[np.where(spect > high)] = high
        
        # Calculate the posterior, by multiplying the kdes of all pulsars
        for jj, freq in enumerate(gwfreqs):
            for pp, psrres in enumerate(resdict):
                kde = psrres['kdes_ul'][jj]

                lpdf[ii] += np.log(kde(spect[jj]))
        
        lpdf[ii] += np.log(prior_kde(gwamp))

    # Smooth the log-pdf
    lpdf_kde = smooth_hist(lpdf, gwamps, sigma=0.75, kind='linear')
    gwamps_l = np.linspace(gwlow, gwhigh, gwbins)
    lpdf_l = lpdf_kde(gwamps_l)
    pdf = np.exp(lpdf_l - np.max(lpdf_l))
    #pdf *= gwbins / (np.sum(pdf) * (gwhigh-gwlow))
    
    # Now calculate the upper-limit
    cdf = np.cumsum(pdf) / np.sum(pdf)
    
    return gwamps_l[cdf > confidence][0], gwamps_l, pdf


def gw_ul_powerlaw_2d(resdict, confidence=0.95, low=-18.0, high=-10.0, gwlow=-17.0, gwhigh=-13.0, si=4.33, bins=100):
    """
    Using the list of result dictionaries, create the power-law gamma=4.33 upper-limits
    """
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    
    gwamps = np.linspace(gwlow, gwhigh, bins)
    gwsis = np.linspace(2.01, 6.99, bins)
    lpdf = np.zeros((bins, bins))

    prior_kde = kde_gwprior(low=gwlow, high=gwhigh, bins=bins)

    for kk, gwsi in enumerate(gwsis):
        for ii, gwamp in enumerate(gwamps):
            # Calculate the spectrum for this GW amplitude
            spect = np.log10(gw_pl_spectrum(gwfreqs, lh_c=gwamp, si=gwsi))
            spect[np.where(spect < low)] = low
            spect[np.where(spect > high)] = high

            # Calculate the posterior, by multiplying the kdes of all pulsars
            for jj, freq in enumerate(gwfreqs):
                for pp, psrres in enumerate(resdict):
                    kde = psrres['kdes_ul'][jj]
                    lpdf[ii,kk] += np.log(kde(spect[jj]))

            lpdf[ii,kk] += np.log(prior_kde(gwamp))
    
    pdf = np.exp(lpdf - np.max(lpdf))

    return gwamps, gwsis, pdf

def loglikelihood(pars, resdict, stype='powerlaw', model='uniform', low=-18.0,
        high=-10.0):
    """
    Logliklihood...  blah
    """
    Tmax = 1.0 / resdict[0]['freqs'][0]
    gwfreqs = resdict[0]['freqs']

    if stype=='powerlaw':
        spect = np.log10(gw_pl_spectrum(gwfreqs, lh_c=pars[0], si=pars[1],
                                        Tmax=Tmax))
    elif stype=='turnover':
        spect = np.log10(gw_turnover_spectrum(gwfreqs, Tmax=Tmax, pars=pars))
    else:
        raise NotImplementedError("Only powerlaw and turnover implemented")

    spect[np.where(spect < low)] = low
    spect[np.where(spect > high)] = high

    return ul_loglikelihood(resdict, spect, low=low, high=high)


def logprior(pars, stype='powerlaw', model='uniform'):
    prior = 0.0
    if stype=='powerlaw' and model=='uniform':
        prior += np.log(10 ** pars[0])
    elif stype=='powerlaw' and model=='sesana':
        m = -15
        s = 0.22
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    elif stype=='powerlaw' and model=='mcwilliams':
        m = np.log10(4.1e-15)
        s = 0.26
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    elif stype=='turnover' and model=='uniform':
        prior += np.log(10 ** pars[0])
    elif stype=='turnover' and model=='sesana':
        m = -15
        s = 0.22
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    elif stype=='turnover' and model=='mcwilliams':
        m = np.log10(4.1e-15)
        s = 0.26
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    return prior

