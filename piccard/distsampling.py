from __future__ import division, print_function

import numpy as np

"""
This file implements sampling from arbitrary truncated distributions. This is
useful for Gibbs sampling when we can analytically sample the posterior
distribution.
"""

class Distribution(object):
    """
    Draws samples from a one/many dimensional probability distribution, by means
    of inversion of the discrete inversion of the cumulative density function

    The pdf can be sorted first to prevent numerical error in the cumulative
    sum. This is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions, the overhead is
    minimal

    A call to this distribution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]

        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is
           imlpicitly normalized"""
        return self.cdf[-1]

    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high = self.sum, size = N)

        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)

        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]

        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)

        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = index + np.random.uniform(size=index.shape)

        return self.transform(index)

def sample_PSD_jeffreys_an(tau, lxmin, lxmax):
    """
    Sample analytically from the distribution: exp(-tau/x) * tau/x**2
    The distribution is truncated outside: 10**lxmin < x < 10**lxmax

    See equation (51) of van Haasteren & Vallisneri, 2014,
    Physical Review D, Volume 90, Issue 10, id.104012

    @param tau:     Scaling parameter
    @param lxmin:   Minimum bound (log10(xmin))
    @param lxmax:   Maximum bound (log10(xmax))

    @return:    log10(x)
    """
    scale = 1 - np.exp(tau*(1.0/(10**lxmax)-1.0/(10**lxmin)))
    eta = np.random.rand(1)[0] * scale
    
    return np.log10(-tau/(np.log(1-eta)-tau/(10**lxmax)))

def sample_func_num(func, tau, lxmin, lxmax, Ninter=500):
    """
    Sample numerically from an arbitrary distribution "func". The distribution
    is truncated outside: 10**lxmin < x < 10**lxmax

    @param tau:     Scaling parameter
    @param lxmin:   Minimum bound (log10(xmin))
    @param lxmax:   Maximum bound (log10(xmax))
    @param Ninter:  Number of interpolation nodes in log(x)

    @return:    log10(x)

    TODO:   Make the arguments to func variable
    """
    lexmin, lexmax = lxmin*np.log(10), lxmax*np.log(10)
    
    alpha = np.linspace(lexmin, lexmax, Ninter, endpoint=False)
    fa = func(alpha, tau)
    dist = Distribution(fa, transform=lambda i:i*(lexmax-lexmin)/Ninter+lexmin)
    
    return dist(1)[0,0]/np.log(10)

def sample_PSD_jeffreys_num(tau, lxmin, lxmax, Ninter=500):
    """
    Sample numerically from the distribution: exp(-tau/x) * tau/x**2
    The distribution is truncated outside: 10**lxmin < x < 10**lxmax

    @param tau:     Scaling parameter
    @param lxmin:   Minimum bound (log10(xmin))
    @param lxmax:   Maximum bound (log10(xmax))
    @param Ninter:  Number of interpolation nodes in log(x)

    @return:    log10(x)
    """
    def Pfa(alpha, tau):
        return np.exp(-tau/np.exp(alpha)) / np.exp(alpha)

    return sample_func_num(Pfa, tau, lxmin, lxmax, Ninter)

def sample_PSD_sqrtjeffreys_num(tau, lxmin, lxmax, Ninter=500):
    """
    Sample numerically from the distribution: exp(-tau/x) * tau/x**(3/2)
    The distribution is truncated outside: 10**lxmin < x < 10**lxmax

    @param tau:     Scaling parameter
    @param lxmin:   Minimum bound (log10(xmin))
    @param lxmax:   Maximum bound (log10(xmax))
    @param Ninter:  Number of interpolation nodes in log(x)

    @return:    log10(x)
    """
    def Pfa(alpha, tau):
        return np.exp(-tau/np.exp(alpha)) / np.exp(0.5*alpha)

    return sample_func_num(Pfa, tau, lxmin, lxmax, Ninter)

def sample_PSD_flat_num(tau, lxmin, lxmax, Ninter=500):
    """
    Sample numerically from the distribution: exp(-tau/x) * tau/x
    The distribution is truncated outside: 10**lxmin < x < 10**lxmax

    @param tau:     Scaling parameter
    @param lxmin:   Minimum bound (log10(xmin))
    @param lxmax:   Maximum bound (log10(xmax))
    @param Ninter:  Number of interpolation nodes in log(x)

    @return:    log10(x)
    """
    def Pfa(alpha, tau):
        return np.exp(-tau/np.exp(alpha))

    return sample_func_num(Pfa, tau, lxmin, lxmax, Ninter)
