from __future__ import division
import numpy as np

"""
Some short functions that calculate the autocorrelation function
"""


def acf(x, t):
    """
    Calculate the autocorrelation for a given lag

    @param x:   Array of samples
    @param t:   time-lag in steps
    """
    xmean = np.mean(x)
    N = len(x)

    num = np.sum( (x[t:] - xmean) * (x[:-t] - xmean) ) / (N - 1)
    den = np.sum( (x - xmean)**2 ) / (N - 1)

    return num / den

def getacf(x, tv):
    """
    Get the autocorrelation function for a specific set of lags
    
    @param x:   Array of samples
    @param tv:  Array of time-lags in steps
    """
    return np.array([acf(x, t) for t in tv])
