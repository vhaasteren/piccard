from __future__ import division
import numpy as np
import math
import scipy.special as ss
import scipy.misc as sm

"""
Script to compute the correlation basis-functions for various anisotropic
configurations of the GW background energy-density, in the sYlm basis

-- Rutger van Haasteren (October 2015)

"""


def real_sph_harm(mm, ll, phi, theta):
    """
    The real-valued spherical harmonics.
    """
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

def Nl(ll):
    return np.sqrt(2.0 * sm.factorial(ll-2) / sm.factorial(ll+2))


def CorrBasis(psr_locs, lmax):
    """
    Calculate the correlation basis matrices in the sYlm basis. Normalized
    version of:

    Gair et al. (2014), Physical Review D, Volume 90, Issue 8, id.082001
    Equation (111)

    Pulsar term included (normalized by (2*ll+1))

    @param psr_locs:    Location of the pulsars [phi, theta]
    @param lmax:        Maximum l to go up to
    """
    assert lmax >= 2

    npsrs = len(psr_locs)
    pphi = psr_locs[:,0]
    ptheta = psr_locs[:,1]

    # Given the pulsar locations, we can create the new basis matrices
    basis = []
    nclm = (lmax+1)**2 - 4
    for ll in range(2, lmax+1):
        for mm in range(-ll, ll+1):
            mat = np.zeros((npsrs, npsrs))
            for aa in range(npsrs):
                for bb in range(aa, npsrs):
                    mat[aa, bb] = 3.0*np.pi*Nl(ll)**2 * (
                        real_sph_harm(mm, ll, pphi[aa], ptheta[aa]) *
                        real_sph_harm(mm, ll, pphi[bb], ptheta[bb]))
                    if aa == bb:
                        mat[aa, bb] += 1.5*np.pi*Nl(ll)**2 / (2*ll+1)
                    else:
                        mat[bb, aa] = mat[aa, bb]
            basis.append(mat)

    return basis
