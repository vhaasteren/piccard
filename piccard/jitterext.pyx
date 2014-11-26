import numpy as np
cimport numpy as np
from libc.math cimport log


def block_shermor_inv(r, Nvec, Jvec, select):
    """
    Sherman-Morrison block-inversion for Jitter

    @param r:       The timing residuals, array (n)
    @param Nvec:    The white noise amplitude, array (n)
    @param Jvec:    The jitter amplitude, array (k)
    @param select:  The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    Nir = np.zeros(len(r))
    nepochs = len(Jvec)

    ni = 1.0 / Nvec
    ji = 1.0 / Jvec
    Jldet = np.einsum('i->', np.log(Nvec))
    xNx = np.dot(r, r * ni)

    for cc in range(nepochs):
        rblock = r[select[cc,0]:select[cc,1]]
        niblock = ni[select[cc,0]:select[cc,1]]

        beta = 1.0 / (np.einsum('i->', niblock)+ji[cc])
        xNx -= beta * np.dot(rblock, niblock)**2
        Jldet += np.log(Jvec[cc]) - np.log(beta)

    return Jldet, xNx


def basic_block_shermor_inv(r, Nvec, Jvec, Umat):
    """
    Basic version of the Sherman-Morrison block-inversion for Jitter
    """
    Nir = np.zeros(len(r))
    Jldet = 0.0
    xNx = 0.0
    
    nobs = Umat.shape[0]
    nbs = Umat.shape[1]
    nbl = nobs / nbs
    
    for cc, col in enumerate(Umat.T):
        #u = (col == 1.0)
        Nblock = Nvec[cc*nbl:cc*(nbl+1)]
        rblock = r[cc*nbl:cc*(nbl+1)]
        
        ji = 1.0 / Jvec[cc]
        ni = 1.0 / Nblock
        beta = 1.0 / (np.sum(ni) + ji)

        #xNx += np.sum(rblock**2 / Nblock) - beta*np.sum(rblock*Nblock)**2
        xNx += np.dot(rblock, rblock * ni) - beta * np.dot(rblock, ni)**2
        
        #Jldet += np.sum(np.log(Nblock)) + np.log(Jvec[cc]) - np.log(beta)
        Jldet += np.einsum('i->', np.log(Nblock)) + np.log(Jvec[cc]) - np.log(beta)
        
    return Jldet, xNx


def basic_cython_block_shermor_inv( \
        np.ndarray[np.double_t,ndim=1] r, \
        np.ndarray[np.double_t,ndim=1] Nvec, \
        np.ndarray[np.double_t,ndim=1] Jvec, \
        np.ndarray[np.double_t,ndim=2] Umat, \
        int nblocks, int blocksize):
    """
    Basic version of the Sherman-Morrison block-inversion for Jitter
    (Cythonized version)
    """
    cdef unsigned int lenr = len(r)
    cdef unsigned int cc, ii, jj
    cdef np.ndarray[np.double_t,ndim=1] Nblock = np.random.randn(blocksize)
    cdef np.ndarray[np.double_t,ndim=1] rblock = np.random.randn(blocksize)
    cdef np.ndarray[np.double_t,ndim=1] ni = np.zeros(nblocks, 'd')
    cdef double Jldet=0.0, ji, beta, xNx=0.0, temp

    for cc in range(nblocks):
        ji = 1.0 / Jvec[cc]
        ni = 1.0 / Nblock
        beta = 1.0 / (np.sum(ni) + ji)

        temp = 0.0
        for ii in range(blocksize):
            Jldet += log(Nblock[ii])
            xNx += rblock[ii]*rblock[ii]*ni[ii]
            temp += rblock[ii]*ni[ii]
        Jldet += log(Jvec[cc]) - log(beta)
        xNx += beta * (temp**2)

    return Jldet, xNx

