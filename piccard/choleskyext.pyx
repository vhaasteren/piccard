cimport numpy as np
import numpy as np
cimport scipy.linalg as sl
import scipy.linalg as sl

cdef extern from "cython_dL_update_hmc.c":
    void dL_update_hmc(double *pdL, double *pdLi, double *pdp, double *pdM, double *pdtj, int N) 

cdef void cython_dL_update_hmc(
        np.ndarray[np.double_t,ndim=2] L,
        np.ndarray[np.double_t,ndim=2] Li,
        np.ndarray[np.double_t,ndim=1] p,
        np.ndarray[np.double_t,ndim=2] M,
        np.ndarray[np.double_t,ndim=1] tj):
    
    #cdef np.ndarray[np.double_t,ndim=2] Li = \
    #    np.ascontiguousarray(sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True))
    cdef int N = len(p)

    assert L.shape[0] == L.shape[1]
    
    assert L.shape[0] == len(p)

    dL_update_hmc(&L[0,0], &Li[0,0], &p[0], &M[0,0], &tj[0], N)

def cython_dL_update(L, Li, p):
    M = np.zeros_like(L, order='C')
    tj = np.zeros(len(L))
    cython_dL_update_hmc(np.ascontiguousarray(L), np.ascontiguousarray(Li), p, M, tj)
    return M, tj

# The aggregated algorithm for use in the Hamiltonian Sampler
def python_dL_update(L, Li, p):
    """
    Formal derivative of rank-one update of Cholesky decomposition,
    adjusted to perform all rank-one updates at once for the derivative
    
    L'L'^{T} = LL^{T} + diag(B)
    dL' = L Phi(L^{-1} dB L^{-T})  With Phi the utril function
    We need: 
    
    L is the Cholesky factor of C = C(a) = C(a(t))
    C(a(0)) = LL^{T},   C(a(t)) = L'L'^{T}
    u is a vector, and a(t) is a scalar function of t
    
    Re-parameterized: also works in the limit where a->0

    :param L:       Current updated Cholesky decomposition (L-prime)
                    Really needs to be lower-diagonal
    :param p:       Vector we'll need to multiply dL with
    """
    n = len(L)
    Ldot = np.zeros_like(L)
    
    # Bahh, this is another ms or so for N=300. Get through argument?
    # Li = sl.solve_triangular(L, np.eye(len(L)), trans=0, lower=True)

    U = np.eye(n)
    M = np.zeros_like(L)
    tj = np.zeros(n)
    
    # The index k represents the row of Ldot we are working with
    for k in range(n):
        # For efficiency, collect these values
        r = L[k,k]
        rdot = 0.5*U[k,:]**2 / r
        cdot = rdot/L[k,k]
        s = U[k,:] / L[k,k]
        
        Ldot[k-1,:] = 0.0    # Clear all data from the previous iteration
        Ldot[k,:] = rdot

        Ldot[k+1:n,:] = U[None,k,:]*U[k+1:,:]/L[k,k] - 0.5*U[None,k,:]**2*L[k+1:,k,None]/L[k,k]**2

        # The change of u does not depend on udot (or adot), so it's the same for all derivatives
        U[k+1:n,:] = U[k+1:n,:] - s[None,:]*L[k+1:n,k,None]
        
        # At this point, Ldot contains the k-th column of the Cholesky factor for all the basis vectors
        # Ldot[:,i] = dL_i[:,k]
        
        # The i-th column of M: np.dot(dL3[:,:,i], p)
        # So... M[k,i] = sum_j dL3[k,j,i] * p[j]
        #M += Ldot * p[k]       # This was for the L-version
        M[k,:] = np.dot(Ldot.T, p)
        
        #tj += np.sum(Li[k,:,None] * Ldot, axis=0)
        tj += np.dot(Li[k, :], Ldot)

    return M, tj

def cholesky_setup():
    Nobs = 229                        # Number of timing-model parameters (the name Nobs doesn't make sense)
    X = np.random.randn(Nobs, Nobs)         # A random matrix
    u = np.random.randn(Nobs)               # A random basis vector (not normalized)
    Cov_0 = np.dot(X, X.T)                  # A pos-def covariance matrix
    L_0 = sl.cholesky(Cov_0, lower=True)    # Cholesky factor of Cov_0
    t = np.linspace(1, 2, Nobs) 

    return L_0, u
