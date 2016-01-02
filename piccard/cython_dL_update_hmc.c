/* cython_dL_update_hmc.c
 *
 * Rutger van Haasteren, December 12 2015, Pasadena
 *
 */

#include <stdio.h>
#include <stdlib.h>


/* The aggregated algorithm for use in the Hamiltonian Sampler */
static void dL_update_hmc(double *pdL, double *pdLi, double *pdp,
                          double *pdM, double *pdtj, int N) {
    /*
    Formal derivative of rank-one update of Cholesky decomposition,
    adjusted to perform all rank-one updates at once for the derivative
    
    L'L'^{T} = LL^{T} + diag(B)
    dL' = L Phi(L^{-1} dB L^{-T})  With Phi the utril function
    We need: 
    
    L is the Cholesky factor of C = C(a) = C(a(t))
    C(a(0)) = LL^{T},   C(a(t)) = L'L'^{T}
    u is a vector, and a(t) is a scalar function of t
    
    Re-parameterized: also works in the limit where a->0

    :param pdL:     Current updated Cholesky decomposition (L-prime)
    :param pdLi:    Inverse of Cholesky decomposition (L^{-1})
    :param pdp:     Vector we'll need to multiply dL with
    :param pdM:     The return matrix M   (output)
    :param pdtj:    The return vector tj  (output)
    :param N:       Size of all the objects
    */
    double *pdLdot, *pdU;
    double tmp, r, *pdrdot, *pdcdot, *pds;
    int i, j, k;

    pdLdot = calloc(N*N,sizeof(double));
    pdU= calloc(N*N,sizeof(double));
    pdrdot = calloc(N,sizeof(double));
    pdcdot = calloc(N,sizeof(double));
    pds = calloc(N,sizeof(double));

    /* Initialize all our quantities */
    for(i=0; i<N; ++i) {
        pdU[i+N*i] = 1.0;

        for(j=0; j<N; ++j) {
            pdM[j+N*i] = 0.0;
        } /* for j */
        pdtj[i] = 0.0;
    } /* for i */

    for(k=0; k<N; ++k) {
        r = pdL[k+N*k];
        
        for(i=0; i<N; ++i) {
            /* Initialize the vector quantities */
            pdrdot[i] = 0.5*pdU[i+N*k]*pdU[i+N*k] / r;
            pdcdot[i] = pdrdot[i]/pdL[k+N*k];
            pds[i] = pdU[i+N*k] / pdL[k+N*k];

            /* Clear Ldot data */
            /* INEFFICIENT: if-statement outside of loop */
            if(k > 0) {
                pdLdot[i+N*(k-1)] = 0.0;
            } /* if k */
            pdLdot[i+N*k] = pdrdot[i];
        } /* for i */

        /* Update Ldot */
        for(i=k+1; i<N; ++i) {
            for(j=0; j<N; ++j) {
                pdLdot[j+N*i] = pds[j]*pdU[j+N*i]
                                - pdcdot[j] * pdL[k+N*i];
            } /* for j */
        } /* for i */

        /* Update U */
        for(i=k+1; i<N; ++i) {
            tmp = pdL[k+N*i];
            for(j=0; j<N; ++j) {
                /*pdU[j+N*i] = pdU[j+N*i] - pds[j]*pdL[k+N*i];*/
                pdU[j+N*i] = pdU[j+N*i] - pds[j]*tmp;
            } /* for j */
        } /* for i */

        /* Update M.   TODO: Make this a BLAS call? */
        /*for(i=k; i<N; ++i) {*/
        for(i=0; i<N; ++i) {
            /*for(j=0; j<N; ++j) {*/
            for(j=k; j<N; ++j) {
                /* pdM[j+N*i] += pdLdot[j+N*i]*pdp[k]; */
                pdM[i+N*k] += pdLdot[i+N*j]*pdp[j];
            } /* for j */
        } /* for i */

        /* Update tj.  TODO: Make this a BLAS call? */
        for(j=0; j<N; ++j) {
            tmp = pdLi[j+N*k];
            for(i=0; i<N; ++i) {
                pdtj[i] += tmp*pdLdot[i+j*N];
            } /* for i */
        } /* for j */

    } /* for k */

    free(pdLdot);
    free(pdU);
    free(pdrdot);
    free(pdcdot);
    free(pds);
    return;
} /* dL_update_hmc */

