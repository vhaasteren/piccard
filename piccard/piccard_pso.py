#!/usr/bin/env python

from __future__ import division

import sys
import os
import numpy as np


class Particle(object):

    def __init__(self, xmin, xmax, logp):
        if len(xmin) != len(xmax):
            raise ValueError("xmin and xmax not same length")

        self.n = len(xmin)
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.vmax = 0.5 * (self.xmax - self.xmin)
        self.v = (2*np.random.rand(self.n) - 1.0) * self.vmax
        self.x = self.xmin + np.random.rand(self.n) * (self.xmax - self.xmin)

        self.logp = logp
        self.logL = self.logp(self.x)
        self.bestL = self.logL
        self.bestx = self.x.copy()


    def sample(self, c1, c2, w, bestg):
        """
        Take one step in parameter space
        """
        oldx = self.x.copy()
        oldv = self.v.copy()

        u = np.random.rand(2)
        self.v = w*oldv + \
                c1*u[0]*(self.bestx - oldx) + \
                c2*u[1]*(bestg - oldx)
        self.x = oldx + self.v

        self.checkBoundaries()

        self.logL = self.logp(self.x)
        if self.logL > self.bestL:
            self.bestL = self.logL
            self.bestx = self.x.copy()


    def checkBoundaries(self):
        """
        Check whether the current position is within the boundaries. If not,
        reflect both the position and velocity.
        """

        indmin = self.x < self.xmin
        if np.sum(indmin) > 0:
            self.x[indmin] = 2.0*self.xmin[indmin] - self.x[indmin]
            self.v[indmin] = -self.v[indmin]

        indmax = self.x > self.xmax
        if np.sum(indmax) > 0:
            self.x[indmax] = 2.0*self.xmax[indmax] - self.x[indmax]
            self.v[indmax] = -self.v[indmax]

        indvmin = self.v < -self.vmax
        if np.sum(indvmin):
            self.v[indvmin] = -self.vmax[indvmin]

        indvmax = self.v > self.vmax
        if np.sum(indvmax):
            self.v[indvmax] = self.vmax[indvmax]



class Swarm(object):

    def __init__(self, nparticles, xmin, xmax, logp, c1=1.193, c2=1.193, w=0.72):
        if len(xmin) != len(xmax):
            raise ValueError("xmin and xmax not same length")

        if nparticles < 1:
            raise ValueError("Number of particles needs to be > 1")

        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.nparticles = nparticles
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.bestL = -np.inf
        self.bestx = np.zeros(len(xmin))

        self.swarm = []

        for ii in range(nparticles):
            particle = Particle(xmin, xmax, logp)
            self.updateMaxFromParticle(particle)
            self.swarm.append(particle)

    def updateMaxFromSwarm(self):
        """
        Update the global maximum from a single particle
        """
        for ii, particle in enumerate(self.swarm):
            self.updateMaxFromParticle(particle)


    def updateMaxFromParticle(self, particle):
        """
        Update the global maximum from the whole swarm
        """
        if particle.bestL > self.bestL:
            self.bestL = particle.bestL
            self.bestx = particle.bestx.copy()

    def iterateOnce(self):
        """
        Evolve all particles one step
        """
        for ii, particle in enumerate(self.swarm):
            particle.sample(self.c1, self.c2, self.w, self.bestx)
            self.updateMaxFromParticle(particle)

    """
    def calcRhat(self, kabuki=True):

        chain = np.zeros((self.nparticles, len(self.xmin)))
        for ii, particle in enumerate(self.swarm):
            chain[:,ii] = particle.x
        Rhat = 10.0

        if kabuki:
            Rhat = self.R_hat_kabuki(chain)
        else:
            Rhat = self.R_hat_pymc(chain)

    # From kabuki/analyze.py, line 285 ("def R_hat")
    def R_hat_kabuki(self, samples):
        #n, num_chains = samples.shape # n=num_samples
        num_chains, n = samples.shape # CORRECTED
        chain_means = np.mean(samples, axis=1)
        # Calculate between-sequence variance
        between_var = n * np.var(chain_means, ddof=1)
        chain_var = np.var(samples, axis=1, ddof=1)
        within_var = np.mean(chain_var) # OK (=pymc)
        marg_post_var = ((n-1.)/n) * within_var + (1./n) * between_var # = pymc s2
        R_hat_sqrt = np.sqrt(marg_post_var/within_var)
        return R_hat_sqrt

    # pymc/diagnostics.py, line 450 ("def gelman_rubin")
    def R_hat_pymc(self, x):
        if np.shape(x) < (2,):
            raise ValueError('Gelman-Rubin diagnostic requires multiple chains.')
        try:
            m,n = np.shape(x)
        except ValueError:
            return [gelman_rubin(np.transpose(y)) for y in np.transpose(x)]
        # Calculate between-chain variance
        B_over_n = np.sum((np.mean(x,1) - np.mean(x))**2)/(m-1)
        # Calculate within-chain variances
        W = np.sum([(x[i] - xbar)**2 for i,xbar in enumerate(np.mean(x,1))]) / (m*(n-1)) # OK (=kabuki)
        # (over) estimate of variance
        s2 = W*(n-1)/n + B_over_n # = marg_post_var
        # Pooled posterior variance estimate
        V = s2 + B_over_n/m
        V = s2 # CORRECTED
        # Calculate PSRF
        R = V/W
        return R
    """


def loglikelihood(parameters):
    mu = 50.0
    sigma = 7.5

    #logl = -0.5*np.sum(((parameters-mu)/sigma)**2) - \
    #        0.5*len(parameters)*np.log(2*np.pi*sigma*sigma) + \
    #        12.34567

    # For now, just use this, since the maximum is exactly 0.0
    logl = -0.5*np.sum(((parameters-mu)/sigma)**2)
    return logl


def RunPSO(likob, chainsdir, nparticles=0, iterations=500):
    ndim = likob.dimensions

    if nparticles == 0:
        nparticles = ndim * 10

    print("Running a PSO in {0} dimensions with {1} particles".format(\
            ndim, nparticles))
    print("")

    swarm = Swarm(nparticles, likob.pmin, likob.pmax, likob.logposterior)
    for ii in range(iterations):
        swarm.iterateOnce()

        if ii % 10 == 0:
            sys.stdout.write("\r {0}: {1}, {2}".format(ii, swarm.bestx[:6], swarm.bestL))
            sys.stdout.flush()

    sys.stdout.write("\nDone\n")
    print swarm.bestx[:10], swarm.bestL

    out = np.append([swarm.bestL], swarm.bestx)

    np.savetxt(chainsdir+'/pso.txt', out)




if __name__ == '__main__':
    ndim = 10
    minx = np.ones(ndim) * -900.0
    maxx = np.ones(ndim) * 900.0
    nparticles = 100
    #w = -0.2089
    #c1 = 1.193
    #c1 = 1.193
    iterations = 100

    print("Running a PSO in {0} dimensions with {1} particles".format(\
            ndim, nparticles))
    print("")

    swarm = Swarm(nparticles, minx, maxx, loglikelihood)
    for ii in range(iterations):
        swarm.iterateOnce()

        if ii % 10 == 0:
            sys.stdout.write("\r {0}: {1}, {2}".format(ii, swarm.bestx[:4], swarm.bestL))
            sys.stdout.flush()

    sys.stdout.write("\nDone\n")

    print swarm.bestx[:10], swarm.bestL
