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

    def __init__(self, nparticles, xmin, xmax, logp, \
            c1=1.193, c2=1.193, w=0.72, bufsize=50):
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

        # Chain has indices: (sample index, particle, parameter)
        self.chain = np.zeros( (bufsize, nparticles, len(xmin)))
        self.cursample = 0
        self.fullbuf = False
        self.bufsize = bufsize

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

    def updateChain(self):
        """
        Update the chain-buffer, after we have iterated once. The chain buffer
        keeps track of the last few steps so that we can check convergence
        """
        for ii, particle in enumerate(self.swarm):
            self.chain[self.cursample, ii, :] = particle.x

        self.cursample += 1

        if self.cursample == self.bufsize:
            self.cursample = 0
            self.fullbuf = True


    def iterateOnce(self):
        """
        Evolve all particles one step
        """
        for ii, particle in enumerate(self.swarm):
            particle.sample(self.c1, self.c2, self.w, self.bestx)
            self.updateMaxFromParticle(particle)

        self.updateChain()


    def Rhat_try1(self):
        """
        Does not work
        """
        npars = len(self.xmin)
        Rhat = np.inf * np.ones(npars)
        n = self.bufsize

        if self.fullbuf:
            for pp in range(npars):
                # chain has order: (sample, particle, parameter)
                samples = self.chain[:, :, pp]
                chain_means = np.mean(samples, axis=0)
                between_var = n * np.var(chain_means, ddof=1)
                chain_var = np.var(samples, axis=0, ddof=1)
                within_var = np.mean(chain_var)

                marg_post_var = ((n-1.0)/n) * within_var + (1.0/n) * between_var
                Rhat[pp] = np.sqrt(marg_post_var / within_var)


        return Rhat

    def Rhat_try2(self):
        """
        Same result as above
        """
        npars = len(self.xmin)
        Rhat = np.inf * np.ones(npars)
        n = self.bufsize
        M = self.nparticles

        if self.fullbuf:
            for pp in range(npars):
                # chain has order: (sample, particle, parameter)
                samples = self.chain[:, :, pp]
                mean_theta_m = np.mean(samples, axis=0)

                if len(mean_theta_m) != M:
                    raise ValueError("FOUT: {0}".format(len(mean_theta_m)))

                mean_theta = np.mean(mean_theta_m)
                B = (n / (M-1.0)) * np.sum( (mean_theta_m - mean_theta)**2 )

                ssqr_m = (1.0 / (n - 1)) * np.sum( (samples - \
                        mean_theta_m * np.ones(samples.shape))**2 )

                W = np.mean(ssqr_m)

                Vhat = (n-1.0) * W / n + (M+1) * B / (n*M)

                print "W, B = ", W, B

                Rhat[pp] = np.sqrt(Vhat / W)

        return Rhat

    def Rhat(self):
        """
        Calculate the potential scale reduction factor (PSRF)

        Note: why do m and n seem to be swapped? Why does this function work?
        """
        npars = len(self.xmin)
        R_hat = np.inf * np.ones(npars)
        n = self.bufsize
        m = self.nparticles

        if self.fullbuf:
            for pp in range(npars):
                # chain has order: (sample, particle, parameter)
                samples = self.chain[:, :, pp]

                # Number of chains (m) and number of samples (n)
                m, n = np.shape(samples)

                # Chain variance
                chain_var = np.var(samples, axis=1, ddof=1) # degrees of freedom = n-ddof

                # Within-chain variance (mean of variances of each chain)
                W = 1./m * np.sum(chain_var)

                # Chain means
                chain_means = np.mean(samples, axis=1)

                # Variance of chain means
                chain_means_var = np.var(chain_means, ddof=1)

                # Between-chain variance
                B = n * chain_means_var

                # Weighted average of within and between variance
                #(marginal posterior variance)
                Var_hat = (float(n-1)/n)*W + B/n

                # Potential scale reduction factor
                R_hat[pp] = np.sqrt(Var_hat / W)
	
	return R_hat


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
        nparticles = int(ndim**2/2) + 5*ndim

    print("Running a PSO in {0} dimensions with {1} particles".format(\
            ndim, nparticles))
    print("")

    swarm = Swarm(nparticles, likob.pmin, likob.pmax, likob.logposterior)
    for ii in range(iterations):
        swarm.iterateOnce()

        if ii % 10 == 0:
            sys.stdout.write("\r {0}: {1}, {2}".format(ii, swarm.bestx[:2], swarm.bestL))
            sys.stdout.flush()

        if np.all(swarm.Rhat() < 1.02):
            sys.stdout.write("\nConverged!\n")
            break

    sys.stdout.write("\nDone\n")
    print swarm.bestx[:10], swarm.bestL

    out = np.append([swarm.bestL], swarm.bestx)

    np.savetxt(chainsdir+'/pso.txt', out)




if __name__ == '__main__':
    ndim = 2
    minx = np.ones(ndim) * -900.0
    maxx = np.ones(ndim) * 900.0
    nparticles = 10
    #w = -0.2089
    #c1 = 1.193
    #c1 = 1.193
    iterations = 1000

    print("Running a PSO in {0} dimensions with {1} particles".format(\
            ndim, nparticles))
    print("")

    swarm = Swarm(nparticles, minx, maxx, loglikelihood)
    for ii in range(iterations):
        swarm.iterateOnce()

        Rhat = swarm.Rhat()

        if ii % 10 == 0:
            sys.stdout.write("\r {0}: {1}, {2}, {3}".format( \
                    ii, swarm.bestx[:2], swarm.bestL, np.max(Rhat)))
            sys.stdout.flush()

        if np.all(Rhat < 1.02):
            sys.stdout.write("\nConverged!")
            break

    sys.stdout.write("\nDone\n")

    print swarm.bestx[:10], swarm.bestL

    #print swarm.Rhat2() - swarm.Rhat()
    print swarm.Rhat()
