#    Adjusted from DNest code, modified by Rutger van Haasteren
#
#    Copyright (C) 2011 Brendon J. Brewer
#    This file is part of DNest, the Diffusive Nested Sampler.
#
#    DNest is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    DNest is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#    along with DNest.  If not, see <http://www.gnu.org/licenses/>.


import copy as cpy
import numpy as np
import numpy.random as rng
import matplotlib

#import dataformat as df

# This is from the file dnestresults --- from here
from numpy import *
from matplotlib.pyplot import *
#from copy import deepcopy
from numpy.random import *
# until here
# TODO: change the upper imports above to the canonical form


class Model:
    """
    Abstract class that usable Models should inherit
    """
    def __init__(self):
        """
        Set the logLikelihood+tieBreaker tuple to nothing
        """
        self.logL = [None, None]

    def fromPrior(self):
        """
        Draw the parameters from the prior
        """
        self.logL[1] = rng.rand()

    def calculateLogLikelihood(self):
        """
        Define the likelihood function
        """
        self.logL[0] = 0.0

    def perturb(self):
        """
        Perturb, for metropolis
        """
        self.logL[1] += 10.0**(1.5 - 6.0*rng.rand())*rng.randn()
        self.logL[1] = np.mod(self.logL[1], 1.0)

    def update(self, level):
        """
        Do a Metropolis step wrt the given level
        """
        assert self.logL >= level.logL
        proposal = cpy.deepcopy(self)
        logH = proposal.perturb()
        if logH > 0.0:
            logH = 0.0
        if rng.rand() <= np.exp(logH) and proposal.logL >= level.logL:
            return [proposal, True]
        else:
            return [self, False]

class LikModel(Model):
    """
    Rutger's example model
    """
    lik = None
    logwidthrange = None

    def __init__(self, lik=None):
        if lik == None:
            raise IOError, "LikModel requires a likelihood object to initialise"

        lik.deleterequired = False
        Model.__init__(self)
        self.lik = lik

        self.numParams = self.lik.dimensions
        self.params = np.zeros(self.numParams)

        self.logwidthrange = np.ones((self.numParams, 2))
        self.logwidthrange[:,0] *= 0.0
        self.logwidthrange[:,1] *= -6.0

    def setMCMCWidthFromFile(self, chainfilename):
        chain = np.loadtxt(chainfilename)
        self.setMCMCWidth(chain[:,2:])

    def setMCMCWidth(self, chain):
        self.logwidthrange[:,0] = np.log10(\
                np.std(chain, axis=0)/(self.lik.pmax - self.lik.pmin)\
                ) + np.log10(3.0)
        self.logwidthrange[:,1] = self.logwidthrange[:,0] - 5.0

    def fromPrior(self):
        """
        Generate all parameters iid from U(-0.5, 0.5)
        """
        self.params = self.lik.samplefromprior()
        Model.fromPrior(self)
        self.calculateLogLikelihood()

    def perturb(self):
        """
        Metropolis proposal: perturb one parameter
        """
        logH = 0.0
        which = rng.randint(self.params.size)

        # Take a step
        #self.params[which] += (self.lik.pmax[which] - self.lik.pmin[which]) *\
        #        10**(0.0 - 5.0*rng.rand())*rng.randn()
        self.params[which] += (self.lik.pmax[which] - self.lik.pmin[which]) *\
                10**(self.logwidthrange[which,0] - self.logwidthrange[which,1]*rng.rand())*rng.randn()

        # Modulate the step
        self.params[which] = np.mod(self.params[which] - self.lik.pmin[which],\
                self.lik.pmax[which] - self.lik.pmin[which]) + self.lik.pmin[which]

        Model.perturb(self)
        self.calculateLogLikelihood()
        return logH

    def calculateLogLikelihood(self):
        """
        One Gaussian, at 0.25, with width 0.1
        """
        pars = self.lik.pstart

        self.logL[0] = self.lik.logposterior(self.params)

    def __str__(self):
        return "".join(str(i) + " " for i in self.params)


class Level:
    """
    Defines a Nested Sampling level
    """
    def __init__(self, logX=0.0, logL=[-1E300, 0.0]):
        """
        Construct a level. By default, logX = 0
        and logL = [-1E300, 0.0]. i.e. the prior.
        I use -1E300 for compatibility with the C++ version.
        """
        self.logX, self.logL = logX, logL
        self.accepts = 0
        self.tries = 0
        self.exceeds = 0
        self.visits = 0

    def renormaliseVisits(regularisation):
        """
        Make level stats of order `regularisation`
        """
        if self.tries >= regularisation:
            self.accepts = int(float(self.accepts+1)/(self.tries+1)*regularisation)
            self.tries = regularisation
        if self.visits >= regularisation:
            self.exceeds = int(float(self.exceeds+1)/(self.visits+1)*regularisation)
            self.visits = regularisation

    def __str__(self):
        """
        Represent the level as a string
        """
        s = str(self.logX) + " " + str(self.logL[0]) + " " \
            + str(self.logL[1]) + " "\
            + str(self.accepts) + " "\
            + str(self.tries) + " " + str(self.exceeds) + " "\
            + str(self.visits) + " "
        return s

class LevelSet:
    """
    Defines a set of levels. Implemented as a list
    """
    def __init__(self, filename=None):
        """
        Optional: load from file `filename`
        """
        self.levels = []
        self.logLKeep = [] # Accumulation, for making new levels

        if filename == None:
            # Start with one level, the prior
            self.levels.append(Level())
        else:
            f = open('levels.txt', 'r')
            lines = f.readlines()
            for l in lines:
                stuff = l.split()
                level = Level(logX=float(stuff[0])\
                ,logL=[float(stuff[1]), float(stuff[2])])
                level.accepts = int(stuff[3])
                level.tries = int(stuff[4])
                level.exceeds = int(stuff[5])
                level.visits = int(stuff[6])
                self.levels.append(level)
            f.close()

    def updateAccepts(self, index, accepted):
        """
        Input: `index`: which level particle was in
        `accepted`: whether it was accepted or not
        """
        self.levels[index].accepts += int(accepted)
        self.levels[index].tries += 1

    def updateExceeds(self, index, logL):
        """
        Input: `index`: which level particle is in
        logL: its logLikelihood
        """
        if index < (len(self.levels)-1):
            self.levels[index].visits += 1
            if logL >= self.levels[index+1].logL:
                self.levels[index].exceeds += 1

    def recalculateLogX(self, regularisation):
        """
        Re-estimate the logX values for the levels
        using the exceeds/visits information.
        """
        self.levels[0].logX = 0.0
        q = np.exp(-1.0)
        for i in xrange(1, len(self.levels)):
            self.levels[i].logX = self.levels[i-1].logX \
                + np.log(float(self.levels[i-1].exceeds + q*regularisation)/(self.levels[i-1].visits + regularisation))

    def renormaliseVisits(self, regularisation):
        """
        Reset all visits, exceeds etc to be of order
        regularisation
        """
        for level in self.levels:
            level.renormaliseVisits(regularisation)

    def updateLogLKeep(self, logL):
        """
        If the logLikelihood is above the highest level,
        store it.
        Input: logLikelihood seen
        """
        if logL > self.levels[-1].logL:
            self.logLKeep.append(logL)

    def maybeAddLevel(self, newLevelInterval):
        added = False
        if len(self.logLKeep) >= newLevelInterval:
            self.logLKeep = sorted(self.logLKeep)
            index = int(0.63212*len(self.logLKeep))
            print("# Creating level " + str(len(self.levels))\
                + " with logL = "\
                + str(self.logLKeep[index][0]))
            newLevel = Level(self.levels[-1].logX - 1.0,\
                    self.logLKeep[index])
            self.levels.append(newLevel)
            self.logLKeep = self.logLKeep[index+1:]
            added = True
            if len(self.logLKeep) == newLevelInterval:
                self.renormaliseVisits(newLevelInterval)
        return added

    def save(self, filename='levels.txt'):
        """
        Write out all of the levels to a text file.
        Default filename='levels.txt'
        """
        f = open(filename, 'w')
        f.write(str(self))
        f.close()

    def __getitem__(self, i):
        """
        This is like overloading operator [] (LevelSet, int)
        """
        return self.levels[i]

    def __str__(self):
        """
        Put all levels in a single string, each level on a line
        """
        return "".join([str(l) + '\n' for l in self.levels])

    def __len__(self):
        """
        Return number of levels
        """
        return len(self.levels)


class Options:
    """
    DNest Options
    """
    def __init__(self, numParticles=1, newLevelInterval=10000,\
            saveInterval=10000, maxNumLevels=100, lamb=10.0,\
            beta=10.0, deleteParticles=True, maxNumSaves=np.inf):
        self.numParticles = numParticles
        self.newLevelInterval = newLevelInterval
        self.saveInterval = saveInterval				
        self.maxNumLevels = maxNumLevels
        self.lamb = lamb
        self.beta = beta
        self.deleteParticles = deleteParticles
        self.maxNumSaves = maxNumSaves

    def load(self, filename="OPTIONS"):
        opts = np.loadtxt(filename, dtype=int)
        self.numParticles = opts[0]
        self.newLevelInterval = opts[1]
        self.saveInterval = opts[2]
        self.maxNumLevels = opts[3]
        self.lamb = float(opts[4])
        self.beta = float(opts[5])
        self.deleteParticles = bool(opts[6])
        self.maxNumSaves = opts[7]
        if self.maxNumSaves == 0:
            self.maxNumSaves = np.inf

class Sampler:
    """
    A single DNest sampler.
    """
    def __init__(self, ModelType, options=Options(), levelsFile=None\
            ,sampleFile="sample.txt"\
            ,sampleInfoFile="sample_info.txt"\
            ,mcmcFile=None, arg=None):
        """
        Input: The class to be used
        Optional: `options`: Options object
        `levelsFile`: Filename to load pre-made levels from
        `sampleFile`: Filename to save samples to
        `sampleInfoFile`: Filename to save sample info to
        """
        self.options = options
        self.models = [ModelType(arg)\
                for i in xrange(0, options.numParticles)]

        if mcmcFile != None:
            for m in self.models:
                m.setMCMCWidthFromFile(mcmcFile)

        self.indices = [0 for i in xrange(0, options.numParticles)]
        self.levels = LevelSet(levelsFile)
        self.initialised = False # Models have been fromPriored?
        self.steps = 0 # Count number of MCMC steps taken

        self.sampleFile = sampleFile
        self.sampleInfoFile = sampleInfoFile
        # Empty the files
        f = open(self.sampleFile, "w")
        f.close()
        f = open(self.sampleInfoFile, "w")
        f.close()
        self.levels.save()

    def initialise(self):
        """
        Initialise the models from the prior
        """
        for which in xrange(0, self.options.numParticles):
            self.models[which].fromPrior()
            if len(self.levels) < self.options.maxNumLevels:
                self.levels.updateLogLKeep(self.models[which].logL)
                self.levels.maybeAddLevel(\
                    self.options.newLevelInterval)
        self.initialised = True

    def run(self):
        """
        Run forever.
        """
        while True:
            self.step()

    def step(self, numSteps=1):
        """
        Take numSteps steps of the sampler. default=1
        """
        if not self.initialised:
            self.initialise()

        for i in xrange(0, numSteps):
            which = rng.randint(self.options.numParticles)
            if rng.rand() <= 0.5:
                self.updateIndex(which)
                self.updateModel(which)
            else:
                self.updateModel(which)
                self.updateIndex(which)
            self.steps += 1

            self.levels.updateExceeds(self.indices[which],\
                self.models[which].logL)

            if len(self.levels) < self.options.maxNumLevels:
                # Accumulate logLKeep, possibly make a new level
                self.levels.updateLogLKeep(self.models[which].logL)
                added = self.levels.maybeAddLevel(\
                    self.options.newLevelInterval)
                if added:
                    self.deleteModel()
                    self.levels.recalculateLogX(self.options.newLevelInterval)
                    self.levels.save()

            if self.steps%self.options.saveInterval == 0:
                # Save a particle and the levels
                print("# Saving a particle. N = "\
                + str(self.steps/self.options.saveInterval) + ".")
                f = open('sample.txt', 'a')
                f.write(str(self.models[which]) + '\n')
                f.close()
                f = open('sample_info.txt', 'a')
                f.write(str(self.indices[which]) + " " \
                + str(self.models[which].logL[0]) + " "\
                + str(self.models[which].logL[1]) + " "\
                + str(which) + "\n")
                f.close()
                self.levels.recalculateLogX(self.options.newLevelInterval)
                self.levels.save()

    def updateModel(self, which):
        """
        Move a particle
        """
        [self.models[which], accepted] = self.models[which]\
            .update(self.levels[self.indices[which]])
        self.levels.updateAccepts(self.indices[which], accepted)

    def updateIndex(self, which):
        """
        Move which level a particle is in
        """
        delta = np.round(10.0**(2.0*rng.rand())*rng.randn())\
                .astype(int)
        if delta == 0:
            delta = 2*rng.randint(2) - 1
        proposed = self.indices[which] + delta
        # Immediate reject if proposed index was out of bounds
        if proposed < 0 or proposed >= len(self.levels):
            return

        # Acceptance probability
        logAlpha = self.levels[self.indices[which]].logX\
            - self.levels[proposed].logX \
            + self.logPush(proposed)\
            - self.logPush(self.indices[which])
        if logAlpha > 0.0:
            logAlpha = 0.0

        if rng.rand() <= np.exp(logAlpha) and \
            self.models[which].logL >= self.levels[proposed].logL:
            self.indices[which] = proposed

    def updateVisits(self, which):
        """
        Update visits/exceeds level statistics
        and logLKeep
        """
        if self.models[which].logL > self.levels[-1].logL\
            and len(self.levels) < self.options.maxNumLevels:
            self.logLKeep.append(self.models[which].logL)

        index = self.indices[which]
        if index < len(self.levels) - 1:
            self.levels[index].visits += 1
            if self.models[which].logL > self.levels[index+1].logL:
                self.levels[index].exceeds += 1

    def deleteModel(self):
        for i in xrange(0, len(self.models)):
            if (len(self.levels) - 1 - self.indices[i]) > (5*self.options.lamb + 1) and self.options.numParticles > 1:
                copy = rng.randint(self.options.numParticles)
                while copy == i:
                    copy = rng.randint(self.options.numParticles)
                self.models[i] = self.models[copy]
                self.indices[i] = self.indices[copy]
                print("# Deleted a particle. Replacing it with a copy of a survivor.")

    def logPush(self, index):
        """
        Calculate the relative weighting of levels,
        for acceptance probability for Sampler.updateIndex()
        """
        assert index >= 0 and index < len(self.levels)

        result = 0.0
        if len(self.levels) >= self.options.maxNumLevels:
            # All levels made, do uniform exploration, use beta
            result -= self.options.beta*np.log(self.levels[index].tries + 1)
        else:
            # All levels not made, do pushed-up exploration,
            # ignore beta
            distance = len(self.levels) - 1 - index
            result = -distance/self.options.lamb
            if not self.options.deleteParticles:
                if result <= -5.0:
                    result = -5.0
        
        return result

    def saveLevels(self, filename="levels.txt"):
        """
        Save the level structure to a file
        default: levels.txt
        """
        self.levels.save(filename)


# Coming from dnestrestults from here until the end
def logsumexp(values):
	biggest = max(values)
	x = values - biggest
	result = log(sum(exp(x))) + biggest
	return result

def logdiffexp(x1, x2):
	biggest = x1
	xx1 = x1 - biggest
	xx2 = x2 - biggest
	result = log(exp(xx1) - exp(xx2)) + biggest
	return result


"""
TODO: give this thing some arguments
"""
def dnestresults():
    numResampleLogX = 1
    temperature = 1.0

    levels = atleast_2d(loadtxt("levels.txt"))
    sample_info = atleast_2d(loadtxt("sample_info.txt"))
    sample = atleast_2d(loadtxt("sample.txt"))

    if sample.shape[0] != sample_info.shape[0]:
        print('ERROR. Size mismatch.')
        exit()

    ion()
    figure(1)
    plot(sample_info[:,0])
    xlabel("Iteration")
    ylabel("Level")
    draw()

    figure(2)
    subplot(2,1,1)
    plot(diff(levels[:,0]))
    ylabel("Compression")
    xlabel("Level")
    subplot(2,1,2)
    plot(levels[:,3]/levels[:,4])
    ylim([0, 1])
    xlabel("Level")
    ylabel("MH Acceptance")
    draw()

    # Convert to lists of tuples
    logl_levels = [(levels[i,1], levels[i, 2]) for i in range(0, levels.shape[0])] # logl, tiebreaker
    logl_samples = [(sample_info[i, 1], sample_info[i, 2], i) for i in range(0, sample.shape[0])] # logl, tiebreaker, id
    logx_samples = zeros((sample_info.shape[0], numResampleLogX))
    logp_samples = zeros((sample_info.shape[0], numResampleLogX))
    logP_samples = zeros((sample_info.shape[0], numResampleLogX))
    P_samples = zeros((sample_info.shape[0], numResampleLogX))
    logz_estimates = zeros((numResampleLogX, 1))
    H_estimates = zeros((numResampleLogX, 1))

    # Find sandwiching level for each sample
    sandwich = int64(sample_info[:,0])
    for i in range(0, sample.shape[0]):
        while sandwich[i] < levels.shape[0]-1 and logl_samples[i] > logl_levels[sandwich[i] + 1]:
            sandwich[i] += 1

    for z in range(0, numResampleLogX):
        # For each level
        for i in range(0, levels.shape[0]):
            # Find the samples sandwiched by this level
            which = nonzero(sandwich == i)[0]
            logl_samples_thisLevel = [] # (logl, tieBreaker, ID)
            for j in range(0, len(which)):
                logl_samples_thisLevel.append(cpy.deepcopy(logl_samples[which[j]]))
            logl_samples_thisLevel = sorted(logl_samples_thisLevel)
            N = len(logl_samples_thisLevel)

            # Generate intermediate logx values
            logx_max = levels[i, 0]
            if i == levels.shape[0]-1:
                logx_min = -1E300
            else:
                logx_min = levels[i+1, 0]
            Umin = exp(logx_min - logx_max)

            if N == 0 or numResampleLogX > 1:
                U = Umin + (1.0 - Umin)*rand(len(which))
            else:
                U = Umin + (1.0 - Umin)*linspace(1.0/(N+1), 1.0 - 1.0/(N+1), N)
            logx_samples_thisLevel = sort(logx_max + log(U))[::-1]

            for j in range(0, len(which)):
                logx_samples[logl_samples_thisLevel[j][2]][z] = logx_samples_thisLevel[j]

                if j != len(which)-1:
                    left = logx_samples_thisLevel[j+1]
                elif i == levels.shape[0]-1:
                    left = -1E300
                else:
                    left = levels[i+1][0]
                    
                if j != 0:
                    right = logx_samples_thisLevel[j-1]
                else:
                    right = levels[i][0]

                logp_samples[logl_samples_thisLevel[j][2]][z] = log(0.5) + logdiffexp(right, left)

        logl = sample_info[:,1]/temperature

        logp_samples[:,z] = logp_samples[:,z] - logsumexp(logp_samples[:,z])
        logP_samples[:,z] = logp_samples[:,z] + logl
        logz_estimates[z] = logsumexp(logP_samples[:,z])
        logP_samples[:,z] -= logz_estimates[z]
        P_samples[:,z] = exp(logP_samples[:,z])
        H_estimates[z] = -logz_estimates[z] + sum(P_samples[:,z]*logl)

        figure(3)
        clf()
        subplot(2,1,1)
        p1 = plot(logx_samples[:,z], sample_info[:,1], 'b.', label='Samples')
        p2 = plot(levels[1:,0], levels[1:,1], 'r.', label='Levels')
        legend(numpoints=1, loc='lower left')
        ylabel('log(L)')
        title(str(z+1) + "/" + str(numResampleLogX) + ", log(Z) = " + str(logz_estimates[z][0]))
        subplot(2,1,2)
        plot(logx_samples[:,z], P_samples[:,z], 'b.')
        xlabel('log(X)')
        ylabel('Weight posterior')
        draw()

    P_samples = mean(P_samples, 1)
    P_samples = P_samples/sum(P_samples)
    logz_estimate = mean(logz_estimates)
    logz_error = std(logz_estimates)
    H_estimate = mean(H_estimates)
    H_error = std(H_estimates)
    ESS = exp(-sum(P_samples*log(P_samples+1E-300)))

    print("log(Z) = " + str(logz_estimate) + " +- " + str(logz_error))
    print("Information = " + str(H_estimate) + " +- " + str(H_error) + " nats.")
    print("Effective sample size = " + str(ESS))

    # Resample to uniform weight
    N = int(ESS)
    posterior_sample = zeros((N, shape(sample)[1]))
    w = P_samples
    w = w/max(w)
    savetxt('weights.txt', w) # Save weights
    for i in range(0, N):
        while True:
            which = randint(sample.shape[0])
            if rand() <= w[which]:
                break
        posterior_sample[i,:] = sample[which,:]
    savetxt("posterior_sample.txt", posterior_sample)

    ioff()
    show()

