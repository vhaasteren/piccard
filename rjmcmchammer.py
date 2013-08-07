# encoding: utf-8
"""
A class, adjusted from ''emcee'', that does RJMCMC sampling

Created by vhaasteren on 2013-08-06.
Copyright (c) 2013 Rutger van Haasteren

"""

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy as np
import multiprocessing

try:    # Fall back to internal emcee implementation
    import emcee as emceehammer
    emcee = emceehammer
except ImportError:
    import mcmchammer as mcmchammer
    emcee = mcmchammer

try:
    import acor
    acor = acor
except ImportError:
    acor = None




# === RJMHSampler for Fourier components (DM & red noise) ===
class RJMHSampler(emcee.MHSampler):
    """
    Derived from the basic Metropolis-Hastings sampler emcee.MHSampler

    This sampler expects the different models to be numbered
    nmod = 0, 1, 2, 3, ...
    Only works on a single pulsar, so the model number is identical to the
    number of frequencies

    The model parameters are assumed to stay the same. In the likelihood some
    parameters will therefore just be ignored (always receives the maximum). The
    user will have to deal with that when reading the chain

    This sampler expects a function that returns a new set of parameters, based
    on a transdimensional jump, and a probability that a transdimensional jump
    is carried out

    """
    def __init__(self, jprob, jumpsize1, jumpsize2, jumpparsfn, propdjumpfn, acceptjumpfn, *args, **kwargs):
        super(RJMHSampler, self).__init__(*args, **kwargs)
        self.jprob = jprob                  # Jump probability
        self.jumpsize1 = jumpsize1          # Size of jump in RN mode
        self.jumpsize2 = jumpsize2          # Size of jump in DM mode
        self.jumpparsfn = jumpparsfn        # After jump, draw parameters function
        self.propdjumpfn = propdjumpfn      # Propose a new trans-dim jump
        self.acceptjumpfn = acceptjumpfn    # Signal we accepted a t-d jump

    def reset(self):
        super(RJMHSampler, self).reset()
        self._nmod1 = np.empty(0, dtype=np.int)     # Red noise model
        self._nmod2 = np.empty(0, dtype=np.int)     # DM model

    def get_lnprob(self, p, nmod1, nmod2):
        """Return the log-probability at the given position and model """
        return self.lnprobfn(p, psrnfinc=[nmod1], psrnfdminc=[nmod2])

    def get_acceptjumpfn(self, lnprob, nmod1, nmod2):
        """Accept a jump, and retrieve the real logposterior"""
        return self.acceptjumpfn(lnprob, psrnfinc=[nmod1], psrnfdminc=[nmod2])

    def sample(self, p0, mod1, mod2, lnprob=None, randomstate=None, thin=1,
            storechain=True, iterations=1):
        """
        Advances the chain ``iterations`` steps as an iterator

        :param p0:
            The initial position vector.

        :param mod1:
            The initial number of frequencies (RN)

        :param mod2:
            The initial number of frequencies (DM)

        :param lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.

        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        :param iterations: (optional)
            The number of steps to run.

        :param thin: (optional)
            If you only want to store and yield every ``thin`` samples in the
            chain, set thin to an integer greater than 1.

        :param storechain: (optional)
            By default, the sampler stores (in memory) the positions and
            log-probabilities of the samples in the chain. If you are
            using another method to store the samples to a file or if you
            don't need to analyse the samples after the fact (for burn-in
            for example) set ``storechain`` to ``False``.

        At each iteration, this generator yields:

        * ``pos`` — The current positions of the chain in the parameter
          space.

        * ``lnprob`` — The value of the log posterior at ``pos`` .

        * ``rstate`` — The current state of the random number generator.

        """

        self.random_state = randomstate

        p = np.array(p0)
        if lnprob is None:
            lnprob = self.get_lnprob(p, mod1, mod2)

        # Resize the chain in advance.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                    np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))
            self._nmod1 = np.append(self._nmod1, np.zeros(N, dtype=np.int))
            self._nmod2 = np.append(self._nmod2, np.zeros(N, dtype=np.int))

        i0 = self.iterations
        # Use range instead of xrange for python 3 compatability
        for i in range(int(iterations)):
            self.iterations += 1

            # Calculate the proposal distribution.
            q = self._random.multivariate_normal(p, self.cov)
            qmod1 = mod1
            qmod2 = mod2

            # J1643-hack. 1% of the time, do a 39-3 or 3-39 jump
            # if self._random.rand() < 0.01:
            #    if qmod1 == 5:
            #        qmod1 = 39
            #        q = self.jumpparsfn(q, qmod1, qmod2)
            #    elif qmod1 == 39:
            #        qmod1 = 5
            #        q = self.jumpparsfn(q, qmod1, qmod2)


            # Decide whether we will do a trans-dimensional jump
            if self._random.rand() < self.jprob:
                # Trans-dim jump. Adjust the models and parameters
                qmod1, qmod2 = self.propdjumpfn(self.jumpsize1, self.jumpsize2)
                q = self.jumpparsfn(q, qmod1, qmod2)

	    # Calculate the new loglikelihood
            newlnprob = self.get_lnprob(q, qmod1, qmod2)
            diff = newlnprob - lnprob

            # M-H acceptance ratio
            if diff < 0:
                diff = np.exp(diff) - self._random.rand()

            if diff > 0:
                if mod1 != qmod1 or mod2 != qmod2:
                    # Now retrieve the *real* lnprob here
                    newlnprob = self.get_acceptjumpfn(newlnprob, qmod1, qmod2)
                p = q
                mod1 = qmod1
                mod2 = qmod2
                lnprob = newlnprob
                self.naccepted += 1

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[ind, :] = p
                self._lnprob[ind] = lnprob
                self._nmod1[ind] = mod1
                self._nmod2[ind] = mod2

            # Heavy duty iterator action going on right here...
            yield p, mod1, mod2, lnprob, self.random_state


    def run_mcmc(self, pos0, mod01, mod02, N, rstate0=None, lnprob0=None, **kwargs):
        """
        Iterate :func:`sample` for ``N`` iterations and return the result.

        :param p0:
            The initial position vector.

        :param mod01:
            The initial frequency model

        :param mod02:
            The initial DM model

        :param N:
            The number of steps to run.

        :param lnprob0: (optional)
            The log posterior probability at position ``p0``. If ``lnprob``
            is not provided, the initial value is calculated.

        :param rstate0: (optional)
            The state of the random number generator. See the
            :func:`random_state` property for details.

        :param **kwargs: (optional)
            Other parameters that are directly passed to :func:`sample`.

        """
        for results in self.sample(pos0, mod01, mod02, lnprob0, rstate0, iterations=N,
                **kwargs):
            pass
        return results

