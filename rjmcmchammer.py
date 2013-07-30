# encoding: utf-8
"""
A class, adjusted from ''emcee'', that does RJMCMC sampling

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
    user will have to deal with that. The

    This sampler expects a function that returns a new set of parameters, based
    on a transdimensional jump, and a probability that a transdimensional jump
    is carried out

    """
    def __init__(self, jprob, jumpparsfn, *args, **kwargs):
        super(RJMHSampler, self).__init__(*args, **kwargs)
        self.jprob = jprob
        self.jumpparsfn = jumpparsfn

    def reset(self):
        super(RJMHSampler, self).reset()
        self._nmod1 = np.empty(0, dtype=np.int)     # Red noise model
        self._nmod2 = np.empty(0, dtype=np.int)     # DM model

    def get_lnprob(self, p):
        pass

    def sample(self, p0, lnprob=None, randomstate=None, thin=1,
            storechain=True, iterations=1):
        """
        Advances the chain ``iterations`` steps as an iterator

        :param p0:
            The initial position vector.

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
            lnprob = self.get_lnprob(p)

        # Resize the chain in advance.
        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate((self._chain,
                    np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))

        i0 = self.iterations
        # Use range instead of xrange for python 3 compatability
        for i in range(int(iterations)):
            self.iterations += 1

            # Calculate the proposal distribution.
            q = self._random.multivariate_normal(p, self.cov)

	    # Calculate the new loglikelihood
            newlnprob = self.get_lnprob(q)
            diff = newlnprob - lnprob

            # M-H acceptance ratio
            if diff < 0:
                diff = np.exp(diff) - self._random.rand()

            if diff > 0:
                p = q
                lnprob = newlnprob
                self.naccepted += 1

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[ind, :] = p
                self._lnprob[ind] = lnprob

            # Heavy duty iterator action going on right here...
            yield p, lnprob, self.random_state


