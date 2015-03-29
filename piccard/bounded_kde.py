import numpy as np
from scipy.special import erf
from scipy.stats import gaussian_kde

class Bounded_kde(gaussian_kde):
    r"""Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, low=None, high=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param low: The lower domain boundary.

        :param high: The upper domain boundary."""
        pts = np.atleast_1d(pts)

        assert pts.ndim == 1, 'Bounded_kde can only be one-dimensional'
        
        super(Bounded_kde, self).__init__(pts, *args, **kwargs)

        self._low = low
        self._high = high

    @property
    def low(self):
        """The lower bound of the domain."""
        return self._low

    @property
    def high(self):
        """The upper bound of the domain."""
        return self._high

    def evaluate(self, xs):
        """Return an estimate of the density evaluated at the given
        points."""
        xs = np.atleast_1d(xs)
        assert xs.ndim == 1, 'points must be one-dimensional'

        pdf = super(Bounded_kde, self).evaluate(xs)

        if self.low is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.low - xs)

        if self.high is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.high - xs)

        return pdf

    __call__ = evaluate
