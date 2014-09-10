# piccard #

This is Rutger van Haasteren's Bayesian-inference pipeline for Pulsar Timing
Array (PTA) data. The code interacts with [Tempo2](http://tempo2.sourceforge.net) through [libstempo](https://github.com/vallis/mc3pta/tree/master/stempo)

The code is use mainly for single-pulsar analysis, and gravitational-wave
detection purposes of full Pulsar Timing Array datasets. The modelling of the
data can include:
* Error bars and EFACs: multiple per pulsar
* White noise: EQUAD/jitter noise
* Red noise: per frequency, or modelled spectrum
* DM variations: per frequency, or modelled spectrum
* Correlated signals: per frequency, or modelled spectrum, with correlations:
  * Uniform: clock-error
  * Dipolar: ephemeris errors
  * Quadrupolar: isotropic stochastic background GR correlations
  * Anisotropic: GR correlated, expanded in spherical harmonics
* Timing models can be numerically included, either by:
  * Using the design matrix (linear timing model)
  * Calling libstempo for the full non-linear timing model

Many types of samplers are included:
* Metropolis-Hastings
* DNest
* twalk
* emcee ([2012](http://adsabs.harvard.edu/abs/2012arXiv1202.3665F)).
* pyMultiNest ([2013](https://github.com/JohannesBuchner/PyMultiNest))
* Reversible-Jump MCMC, based on emcee's Metropolis

For common-mode mitigation, the signals can be reconstructed mitigating
arbitrary signals simultaneously. Especially handy for getting rid of DM
variations. Expanded on the techniques of Lee et al. (in prep.), and similar to
what is included in TempoNest
([2013](https://github.com/LindleyLentati/TempoNest)).

A alpha-phase interface for the generation of mock data is provided. Needs to be
expanded still.


## Manual ##
Under construction. Last update 2014-09-10. Check.


## Requirements ##

* Python 2.7
* [numpy](http://numpy.scipy.org)
* [scipy](http://numpy.scipy.org)
* [matplotlib](http://matplotlib.org), for plotting only
* [tempo2](http://tempo2.sourceforge.net)
* [libstempo](https://github.com/vallis/mc3pta/tree/master/stempo)
* [emcee](http://dan.iel.fm/emcee) (optional)
* [pyMultiNest](https://github.com/JohannesBuchner/PyMultiNest) (optional)


## Contact ##

* [_Rutger van Haasteren_](mailto:vhaasteren@gmail.com)

