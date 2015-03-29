from __future__ import print_function

# Make sure we are not in the piccard source directory
#try:
#    from piccard import jitterext
#except ImportError:
#    msg = """Error importing piccard: you cannot import piccard while being in
#    piccard source directory; please exit the piccard source tree first, and
#    relaunch your python intepreter."""
#    raise ImportError(msg)

from piccard import *
from ptafuncs import *
from piccard_samplers import *
from piccard_pso import *
from piccard_freqstat import *
from piccard_gibbs import *
from resampler import *
from distsampling import *
from resamplelib import *

__version__ = 2014.12

def test():
    # Run some tests here
    print("{0} tests have passed".format(0))
