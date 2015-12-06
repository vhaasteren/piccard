#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division, print_function

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
from functools import partial

from transformations import *
from stingrays import *

def hmcLikelihood(h5filename=None, jsonfilename=None, **kwargs):
    """Wrapper for the compound of the stingray transformation and the interval
    transformation
    """
    if 'wrapperclass' in kwargs:
        raise ValueError("hmcLikelihood already pre-sets wrapperclass")

    return intervalLikelihood(h5filename=h5filename,
            jsonfilename=jsonfilename,
            wrapperclass=hpStingrayLikelihood,
            **kwargs)

def tmHmcLikelihood1(h5filename=None, jsonfilename=None, **kwargs):
    """Wrapper for the compound of the stingray transformation and the interval
    transformation
    """
    if 'wrapperclass' in kwargs:
        raise ValueError("hmcLikelihood already pre-sets wrapperclass")

    return intervalLikelihood(h5filename=h5filename,
            jsonfilename=jsonfilename,
            wrapperclass=tmStingrayLikelihood,
            **kwargs)

def tmHmcLikelihood2(h5filename=None, jsonfilename=None, **kwargs):
    """Wrapper for the compound of the stingray transformation and the interval
    transformation
    """
    if 'wrapperclass' in kwargs:
        raise ValueError("hmcLikelihood already pre-sets wrapperclass")

    return intervalLikelihood(h5filename=h5filename,
            jsonfilename=jsonfilename,
            wrapperclass=tmStingrayLikelihood2,
            **kwargs)

def muHmcLikelihood(h5filename=None, jsonfilename=None, **kwargs):
    """Wrapper for the compound of the stingray transformation and the interval
    transformation
    """
    if 'wrapperclass' in kwargs:
        raise ValueError("hmcLikelihood already pre-sets wrapperclass")

    return intervalLikelihood(h5filename=h5filename,
            jsonfilename=jsonfilename,
            wrapperclass=muStingrayLikelihood,
            **kwargs)
    
