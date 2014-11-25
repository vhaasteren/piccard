#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
test_ptafuncs.py
"""

from __future__ import division

import numpy as np

from ptafuncs import *


def test_argsortTOAs():
    toas = np.array([1.1, 1.2, 1.3, 1.4, 15.4, 15.5, 15.5, 15.6, 15.7, \
            15.8, 15.9, 30.1, 30.2, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7])
    flags = ['be1']*19
    flags[2] = 'be2'
    flags[3] = 'be2'
    flags[8] = 'be2'
    flags[9] = 'be2'
    flags[11] = 'be2'
    flags[12] = 'be2'
    flags[13] = 'be2'

    isort, isort_inv = argsortTOAs(toas, np.array(flags))

    print(isort == np.array([ 0,  1,  2,  3,  8,  9,  4,  5,  6,  7, 10, \
            11, 12, 13, 14, 15, 16, 17, 18]))
    print(isort[isrot_inv] == np.arange(19))
