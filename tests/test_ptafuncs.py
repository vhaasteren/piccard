#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
test_ptafuncs.py
"""

from __future__ import division

import numpy as np

from piccard import *


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

    isort, isort_inv = argsortTOAs(toas, np.array(flags), which='jitterext')

    assert np.all(isort == np.array([ 0,  1,  2,  3,  8,  9,  4,  5,  6,  7, 10, \
            11, 12, 13, 14, 15, 16, 17, 18]))
    assert np.all(isort[isort_inv] == np.arange(19))

    # More complicated test
    Nepochs = 100
    Nblocksize = 32
    Ntoas = Nepochs*Nblocksize
    Umat = np.zeros((Ntoas, Nepochs))
    for cc in range(Nepochs):
        Umat[cc*Nblocksize:(cc+1)*Nblocksize, cc] = 1.0

    # Create the toas, all lined up within epoch blocks, and unsorted per epoch
    avetoas = np.linspace(0, 10000, Nepochs)
    toas = np.dot(Umat, avetoas) + np.random.randn(Ntoas) * 0.1

    # Create the flags for all the toas
    uflags = np.array(['flag1', 'flag2', 'flag3', 'flag4'])
    flagids = np.random.randint(0, len(uflags), Ntoas)
    flags = uflags[flagids]

    # Calculate the sorting
    isort, iisort = argsortTOAs(toas, flags, which='jitterext', dt=10.0)

    # Check the sorting
    assert checkTOAsort(toas[isort], flags[isort], which='time', dt=10.0) == False
    assert checkTOAsort(toas[isort], flags[isort], which='jitterext', dt=10.0) == True

    toas = toas[isort]
    flags = flags[isort]

    # Check the sorting here with a piece of code independent from the function
    # checkTOAsort
    for flag in uflags:
        flagmask = (flags == flag)
        for cc, col in enumerate(Umat.T):
            colmask = col.astype(np.bool)
            order = flagmask[colmask]

            epinds = np.flatnonzero(order)
            if len(epinds > 0):
                assert len(epinds) == epinds[-1] - epinds[0] + 1

    # Calculate the sorting
    isort, iisort = argsortTOAs(toas, flags, which='time', dt=10.0)

    # Check the sorting
    assert checkTOAsort(toas[isort], flags[isort], which='time', dt=10.0) == True
    assert checkTOAsort(toas[isort], flags[isort], which='jitterext', dt=10.0) != True

    toas = toas[isort]
    flags = flags[isort]



if __name__=="__main__":
    test_argsortTOAs()
