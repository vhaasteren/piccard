#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
piccard_constants.py

"""


# Some constants used in Piccard
# For DM calculations, use this constant
# See You et al. (2007) - http://arxiv.org/abs/astro-ph/0702366
# Lee et al. (in prep.) - ...
# Units here are such that delay = DMk * DM * freq^-2 with freq in MHz
pic_DMk = 4.15e3        # Units MHz^2 cm^3 pc sec

pic_spd = 86400.0       # Seconds per day
#pic_spy = 31556926.0   # Wrong definition of YEAR!!!
pic_spy =  31557600.0   # Seconds per year (yr = 365.25 days, so Julian years)
pic_T0 = 53000.0        # MJD to which all HDF5 toas are referenced
