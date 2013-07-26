#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

"""
Usage:
python createh5file.py directory hdf5file

directory: a directory that contains a bunch of par/tim files, with matching par
and tim files (no included files, unless with proper path, since tempo2 will not
know where to find them).
hdf5file: the filename of a non-existing hdf5file.

TODO: This file should probably check that the hdf5file does not exist, or
truncate it first
"""

import piccard as pic
import os
import glob
from subprocess import call
import sys as sys
from optparse import OptionParser

 
################ Functions ################

def run(command, disp):
    commandline = command % globals()
    if disp:
        print "--> %s" % commandline
    
    try:
        assert(os.system(commandline) == 0)
    except:
        print 'Script %s failed at command "%s".' % (sys.argv[0],commandline)
        sys.exit(1)




################ Main program ################

parser = OptionParser(usage="usage: %prog [options] directory hdf5file \n \
                      - directory : a directory that contains a bunch of par/tim files, with matching par and tim files (no included files, unless with proper path, since tempo2 will not know where to find them).\n \
                      - hdf5file: the filename of a non-existing hdf5file.",
                      version="18.01.13, version 1.0, EPTA-DA group ")   

parser.add_option("-d", "--dataOnly",
                  action="store_true", dest="dataOnly", default=False,
                  help="Just include the data in h5 file [default false]") 



(options, args) = parser.parse_args()

if len(args) < 2:
    parser.error("I need to know the the directory and the h5file (createh5only -h )") 


path = args[0]
h5file = args[1]

print "Directory :", path
print "h5 file :", h5file

##### Test if hdf5file exit
FileExist = os.path.isfile(h5file)
while FileExist :
    choice = raw_input("The file "+h5file+" exist. Do you want to overwrite it [y/n] ? ")
    if choice=="no" or choice=="n" :
        h5file  = raw_input("Enter a new filename ? ")
        FileExist  = os.path.isfile(h5file)
    else :
        run("rm "+h5file+"\n",True)
        FileExist = False



##### Initialise the t2df (tempo2 datafile) object, giving it the hdf5 filename
t2df = pic.DataFile(h5file)

##### Add all pulsars to the hdf5 file
for infile in glob.glob(os.path.join(path, '*.par') ):
  filename = os.path.splitext(infile)
  t2df.addpulsar(filename[0]+'.par', filename[0]+'.tim')

