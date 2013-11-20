#!/usr/bin/env python

# using gfortran from http://gcc.gnu.org/wiki/GFortranBinariesMacOS on Mac OSX 10.4

configure_options = [
  '--with-fc=0',
  '--with-debugging=0',
  '--with-scalar-type=complex',
  '--download-mpich',
  '--download-hdf5',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
