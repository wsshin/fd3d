#ifndef GUARD_h5_h
#define GUARD_h5_h

#include "petsc.h"
#include "hdf5.h"
#include "type.h"

/**
 * h5get_data
 * -----------
 * Retrieve an array data stored under a given data set name in an HDF5 file.
 */
PetscErrorCode h5get_data(hid_t file_id, const char *dataset_name, hid_t mem_type_id, void *buf);

/**
 * ri2c
 * -----------
 * Construct a complex array from an array with alternating real and imaginary values as elements.
 */
PetscErrorCode ri2c(const void *pri, void *pc, const int numelem);

#endif
