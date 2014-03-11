#include "h5.h"

#undef __FUNCT__
#define __FUNCT__ "h5get_data"
/**
 * h5get_data
 * -----------
 * Retrieve an array data stored under a given data set name in an HDF5 file.
 */
PetscErrorCode h5get_data(hid_t inputfile_id, const char *dataset_name, hid_t mem_type_id, void *buf)
{
	PetscFunctionBegin;

	hid_t dataset_id;
	herr_t status;

	dataset_id = H5Dopen(inputfile_id, dataset_name, H5P_DEFAULT);
	status = H5Dread(dataset_id, mem_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
	status = H5Dclose(dataset_id);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "init_c"
/**
 * init_c
 * -----------
 * Construct a complex array initialized with ones.
 */
PetscErrorCode init_c(void *pc, const int numelem)
{
	PetscFunctionBegin;

	PetscScalar *pC = (PetscScalar *) pc;

	int i;
	for (i = 0; i < numelem; ++i) {
		*pC++ = 1.0;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ri2c"
/**
 * ri2c
 * -----------
 * Construct a complex array from an array with alternating real and imaginary values as elements.
 */
PetscErrorCode ri2c(const void *pri, void *pc, const int numelem)
{
	PetscFunctionBegin;

	PetscReal *pRI = (PetscReal *) pri;
	PetscScalar *pC = (PetscScalar *) pc;

	int i;
	for (i = 0; i < numelem; ++i) {
		*pC = *pRI++;
		*pC++ += PETSC_i * (*pRI++);
	}

	PetscFunctionReturn(0);
}

