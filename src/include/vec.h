#ifndef GUARD_vec_h
#define GUARD_vec_h

#include "gridinfo.h"
#include "type.h"

#include "petsc.h"

typedef struct {
	PetscScalar comp[Naxis];  // x,y,z component of field
} Field;  // used for DAVecGetArray()

typedef PetscErrorCode (*FunctionSetComponentAt)(PetscScalar *component, Axis axis, const PetscInt ind[], GridInfo *gi);

PetscErrorCode createVecHDF5(Vec *vec, const char *dataset_name, GridInfo gi);

PetscErrorCode createFieldArray(Vec *field, FunctionSetComponentAt setComponentAt, GridInfo gi);

/**
 * set_scale_Epmc_at
 * -------------
 * Set an element of the vector that scales E fields on PMC by a factor of 2.
 */
PetscErrorCode set_scale_Epmc_at(PetscScalar *scale_Epmc_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sparam_mu_at
 * -------------
 * Set an element of the vector of s-parameter factors for mu.
 */
PetscErrorCode set_sparam_mu_at(PetscScalar *sparam_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_dparam_mu_at
 * -------------
 * Set an element of the vector of d-parameter factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for mu.
 */
PetscErrorCode set_dparam_mu_at(PetscScalar *dparam_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sparam_eps_at
 * -------------
 * Set an element of the vector of s-parameter factors for eps.
 */
PetscErrorCode set_sparam_eps_at(PetscScalar *sparam_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_dparam_eps_at
 * -------------
 * Set an element of the vector of d-parameter factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for eps.
 */
PetscErrorCode set_dparam_eps_at(PetscScalar *dparam_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * sqrtVec
 * -------------
 * For a given vector, replace every element of the vector with the sqrt of it.
 */
PetscErrorCode sqrtVec(Vec vec, GridInfo gi);

/**
 * infMaskVec
 * -----------------
 * For a given vector, replace every element of the vector with 0.0 if the element is Inf, and 1.0 
 * otherwise.
 */
PetscErrorCode infMaskVec(Vec vec, GridInfo gi);
	 
/**
 * complementMaskVec
 * -----------------
 * For a given vector, replace every element of the vector with 1.0 if the element is zero, and 0.0 
 * otherwise.
 */
PetscErrorCode complementMaskVec(Vec vec, GridInfo gi);

/**
 * set_dLe_at
 * -------------
 * Set an element of the vector whose elements are dL centered by E-field components.
 */
PetscErrorCode set_dLe_at(PetscScalar *dLe_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_dLh_at
 * -------------
 * Set an element of the vector whose elements are dL centered by H-field components.
 */
PetscErrorCode set_dLh_at(PetscScalar *dLh_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sparamLe_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sparamLe_at(PetscScalar *sparamLe_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sparamLh_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sparamLh_at(PetscScalar *sparamLh_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_dSe_at
 * -------------
 * Set an element of the vector whose elements are dS centered by E-field components.
 */
PetscErrorCode set_dSe_at(PetscScalar *dSe_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_dSh_at
 * -------------
 * Set an element of the vector whose elements are dS centered by H-field components.
 */
PetscErrorCode set_dSh_at(PetscScalar *dSh_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sparamSe_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sparamSe_at(PetscScalar *sparamSe_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sparamSh_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sparamSh_at(PetscScalar *sparamSh_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_index_at
 * -------------
 */
PetscErrorCode set_index_at(PetscScalar *index_value, Axis axis, const PetscInt ind[], GridInfo *gi);

#endif
