#ifndef GUARD_vec_h
#define GUARD_vec_h

#include "gridinfo.h"
#include "type.h"

#include "petsc.h"
#include "petscviewerhdf5.h"   


typedef struct {
	PetscScalar comp[Naxis];  // x,y,z component of field
} Field;  // used for DAVecGetArray()

typedef PetscErrorCode (*FunctionSetComponentAt)(PetscScalar *component, Axis axis, const PetscInt ind[], GridInfo *gi);

PetscErrorCode createVecHDF5(Vec *vec, const char *dataset_name, GridInfo gi);

PetscErrorCode createVecPETSc(Vec *vec, const char *dataset_name, GridInfo gi);

PetscErrorCode createFieldArray(Vec *field, FunctionSetComponentAt setComponentAt, GridInfo gi);

/**
 * set_mask_prim_at
 * -------------
 * Mask the primary fields at the negative boundaries according to their boundary conditions.
 */
PetscErrorCode set_mask_prim_at(PetscScalar *mask_prim_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_mask_dual_at
 * -------------
 * Mask the dual fields at the negative boundaries according to their boundary conditions.
 */
PetscErrorCode set_mask_dual_at(PetscScalar *mask_dual_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_double_Fbc_at
 * -------------
 * Set an element of the vector that scales fields at boundaries by a factor of 2.
 */
PetscErrorCode set_double_Fbc_at(PetscScalar *double_Fbc_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sfactor_mu_at
 * -------------
 * Set an element of the vector of s-factors for mu.
 */
PetscErrorCode set_sfactor_mu_at(PetscScalar *sfactor_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_dfactor_mu_at
 * -------------
 * Set an element of the vector of d-factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for mu.
 */
PetscErrorCode set_dfactor_mu_at(PetscScalar *dfactor_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sfactor_eps_at
 * -------------
 * Set an element of the vector of s-factors for eps.
 */
PetscErrorCode set_sfactor_eps_at(PetscScalar *sfactor_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_dfactor_eps_at
 * -------------
 * Set an element of the vector of d-factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for eps.
 */
PetscErrorCode set_dfactor_eps_at(PetscScalar *dfactor_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * sqrtVec
 * -------------
 * For a given vector, replace every element of the vector with the sqrt of it.
 */
PetscErrorCode sqrtVec(Vec vec, GridInfo gi);

/**
 * maskInf2One
 * -----------------
 * For a given vector, replace every Inf element to 1.0. 
 * otherwise.
 */
PetscErrorCode maskInf2One(Vec vec, GridInfo gi);

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
 * set_sfactorLe_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sfactorLe_at(PetscScalar *sfactorLe_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sfactorLh_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sfactorLh_at(PetscScalar *sfactorLh_value, Axis axis, const PetscInt ind[], GridInfo *gi);

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
 * set_sfactorSe_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sfactorSe_at(PetscScalar *sfactorSe_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_sfactorSh_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sfactorSh_at(PetscScalar *sfactorSh_value, Axis axis, const PetscInt ind[], GridInfo *gi);

/**
 * set_index_at
 * -------------
 */
PetscErrorCode set_index_at(PetscScalar *index_value, Axis axis, const PetscInt ind[], GridInfo *gi);

#endif
