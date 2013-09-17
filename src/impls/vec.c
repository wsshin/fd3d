#include <assert.h>
#include "vec.h"

#undef __FUNCT__
#define __FUNCT__ "createVecPETSc"
PetscErrorCode createVecPETSc(Vec *vec, const char *dataset_name, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	PetscViewer viewer;
	char fieldfile_name[PETSC_MAX_PATH_LEN];

	ierr = PetscStrcpy(fieldfile_name, gi.input_name); CHKERRQ(ierr);
	ierr = PetscStrcat(fieldfile_name, "."); CHKERRQ(ierr);
	ierr = PetscStrcat(fieldfile_name, dataset_name); CHKERRQ(ierr);
	//ierr = PetscStrcat(fieldfile_name, ".gz"); CHKERRQ(ierr);
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, fieldfile_name, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
	ierr = VecDuplicate(gi.vecTemp, vec); CHKERRQ(ierr);
	//ierr = VecCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
	ierr = VecLoad(*vec, viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createVecHDF5"
PetscErrorCode createVecHDF5(Vec *vec, const char *dataset_name, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	PetscViewer viewer;
	ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, gi.inputfile_name, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerHDF5PushGroup(viewer, "/"); CHKERRQ(ierr);  // assume that all datasets are under "/".

	ierr = VecDuplicate(gi.vecTemp, vec); CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject) *vec, ++dataset_name); CHKERRQ(ierr);  // ++ to remove '/'
	ierr = VecLoad(*vec, viewer); CHKERRQ(ierr);
	ierr = PetscViewerHDF5PopGroup(viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setFieldArray"
PetscErrorCode setFieldArray(Vec field, FunctionSetComponentAt setComponentAt, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Field ***field_array;  // 3D array that images Vec field
	ierr = DMDAVecGetArray(gi.da, field, &field_array); CHKERRQ(ierr);

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt ind[Naxis], axis;  // x, y, z indices of grid point
	for (ind[Zz] = oz; ind[Zz] < oz+nz; ++ind[Zz]) {
		for (ind[Yy] = oy; ind[Yy] < oy+ny; ++ind[Yy]) {
			for (ind[Xx] = ox; ind[Xx] < ox+nx; ++ind[Xx]) {
				for (axis = 0; axis < Naxis; ++axis) {
					/** field_array is just the array-representation of the vector field.  So, 
					setting values on field_array is actually setting values on field.*/
					/** setComponentAt() may modify gi internally, but the original gi is intact
					becaues gi has already been copied. */
					ierr = setComponentAt(&field_array[ind[Zz]][ind[Yy]][ind[Xx]].comp[axis], (Axis)axis, ind, &gi); CHKERRQ(ierr);
				}
			}
		}
	}

	ierr = DMDAVecRestoreArray(gi.da, field, &field_array); CHKERRQ(ierr);


	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createFieldArray"
PetscErrorCode createFieldArray(Vec *field, FunctionSetComponentAt setComponentAt, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = VecDuplicate(gi.vecTemp, field); CHKERRQ(ierr);
	ierr = setFieldArray(*field, setComponentAt, gi); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_mask_prim_at"
/**
 * set_mask_prim_at
 * -------------
 * Mask the primary fields at the negative boundaries according to their boundary conditions.
 */
PetscErrorCode set_mask_prim_at(PetscScalar *mask_prim_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*mask_prim_value = 1.0;
	Axis u = (Axis)((axis+1) % Naxis);
	if (ind[u]==0) {
		if ((gi->ge==Prim && gi->bc[u]==PEC) || (gi->ge==Dual && gi->bc[u]==PMC)) {
			*mask_prim_value = 0.0;
		}
	}

	u = (Axis)((axis+2) % Naxis);
	if (ind[u]==0) {
		if ((gi->ge==Prim && gi->bc[u]==PEC) || (gi->ge==Dual && gi->bc[u]==PMC)) {
			*mask_prim_value = 0.0;
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_mask_dual_at"
/**
 * set_mask_dual_at
 * -------------
 * Mask the dual fields at the negative boundaries according to their boundary conditions.
 */
PetscErrorCode set_mask_dual_at(PetscScalar *mask_dual_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*mask_dual_value = 1.0;
	if (ind[axis]==0) {
		if ((gi->ge==Prim && gi->bc[axis]==PEC) || (gi->ge==Dual && gi->bc[axis]==PMC)) {
			*mask_dual_value = 0.0;
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_double_Fbc_at"
/**
 * set_double_Fbc_at
 * -------------
 * Set an element of the vector that scales fields at boundaries by a factor of 2.
 */
PetscErrorCode set_double_Fbc_at(PetscScalar *double_Fbc_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*double_Fbc_value = 2.0;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	BC bc;

	if (gi->x_type==Etype && gi->ge==Prim) {
		bc = PMC;
	} else if (gi->x_type==Etype && gi->ge==Dual) {
		bc = PEC;
	} else if (gi->x_type==Htype && gi->ge==Dual) {
		bc = PEC;
	} else {
		assert(gi->x_type==Htype && gi->ge==Prim);
		bc = PMC;
	}

	if ((gi->x_type==Etype && gi->ge==Dual) || (gi->x_type==Htype && gi->ge==Prim)) {
		if (ind0==0 && gi->bc[axis0]==bc) {
			*double_Fbc_value *= 2.0;
		}
	} else {
		assert((gi->x_type==Etype && gi->ge==Prim) || (gi->x_type==Htype && gi->ge==Dual));
		if (gi->bc[axis1]==bc && ind1==0) {
			*double_Fbc_value *= 2.0;
		}

		if (gi->bc[axis2]==bc && ind2==0) {
			*double_Fbc_value *= 2.0;
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sfactor_mu_at"
/**
 * set_sfactor_mu_at
 * -------------
 * Set an element of the vector of s-factors for mu.
 */
PetscErrorCode set_sfactor_mu_at(PetscScalar *sfactor_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	/** When w = x, y, z, mu_w is defined at Hw points.  Therefore mu_w is at the 
	  dual grid point in w axis, and at primary grid points in the other two axes. */
	*sfactor_mu_value = gi->s_factor[axis1][gm][ind1] * gi->s_factor[axis2][gm][ind2] / gi->s_factor[axis0][ge][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dfactor_mu_at"
/**
 * set_dfactor_mu_at
 * -------------
 * Set an element of the vector of d-factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for mu.
 */
PetscErrorCode set_dfactor_mu_at(PetscScalar *dfactor_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	/** When w = x, y, z, mu_w is defined at Hw points.  Therefore mu_w is at the 
	  dual grid point in w axis, and at primary grid points in the other two axes. */
	*dfactor_mu_value = gi->dl[axis1][gm][ind1] * gi->dl[axis2][gm][ind2] / gi->dl[axis0][ge][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sfactor_eps_at"
/**
 * set_sfactor_eps_at
 * -------------
 * Set an element of the vector of s-factors for eps.
 */
PetscErrorCode set_sfactor_eps_at(PetscScalar *sfactor_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	/** When w = x, y, z, eps_w is defined at Ew points.  Therefore eps_w is at the 
	  primary grid point in w axis, and at dual grid points in the other two axes. */
	*sfactor_eps_value = gi->s_factor[axis1][ge][ind1] * gi->s_factor[axis2][ge][ind2] / gi->s_factor[axis0][gm][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dfactor_eps_at"
/**
 * set_dfactor_eps_at
 * -------------
 * Set an element of the vector of d-factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for eps.
 */
PetscErrorCode set_dfactor_eps_at(PetscScalar *dfactor_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	/** When w = x, y, z, eps_w is defined at Ew points.  Therefore eps_w is at the 
	  primary grid point in w axis, and at dual grid points in the other two axes. */
	*dfactor_eps_value = gi->dl[axis1][ge][ind1] * gi->dl[axis2][ge][ind2] / gi->dl[axis0][gm][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sqrt_at"
/**
 * set_sqrt_at
 * -------------
 * Replace an element of the vector with the sqrt of it.
 */
PetscErrorCode set_sqrt_at(PetscScalar *value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;
	*value = PetscSqrtScalar(*value);
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "sqrtVec"
/**
 * sqrtVec
 * -------------
 * For a given vector, replace every element of the vector with the sqrt of it.
 */
PetscErrorCode sqrtVec(Vec vec, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr; 

	ierr = setFieldArray(vec, set_sqrt_at, gi); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_maskInf2One_at"
/**
 * set_maskInf2One_at
 * -------------
 * Create a vector whose Inf elements are replaced by 1.0's
 */
PetscErrorCode set_maskInf2One_at(PetscScalar *value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;
	if (PetscIsInfOrNanScalar(*value)) {
		*value = 1.0;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "maskInf2One"
/**
 * maskInf2One
 * -----------------
 * For a given vector, replace every Inf element to 1.0. 
 * otherwise.
 */
PetscErrorCode maskInf2One(Vec vec, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr; 

	ierr = setFieldArray(vec, set_maskInf2One_at, gi); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_infMask_at"
/**
 * set_infMask_at
 * -------------
 * Create a vector that masks out infinitely large elements of a vector.
 */
PetscErrorCode set_infMask_at(PetscScalar *value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;
	if (PetscIsInfOrNanScalar(*value)) {
		*value = 0.0;
	} else {
		*value = 1.0;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "infMaskVec"
/**
 * infMaskVec
 * -----------------
 * For a given vector, replace every element of the vector with 0.0 if the element is Inf, and 1.0 
 * otherwise.
 */
PetscErrorCode infMaskVec(Vec vec, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr; 

	ierr = setFieldArray(vec, set_infMask_at, gi); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_complementMask_at"
/**
 * set_complementMask_at
 * ---------------------
 * Replace an element of the vector with 1.0 if the element is zero, and 0.0 otherwise.
 */
PetscErrorCode set_complementMask_at(PetscScalar *value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;
	*value = (*value==0.0 ? 1.0 : 0.0);
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "complementMaskVec"
/**
 * complementMaskVec
 * -----------------
 * For a given vector, replace every element of the vector with 1.0 if the element is zero, and 0.0 
 * otherwise.
 */
PetscErrorCode complementMaskVec(Vec vec, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr; 

	ierr = setFieldArray(vec, set_complementMask_at, gi); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dLe_at"
/**
 * set_dLe_at
 * -------------
 * Set an element of the vector whose elements are dL centered by E-field components.
 */
PetscErrorCode set_dLe_at(PetscScalar *dLe_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);
	PetscInt ind0 = ind[axis0];

	*dLe_value = gi->dl[axis0][gm][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dLh_at"
/**
 * set_dLh_at
 * -------------
 * Set an element of the vector whose elements are dL centered by H-field components.
 */
PetscErrorCode set_dLh_at(PetscScalar *dLh_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	GridType ge = gi->ge;
	PetscInt ind0 = ind[axis0];

	*dLh_value = gi->dl[axis0][ge][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sfactorLe_at"
/**
 * set_sfactorLe_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sfactorLe_at(PetscScalar *sfactorLe_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);
	PetscInt ind0 = ind[axis0];

	*sfactorLe_value = gi->s_factor[axis0][gm][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sfactorLh_at"
/**
 * set_sfactorLh_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sfactorLh_at(PetscScalar *sfactorLh_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	GridType ge = gi->ge;
	PetscInt ind0 = ind[axis0];

	*sfactorLh_value = gi->s_factor[axis0][ge][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dSe_at"
/**
 * set_dSe_at
 * -------------
 * Set an element of the vector whose elements are dS centered by E-field components.
 */
PetscErrorCode set_dSe_at(PetscScalar *dSe_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*dSe_value = gi->dl[axis1][ge][ind1] * gi->dl[axis2][ge][ind2];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dSh_at"
/**
 * set_dSh_at
 * -------------
 * Set an element of the vector whose elements are dS centered by H-field components.
 */
PetscErrorCode set_dSh_at(PetscScalar *dSh_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*dSh_value = gi->dl[axis1][gm][ind1] * gi->dl[axis2][gm][ind2];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sfactorSe_at"
/**
 * set_sfactorSe_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sfactorSe_at(PetscScalar *sfactorSe_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*sfactorSe_value = gi->s_factor[axis1][ge][ind1] * gi->s_factor[axis2][ge][ind2];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sfactorSh_at"
/**
 * set_sfactorSh_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sfactorSh_at(PetscScalar *sfactorSh_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	GridType ge = gi->ge;
	GridType gm = (GridType)((ge+1) % Ngt);

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*sfactorSh_value = gi->s_factor[axis1][gm][ind1] * gi->s_factor[axis2][gm][ind2];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_index_at"
/**
 * set_index_at
 * -------------
 */
PetscErrorCode set_index_at(PetscScalar *index_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*index_value = axis + Naxis*ind[Xx] + Naxis*gi->N[Xx]*ind[Yy] + Naxis*gi->N[Xx]*gi->N[Yy]*ind[Zz];

	PetscFunctionReturn(0);
}
