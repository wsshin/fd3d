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
#define __FUNCT__ "set_scale_Epec_at"
/**
 * set_scale_Epec_at
 * -------------
 * Set an element of the vector that scales E fields on PEC by a factor of 2.
 */
PetscErrorCode set_scale_Epec_at(PetscScalar *scale_Epec_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*scale_Epec_value = 1.0;
	if (gi->bc[axis][Neg]==PEC && ind[axis]==0) {
		*scale_Epec_value *= 2.0;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sparam_mu_at"
/**
 * set_sparam_mu_at
 * -------------
 * Set an element of the vector of s-parameter factors for mu.
 */
PetscErrorCode set_sparam_mu_at(PetscScalar *sparam_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	/** When w = x, y, z, mu_w is defined at Hw points.  Therefore mu_w is at the 
	  dual grid point in w axis, and at primary grid points in the other two axes. */
	*sparam_mu_value = gi->s_prim[axis1][ind1] * gi->s_prim[axis2][ind2] / gi->s_dual[axis0][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dparam_mu_at"
/**
 * set_dparam_mu_at
 * -------------
 * Set an element of the vector of d-parameter factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for mu.
 */
PetscErrorCode set_dparam_mu_at(PetscScalar *dparam_mu_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	PetscInt ind0 = ind[axis0];
	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	/** When w = x, y, z, mu_w is defined at Hw points.  Therefore mu_w is at the 
	  dual grid point in w axis, and at primary grid points in the other two axes. */
	*dparam_mu_value = gi->d_prim[axis1][ind1] * gi->d_prim[axis2][ind2] / gi->d_dual[axis0][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sparam_eps_at"
/**
 * set_sparam_eps_at
 * -------------
 * Set an element of the vector of s-parameter factors for eps.
 */
PetscErrorCode set_sparam_eps_at(PetscScalar *sparam_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	/** When w = x, y, z, eps_w is defined at Ew points.  Therefore eps_w is at the 
	  primary grid point in w axis, and at dual grid points in the other two axes. */
	*sparam_eps_value = gi->s_dual[axis1][ind[axis1]] * gi->s_dual[axis2][ind[axis2]] / gi->s_prim[axis0][ind[axis0]];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_dparam_eps_at"
/**
 * set_dparam_eps_at
 * -------------
 * Set an element of the vector of d-parameter factors (dy*dz/dx, dz*dx/dy, dx*dy/dz) for eps.
 */
PetscErrorCode set_dparam_eps_at(PetscScalar *dparam_eps_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	/** When w = x, y, z, eps_w is defined at Ew points.  Therefore eps_w is at the 
	  primary grid point in w axis, and at dual grid points in the other two axes. */
	*dparam_eps_value = gi->d_dual[axis1][ind[axis1]] * gi->d_dual[axis2][ind[axis2]] / gi->d_prim[axis0][ind[axis0]];

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
	PetscInt ind0 = ind[axis0];

	*dLe_value = gi->d_prim[axis0][ind0];

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
	PetscInt ind0 = ind[axis0];

	*dLh_value = gi->d_dual[axis0][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sparamLe_at"
/**
 * set_sparamLe_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sparamLe_at(PetscScalar *sparamLe_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	PetscInt ind0 = ind[axis0];

	*sparamLe_value = gi->s_prim[axis0][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sparamLh_at"
/**
 * set_sparamLh_at
 * -------------
 * Set an element of the vector whose elements are length stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sparamLh_at(PetscScalar *sparamLh_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis0 = axis;
	PetscInt ind0 = ind[axis0];

	*sparamLh_value = gi->s_dual[axis0][ind0];

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

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*dSe_value = gi->d_dual[axis1][ind1] * gi->d_dual[axis2][ind2];

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

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*dSh_value = gi->d_prim[axis1][ind1] * gi->d_prim[axis2][ind2];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sparamSe_at"
/**
 * set_sparamSe_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by E-field 
 * components.
 */
PetscErrorCode set_sparamSe_at(PetscScalar *sparamSe_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*sparamSe_value = gi->s_dual[axis1][ind1] * gi->s_dual[axis2][ind2];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_sparamSh_at"
/**
 * set_sparamSh_at
 * -------------
 * Set an element of the vector whose elements are the area stretch factors centered by H-field 
 * components.
 */
PetscErrorCode set_sparamSh_at(PetscScalar *sparamSh_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	*sparamSh_value = gi->s_prim[axis1][ind1] * gi->s_prim[axis2][ind2];

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
