#include "vec.h"

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

	gi.is_pFunc_set = PETSC_FALSE;  // allows setComponentAt() sets gi.pFunc at its first call
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

	if (gi.is_pFunc_set) {
		Py_DECREF(gi.pFunc);
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
#define __FUNCT__ "set_scale_Epmc_at"
/**
 * set_scale_Epmc_at
 * -------------
 * Set an element of the vector that scales E fields on PMC by a factor of 2.
 */
PetscErrorCode set_scale_Epmc_at(PetscScalar *scale_Epmc_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	*scale_Epmc_value = 1.0;

	Axis axis1 = (Axis)((axis+1) % Naxis);
	Axis axis2 = (Axis)((axis+2) % Naxis);

	PetscInt ind1 = ind[axis1];
	PetscInt ind2 = ind[axis2];

	if (gi->bc[axis1][Neg]==PMC && ind1==0) {
		*scale_Epmc_value *= 2.0;
	}

	if (gi->bc[axis2][Neg]==PMC && ind2==0) {
		*scale_Epmc_value *= 2.0;
	}

	/** Note that the above makes E fields on the common edges between two PMC planes
	  scaled by a factor of 2.0*2.0 = 4.0. */

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_mu_at"
/**
 * set_mu_at
 * -------------
 * Set an element of the vector of mu (permeability) compatible with matrices. 
 */
PetscErrorCode set_mu_at(PetscScalar *mu_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	if (!gi->is_pFunc_set) {
		gi->pFunc = PyObject_GetAttrString(gi->pSim, "get_mu_at");
		gi->is_pFunc_set = PETSC_TRUE;
	}

	PyObject *pValue = PyObject_CallFunction(gi->pFunc, (char *) "iiiii", axis, ind[Xx], ind[Yy], ind[Zz], gi->bg_only);  // "iiii" means 4 integers
	Py_complex py_mu = PyComplex_AsCComplex(pValue);
	Py_DECREF(pValue);

	*mu_value = py_mu.real + PETSC_i * py_mu.imag;

	//PetscFPrintf(PETSC_COMM_WORLD, stdout, "mu: %f + (%f) i\n", PetscRealPart(*mu_value), PetscImaginaryPart(*mu_value));

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
	  primary grid point in w axis, and at dual grid points in the other two axes. */
	*sparam_mu_value = gi->s_dual[axis1][ind1] * gi->s_dual[axis2][ind2] / gi->s_prim[axis0][ind0];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_eps_at"
/**
 * set_eps_at
 * -------------
 * Set an element of the vector of the eps (permittivity) compatible with matrices. 
 */
PetscErrorCode set_eps_at(PetscScalar *eps_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	if (!gi->is_pFunc_set) {
		gi->pFunc = PyObject_GetAttrString(gi->pSim, "get_eps_at");
		gi->is_pFunc_set = PETSC_TRUE;
	}

	PyObject *pValue = PyObject_CallFunction(gi->pFunc, (char *) "iiiii", (PetscInt) axis, ind[Xx], ind[Yy], ind[Zz], gi->bg_only);  // "iiii" means 4 integers
	Py_complex py_eps = PyComplex_AsCComplex(pValue);
	Py_DECREF(pValue);

	if (isinf(py_eps.real)) {
		*eps_value = 1.0;  // ininitely large eps is masked out by epsMask in create_A_and_b()
	} else {
		*eps_value = py_eps.real + PETSC_i * py_eps.imag;
	}

	//PetscFPrintf(PETSC_COMM_WORLD, stdout, "eps: %f + (%f) i\n", PetscRealPart(*eps_value), PetscImaginaryPart(*eps_value));

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_epsNode_at"
/**
 * set_epsNode_at
 * -------------
 * Set an element of the vector of the eps (permittivity) at nodes compatible with matrices. 
 */
PetscErrorCode set_epsNode_at(PetscScalar *eps_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	if (!gi->is_pFunc_set) {
		gi->pFunc = PyObject_GetAttrString(gi->pSim, "get_eps_node_at");
		gi->is_pFunc_set = PETSC_TRUE;
	}

	PyObject *pValue = PyObject_CallFunction(gi->pFunc, (char *) "iiii", ind[Xx], ind[Yy], ind[Zz], gi->bg_only);  // "iiii" means 4 integers
	Py_complex py_eps = PyComplex_AsCComplex(pValue);
	Py_DECREF(pValue);

	if (isinf(py_eps.real)) {
		*eps_value = 1.0;  // ininitely large eps is masked out by epsMask in create_A_and_b()
	} else {
		*eps_value = py_eps.real + PETSC_i * py_eps.imag;
	}

	//PetscFPrintf(PETSC_COMM_WORLD, stdout, "eps: %f + (%f) i\n", PetscRealPart(*eps_value), PetscImaginaryPart(*eps_value));

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_epsMask_at"
/**
 * set_epsMask_at
 * -------------
 * Set an element of the mask vector of the eps (permittivity) compatible with matrices. 
 * The mask vector masks out infinitely large eps components.
 */
PetscErrorCode set_epsMask_at(PetscScalar *epsMask_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	if (!gi->is_pFunc_set) {
		gi->pFunc = PyObject_GetAttrString(gi->pSim, "get_eps_at");
		gi->is_pFunc_set = PETSC_TRUE;
	}

	PyObject *pValue = PyObject_CallFunction(gi->pFunc, (char *) "iiiii", (PetscInt) axis, ind[Xx], ind[Yy], ind[Zz], gi->bg_only);  // "iiii" means 4 integers
	Py_complex py_eps = PyComplex_AsCComplex(pValue);
	Py_DECREF(pValue);

	if (isinf(py_eps.real)) {
		*epsMask_value = 0.0;
	} else {
		*epsMask_value = 1.0;
	}

	//PetscFPrintf(PETSC_COMM_WORLD, stdout, "eps: %f + (%f) i\n", PetscRealPart(*eps_value), PetscImaginaryPart(*eps_value));

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
	  dual grid point in w axis, and at primary grid points in the other two axes. */
	*sparam_eps_value = gi->s_prim[axis1][ind[axis1]] * gi->s_prim[axis2][ind[axis2]] / gi->s_dual[axis0][ind[axis0]];

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

	*sparamLe_value = gi->s_dual[axis0][ind0];

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

	*sparamLh_value = gi->s_prim[axis0][ind0];

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

	*sparamSe_value = gi->s_prim[axis1][ind1] * gi->s_prim[axis2][ind2];

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

	*sparamSh_value = gi->s_dual[axis1][ind1] * gi->s_dual[axis2][ind2];

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_src_at"
/**
 * set_src_at 
 * -------------
 * Set an element of the source vector J_src.
 */
PetscErrorCode set_src_at(PetscScalar *src_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	if (!gi->is_pFunc_set) {
		gi->pFunc = PyObject_GetAttrString(gi->pSim, "get_src_at");
		gi->is_pFunc_set = PETSC_TRUE;
	}

	PyObject *pValue = PyObject_CallFunction(gi->pFunc, (char *) "iiii", axis, ind[Xx], ind[Yy], ind[Zz]);  // "iiii" means 4 integers
	Py_complex py_src = PyComplex_AsCComplex(pValue);
	Py_DECREF(pValue);

	*src_value = py_src.real + PETSC_i * py_src.imag;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "set_x_inc_at"
/**
 * set_x_inc_at
 * -------------
 * Set an element of the vector of the incident E field for the TF/SF.
 */
PetscErrorCode set_x_inc_at(PetscScalar *x_inc_value, Axis axis, const PetscInt ind[], GridInfo *gi)
{
	PetscFunctionBegin;

	if (!gi->is_pFunc_set) {
		gi->pFunc = PyObject_GetAttrString(gi->pSim, "get_incidentE_at");
		gi->is_pFunc_set = PETSC_TRUE;
	}

	PyObject *pValue = PyObject_CallFunction(gi->pFunc, (char *) "iiii", axis, ind[Xx], ind[Yy], ind[Zz]);  // "iiii" means 4 integers
	Py_complex py_x_inc = PyComplex_AsCComplex(pValue);
	Py_DECREF(pValue);

	*x_inc_value = py_x_inc.real + PETSC_i * py_x_inc.imag;

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
