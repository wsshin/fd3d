#include "gridinfo.h"

#undef __FUNCT__
#define __FUNCT__ "setGridInfo"
/**
 * setGridInfo
 * -----------
 * Set up the grid info.
 */
PetscErrorCode setGridInfo(GridInfo *gi, char *input_name)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	PetscInt axis, sign, n;

	/** Import Simulation object. */
	/** If Python does something funny, then most likely an error occurs in the folliwng three
	lines.  So check Python errors for the three lines.  Ideally, we need to check a Python error
	after each Python/C API function invocation, but I refrain from putting too many error checking
	codes for readability. */
	PyObject *pModule = PyImport_ImportModule(input_name); if (PyErr_Occurred()) PyErr_Print();
	gi->pSim = PyObject_GetAttrString(pModule, "sim"); if (PyErr_Occurred()) PyErr_Print();
	Py_DECREF(pModule); if (PyErr_Occurred()) PyErr_Print();

	/** Import values defined in the input file. */
	PyObject *pFunc, *pFunc_get_d_prim, *pFunc_get_d_dual, *pFunc_get_N, *pFunc_get_BC, *pFunc_get_s_prim, *pFunc_get_s_dual, *pFunc_get_exp_neg_ikL;
	PyObject *pValue;

	pFunc = PyObject_GetAttrString(gi->pSim, "get_wvlen");
	pValue = PyObject_CallFunction(pFunc, NULL);
	Py_DECREF(pFunc);
	Py_complex py_lambda = PyComplex_AsCComplex(pValue);
	gi->lambda = py_lambda.real + PETSC_i * py_lambda.imag;
	Py_DECREF(pValue);

	pFunc = PyObject_GetAttrString(gi->pSim, "get_omega");
	pValue = PyObject_CallFunction(pFunc, NULL);
	Py_DECREF(pFunc);
	Py_complex py_omega = PyComplex_AsCComplex(pValue);
	gi->omega = py_omega.real + PETSC_i * py_omega.imag;
	Py_DECREF(pValue);

	pFunc = PyObject_GetAttrString(gi->pSim, "get_BiCG_max_iter");
	pValue = PyObject_CallFunction(pFunc, NULL);
	Py_DECREF(pFunc);
	gi->max_iter = PyInt_AsLong(pValue);
	if (gi->max_iter < 0) gi->max_iter = PETSC_MAX_INT;
	Py_DECREF(pValue);

	pFunc = PyObject_GetAttrString(gi->pSim, "get_BiCG_tol");
	pValue = PyObject_CallFunction(pFunc, NULL);
	Py_DECREF(pFunc);
	gi->tol = PyFloat_AsDouble(pValue);
	Py_DECREF(pValue);

/*
	pFunc = PyObject_GetAttrString(gi->pSim, "get_snapshot_interval");
	pValue = PyObject_CallFunction(pFunc, NULL);
	Py_DECREF(pFunc);
	gi->snapshot_interval = PyInt_AsLong(pValue);
	if (gi->snapshot_interval < 0) gi->snapshot_interval = PETSC_MAX_INT;
	Py_DECREF(pValue);
*/

	pFunc_get_N = PyObject_GetAttrString(gi->pSim, "get_N");
	pFunc_get_BC = PyObject_GetAttrString(gi->pSim, "get_BC");
	pFunc_get_d_prim = PyObject_GetAttrString(gi->pSim, "get_d_prim");
	pFunc_get_d_dual = PyObject_GetAttrString(gi->pSim, "get_d_dual");
	pFunc_get_s_prim = PyObject_GetAttrString(gi->pSim, "get_s_prim");
	pFunc_get_s_dual = PyObject_GetAttrString(gi->pSim, "get_s_dual");
	pFunc_get_exp_neg_ikL = PyObject_GetAttrString(gi->pSim, "get_exp_neg_ikL");

	for (axis = 0; axis < Naxis; ++axis) {
		pValue = PyObject_CallFunction(pFunc_get_N, (char *) "i", axis);
		gi->N[axis] = PyInt_AsLong(pValue);
		Py_DECREF(pValue);

		pValue = PyObject_CallFunction(pFunc_get_exp_neg_ikL, (char *) "i", axis);
		Py_complex py_exp_neg_ikL = PyComplex_AsCComplex(pValue);
		gi->exp_neg_ikL[axis] = py_exp_neg_ikL.real + PETSC_i * py_exp_neg_ikL.imag;
		Py_DECREF(pValue);

		ierr = PetscMalloc6(
				gi->N[axis], PetscScalar, &gi->d_prim[axis],
				gi->N[axis], PetscScalar, &gi->d_dual[axis],
				gi->N[axis], PetscScalar, &gi->s_prim[axis],
				gi->N[axis], PetscScalar, &gi->s_dual[axis],
				gi->N[axis], PetscScalar, &gi->d_prim_orig[axis],
				gi->N[axis], PetscScalar, &gi->d_dual_orig[axis]); CHKERRQ(ierr);
		for (n = 0; n < gi->N[axis]; ++n) {
			Py_complex py_s, py_d;

			pValue = PyObject_CallFunction(pFunc_get_s_prim, (char *) "ii", axis, n);
			py_s = PyComplex_AsCComplex(pValue);
			gi->s_prim[axis][n] = py_s.real + PETSC_i * py_s.imag;
			Py_DECREF(pValue);

			pValue = PyObject_CallFunction(pFunc_get_s_dual, (char *) "ii", axis, n);
			py_s = PyComplex_AsCComplex(pValue);
			gi->s_dual[axis][n] = py_s.real + PETSC_i * py_s.imag;
			Py_DECREF(pValue);

			pValue = PyObject_CallFunction(pFunc_get_d_prim, (char *) "ii", axis, n);
			py_d = PyComplex_AsCComplex(pValue);
			gi->d_prim[axis][n] = py_d.real + PETSC_i * py_d.imag;
			gi->d_prim_orig[axis][n] = gi->d_prim[axis][n];
			Py_DECREF(pValue);

			pValue = PyObject_CallFunction(pFunc_get_d_dual, (char *) "ii", axis, n);
			py_d = PyComplex_AsCComplex(pValue);
			gi->d_dual[axis][n] = py_d.real + PETSC_i * py_d.imag;
			gi->d_dual_orig[axis][n] = gi->d_dual[axis][n];
			Py_DECREF(pValue);
		}

		for (sign = 0; sign < Nsign; ++sign) {
			pValue = PyObject_CallFunction(pFunc_get_BC, (char *) "ii", axis, sign);
			gi->bc[axis][sign] = (BC) PyInt_AsLong(pValue);
			Py_DECREF(pValue);
		}
	}

	Py_DECREF(pFunc_get_N);
	Py_DECREF(pFunc_get_BC);
	Py_DECREF(pFunc_get_exp_neg_ikL);
	Py_DECREF(pFunc_get_d_prim);
	Py_DECREF(pFunc_get_d_dual);
	Py_DECREF(pFunc_get_s_prim);
	Py_DECREF(pFunc_get_s_dual);

	gi->Ntot = gi->N[Xx] * gi->N[Yy] * gi->N[Zz] * Naxis;  // total # of unknowns

	/** Create distributed array (DA) representing Yee's grid, and set it in grid info. */
	const DMDABoundaryType ptype = DMDA_BOUNDARY_PERIODIC;
	const DMDAStencilType stype = DMDA_STENCIL_BOX;
	const PetscInt dof = Naxis;
	const PetscInt swidth = 1;

	ierr = DMDACreate3d(
			PETSC_COMM_WORLD,  // MPI communicator
			ptype, ptype, ptype, stype,  // type of peroodicity and stencil
			gi->N[Xx], gi->N[Yy], gi->N[Zz],   // global # of grid points in x,y,z
			PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,  // # of processes in x,y,z
			dof, swidth,  // degree of freedom, width of stencil
			PETSC_NULL, PETSC_NULL, PETSC_NULL,  // # of grid points in each process in x,y,z
			&gi->da); CHKERRQ(ierr);
	/*
	   ierr = DACreate3d(
	   PETSC_COMM_WORLD,  // MPI communicator
	   ptype, stype,  // type of peroodicity and stencil
	   gi->N[Xx], gi->N[Yy], gi->N[Zz],   // global # of grid points in x,y,z
	   8, 4, 8,  // # of processes in x,y,z
	   dof, swidth,  // degree of freedom, width of stencil
	   PETSC_NULL, PETSC_NULL, PETSC_NULL,  // # of grid points in each process in x,y,z
	   &gi->da); CHKERRQ(ierr);
	 */

	ierr = DMDAGetGhostCorners(gi->da, &gi->start_g[Xx], &gi->start_g[Yy], &gi->start_g[Zz], &gi->Nlocal_g[Xx], &gi->Nlocal_g[Yy], &gi->Nlocal_g[Zz]); CHKERRQ(ierr);
	ierr = DMDAGetCorners(gi->da, &gi->start[Xx], &gi->start[Yy], &gi->start[Zz], &gi->Nlocal[Xx], &gi->Nlocal[Yy], &gi->Nlocal[Zz]); CHKERRQ(ierr);
	gi->Nlocal_tot = gi->Nlocal[Xx] * gi->Nlocal[Yy] * gi->Nlocal[Zz] * Naxis;


	/** Create a template vector.  Other vectors are created to have duplicate
	  structure of this. */
	ierr = DMCreateGlobalVector(gi->da, &gi->vecTemp); CHKERRQ(ierr);

	/** Get local-to-global mapping from DA. */
	ierr = DMGetLocalToGlobalMapping(gi->da, &gi->map); CHKERRQ(ierr);

	/** Get the guess solution. */
	pFunc = PyObject_GetAttrString(gi->pSim, "get_sol_guess");
	pValue = PyObject_CallFunction(pFunc, NULL);
	if (PyErr_Occurred()) {
		PetscMPIInt rank;
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
		if (rank==0) PyErr_Print();
	}
	Py_DECREF(pFunc);
	if (pValue==Py_None) {
		gi->has_x0 = PETSC_FALSE;
	} else {
		gi->has_x0 = PETSC_TRUE;
		ierr = DMCreateGlobalVector(gi->da, &gi->x0); CHKERRQ(ierr);
		char *x0_name = PyString_AsString(pValue);
		//const char *prefix = "/in/";
		//char x0_name_prefixed[PETSC_MAX_PATH_LEN];
		//ierr = PetscStrcpy(x0_name_prefixed, getenv("FD3D_ROOT")); CHKERRQ(ierr);
		//ierr = PetscStrcat(x0_name_prefixed, prefix); CHKERRQ(ierr);
		//ierr = PetscStrcat(x0_name_prefixed, x0_name); CHKERRQ(ierr);
		PetscViewer viewer;
		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, x0_name_prefixed, FILE_MODE_READ, &viewer);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, x0_name, FILE_MODE_READ, &viewer);
		ierr = VecLoad(gi->x0, viewer); CHKERRQ(ierr);
		ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
		/** We should not free x0_name, because PyString_AsString() does not copy the 
		  original string. */
	}
	Py_DECREF(pValue);

	/** Get the reference solution. */
	pFunc = PyObject_GetAttrString(gi->pSim, "get_sol_reference");
	pValue = PyObject_CallFunction(pFunc, NULL);
	if (PyErr_Occurred()) {
		PetscMPIInt rank;
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
		if (rank==0) PyErr_Print();
	}
	Py_DECREF(pFunc);
	if (pValue==Py_None) {
		gi->has_xref = PETSC_FALSE;
	} else {
		gi->has_xref = PETSC_TRUE;
		ierr = DMCreateGlobalVector(gi->da, &gi->xref); CHKERRQ(ierr);
		char *xref_name = PyString_AsString(pValue);
		//const char *prefix = "/in/";
		//char xref_name_prefixed[PETSC_MAX_PATH_LEN];
		//ierr = PetscStrcpy(xref_name_prefixed, getenv("FD3D_ROOT")); CHKERRQ(ierr);
		//ierr = PetscStrcat(xref_name_prefixed, prefix); CHKERRQ(ierr);
		//ierr = PetscStrcat(xref_name_prefixed, xref_name); CHKERRQ(ierr);
		PetscViewer viewer;
		//PetscViewerBinaryOpen(PETSC_COMM_WORLD, xref_name_prefixed, FILE_MODE_READ, &viewer);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, xref_name, FILE_MODE_READ, &viewer);
		ierr = VecLoad(gi->xref, viewer); CHKERRQ(ierr);
		ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
		ierr = VecNorm(gi->xref, NORM_INFINITY, &gi->norm_xref); CHKERRQ(ierr);
		/** We should not free xref_name, because PyString_AsString() does not copy the 
		  original string. */
	}
	Py_DECREF(pValue);

	/** Get the incident field distribution for TF/SF. */
	pFunc = PyObject_GetAttrString(gi->pSim, "has_incidentE");
	pValue = PyObject_CallFunction(pFunc, NULL);
	Py_DECREF(pFunc);
	gi->has_xinc = (PetscBool) PyInt_AsLong(pValue);
	Py_DECREF(pValue);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setOptions"
/**
 * setOptions
 * -----------
 * Set options in the grid info.
 */
PetscErrorCode setOptions(GridInfo *gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Set flags with default values.  If the relevant fields are not set in the input file or 
	  fd3d_config file, these default values are used. */
	gi->bg_only = PETSC_FALSE;
	gi->x_type = Etype;
	gi->pml_type = SCPML;
	gi->pc_type = PCIdentity;
	gi->krylov_type = BiCG;
	gi->is_symmetric = PETSC_FALSE;
	gi->add_conteq = PETSC_FALSE;
	gi->factor_conteq = -1.0;
	gi->snapshot_interval = -1;
	gi->use_ksp = PETSC_FALSE;
	gi->output_mat_and_vec = PETSC_FALSE;
	gi->solve_eigen = PETSC_FALSE;
	gi->solve_singular = PETSC_FALSE;
	gi->verbose_level = VBDetail;
	gi->output_relres = PETSC_FALSE;

	const char *opt_name;
	char opt_str[PETSC_MAX_PATH_LEN];
	PetscBool has_opt, is_target_value, has_opt_val;

	/** Below, PetscOptionsGetString() null-terminates opt_str, I guess. */

	/** Field type */
	opt_name = "-fd3d_x_type";
	ierr = PetscOptionsGetString(PETSC_NULL, opt_name, opt_str, PETSC_MAX_PATH_LEN-1, &has_opt); CHKERRQ(ierr);  
	if (has_opt) {
		if (!(ierr = PetscStrcasecmp(opt_str, "E", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->x_type = Etype;
		} else if (!(ierr = PetscStrcasecmp(opt_str, "H", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->x_type = Htype;
		} else {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Unsupported option flag in %s; proceed with the default value.\n", opt_name); CHKERRQ(ierr);
		}
	}

	/** PML type */
	opt_name = "-fd3d_pml";
	ierr = PetscOptionsGetString(PETSC_NULL, opt_name, opt_str, PETSC_MAX_PATH_LEN-1, &has_opt); CHKERRQ(ierr);  
	if (has_opt) {
		if (!(ierr = PetscStrcasecmp(opt_str, "scpml", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->pml_type = SCPML;
		} else if (!(ierr = PetscStrcasecmp(opt_str, "upml", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->pml_type = UPML;
		} else {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Unsupported option flag in %s; proceed with the default value.\n", opt_name); CHKERRQ(ierr);
		}
	}

	/** Preconditioner type */
	opt_name = "-fd3d_pc";
	ierr = PetscOptionsGetString(PETSC_NULL, opt_name, opt_str, PETSC_MAX_PATH_LEN-1, &has_opt); CHKERRQ(ierr);  
	if (has_opt) {
		if (!(ierr = PetscStrcasecmp(opt_str, "sparam", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->pc_type = PCSparam;
		} else if (!(ierr = PetscStrcasecmp(opt_str, "eps", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->pc_type = PCEps;
		} else if (!(ierr = PetscStrcasecmp(opt_str, "jacobi", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->pc_type = PCJacobi;
		} else if (!(ierr = PetscStrcasecmp(opt_str, "identity", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->pc_type = PCIdentity;
		}else {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Unsupported option flag in %s; proceed with the default value.\n", opt_name); CHKERRQ(ierr);
		}
	}

	/** Symmetry */
	opt_name = "-fd3d_symmetric";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->is_symmetric); CHKERRQ(ierr);

	/** Addtion of the continuity equation */
	opt_name = "-fd3d_add_conteq";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->add_conteq); CHKERRQ(ierr);
	if (gi->add_conteq) {
		PetscBool has_opt_val;
		ierr = PetscOptionsGetReal(PETSC_NULL, opt_name, &gi->factor_conteq, &has_opt_val); CHKERRQ(ierr);
		if (!has_opt_val) {
			gi->factor_conteq = -1.0;
		}
	}

	/** Iterative solution snapshot interval */
	opt_name = "-fd3d_take_snapshots";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &has_opt); CHKERRQ(ierr);
	if (has_opt) {
		ierr = PetscOptionsGetInt(PETSC_NULL, opt_name, (PetscInt*) &gi->snapshot_interval, &has_opt_val); CHKERRQ(ierr);
		if (!has_opt_val) {
			gi->snapshot_interval = 1;
		}
	}

	/** Output A and b */
	opt_name = "-fd3d_output_mat_and_vec";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->output_mat_and_vec); CHKERRQ(ierr);

	/** KSP */
	opt_name = "-fd3d_use_ksp";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->use_ksp); CHKERRQ(ierr);

	/** Krylove subspace method */
	opt_name = "-fd3d_krylov";
	ierr = PetscOptionsGetString(PETSC_NULL, opt_name, opt_str, PETSC_MAX_PATH_LEN-1, &has_opt); CHKERRQ(ierr);  
	if (has_opt) {
		if (!(ierr = PetscStrcasecmp(opt_str, "bicg", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->krylov_type = BiCG;
		} else if (!(ierr = PetscStrcasecmp(opt_str, "qmr", &is_target_value)) && is_target_value) {
			CHKERRQ(ierr);
			gi->krylov_type = QMR;
		} else {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Unsupported option flag in %s; proceed with the default value.\n", opt_name); CHKERRQ(ierr);
		}
	}

	/** Eigensolver */
	opt_name = "-fd3d_eigen";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->solve_eigen); CHKERRQ(ierr);

	/** Singular value */
	opt_name = "-fd3d_singular";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->solve_singular); CHKERRQ(ierr);

	/** Verbose level */
	opt_name = "-fd3d_verbose";
	ierr = PetscOptionsGetInt(PETSC_NULL, opt_name, (PetscInt*) &gi->verbose_level, &has_opt_val); CHKERRQ(ierr);
	if (!has_opt_val) {
		gi->verbose_level = VBDetail;
	}

	/** Output the norms of relative residual vectors */
	opt_name = "-fd3d_output_relres";
	ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->output_relres); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
