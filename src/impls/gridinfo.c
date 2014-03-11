#include "gridinfo.h"
#include "vec.h"
#include <assert.h>

#undef __FUNCT__
#define __FUNCT__ "append_char"
/**
 * append_char
 * -----------
 * Append a character to a string.
 */
PetscErrorCode append_char(char *target, const char c)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	size_t len;
	ierr = PetscStrlen(target, &len); CHKERRQ(ierr);
	target[len++] = c;
	target[len] = '\0';

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setGridInfo"
/**
 * setGridInfo
 * -----------
 * Set up the grid info.
 */
PetscErrorCode setGridInfo(GridInfo *gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	hid_t inputfile_id;
	herr_t status;
	PetscInt axis, gt;
	PetscReal temp_real;
	PetscReal temp_array[Naxis];
	FILE *file;
	char file_name[PETSC_MAX_PATH_LEN];

	inputfile_id = H5Fopen(gi->inputfile_name, H5F_ACC_RDONLY, H5P_DEFAULT);

	/** Import values defined in the input file. */
	ierr = h5get_data(inputfile_id, "/f", H5T_NATIVE_DOUBLE, &temp_real); CHKERRQ(ierr);
	gi->x_type = (PetscInt) temp_real;
	ierr = h5get_data(inputfile_id, "/ge", H5T_NATIVE_DOUBLE, &temp_real); CHKERRQ(ierr);
	gi->ge = (PetscInt) temp_real;
	ierr = h5get_data(inputfile_id, "/lambda", H5T_NATIVE_DOUBLE, &gi->lambda); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/omega", H5T_NATIVE_DOUBLE, &gi->omega); CHKERRQ(ierr);  // if gi->omega is PetscScalar, its imaginary part is garbage, and can be different between processors, which generates an error
	ierr = h5get_data(inputfile_id, "/x0_type", H5T_NATIVE_DOUBLE, &temp_real); CHKERRQ(ierr);
	gi->x0_type = (PetscInt) temp_real;
	//ierr = h5get_data(inputfile_id, "/maxit", H5T_NATIVE_INT, &gi->max_iter); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/maxit", H5T_NATIVE_DOUBLE, &temp_real); CHKERRQ(ierr);
	gi->max_iter = (PetscInt) temp_real;
	ierr = h5get_data(inputfile_id, "/tol", H5T_NATIVE_DOUBLE, &gi->tol); CHKERRQ(ierr);
	//ierr = h5get_data(inputfile_id, "/N", H5T_NATIVE_INT, gi->N); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/N", H5T_NATIVE_DOUBLE, temp_array); CHKERRQ(ierr);
	for (axis = 0; axis < Naxis; ++axis) {
		gi->N[axis] = (PetscInt) temp_array[axis];
	}
	//ierr = h5get_data(inputfile_id, "/bc", H5T_NATIVE_INT, gi->bc); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/bc", H5T_NATIVE_DOUBLE, temp_array); CHKERRQ(ierr);
	for (axis = 0; axis < Naxis; ++axis) {
		gi->bc[axis] = (PetscInt) temp_array[axis];
	}

	PetscReal e_ikL[Naxis][Nri];
	ierr = h5get_data(inputfile_id, "/e_ikL", H5T_NATIVE_DOUBLE, e_ikL); CHKERRQ(ierr);
	ierr = ri2c(e_ikL, (void *) gi->exp_neg_ikL, Naxis); CHKERRQ(ierr);

	ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\tN = [%d, %d, %d]\n", gi->N[Xx], gi->N[Yy], gi->N[Zz]); CHKERRQ(ierr);

	/** Import values defined in the input file. */
	/*
	   pFunc = PyObject_GetAttrString(gi->pSim, "get_snapshot_interval");
	   pValue = PyObject_CallFunction(pFunc, NULL);
	   Py_DECREF(pFunc);
	   gi->snapshot_interval = PyInt_AsLong(pValue);
	   if (gi->snapshot_interval < 0) gi->snapshot_interval = PETSC_MAX_INT;
	   Py_DECREF(pValue);
	 */

	char datasetname[PETSC_MAX_PATH_LEN];
	for (axis = 0; axis < Naxis; ++axis) {
		for (gt = 0; gt < Ngrid; ++gt) {
			PetscReal temp[gi->N[axis] * Nri];
			ierr = PetscMalloc4(
					gi->N[axis], PetscScalar, &gi->dl[axis][gt],
					gi->N[axis], PetscScalar, &gi->dl_orig[axis][gt], 
					gi->N[axis], PetscScalar, &gi->s_factor[axis][gt],
					gi->N[axis], PetscScalar, &gi->s_factor_orig[axis][gt]); CHKERRQ(ierr);

			ierr = PetscStrcpy(datasetname, "/d"); CHKERRQ(ierr);
			ierr = PetscStrcat(datasetname, AxisName[axis]); CHKERRQ(ierr);
			ierr = append_char(datasetname, '_'); CHKERRQ(ierr);
			ierr = PetscStrncat(datasetname, GridTypes[gt], 4); CHKERRQ(ierr);
			ierr = h5get_data(inputfile_id, datasetname, H5T_NATIVE_DOUBLE, temp); CHKERRQ(ierr);
			ierr = ri2c(temp, (void *) gi->dl_orig[axis][gt], gi->N[axis]); CHKERRQ(ierr);
			ierr = init_c(gi->dl[axis][gt], gi->N[axis]); CHKERRQ(ierr);

			ierr = PetscStrcpy(datasetname, "/s"); CHKERRQ(ierr);
			ierr = PetscStrcat(datasetname, AxisName[axis]); CHKERRQ(ierr);
			ierr = append_char(datasetname, '_'); CHKERRQ(ierr);
			ierr = PetscStrncat(datasetname, GridTypes[gt], 4); CHKERRQ(ierr);
			ierr = h5get_data(inputfile_id, datasetname, H5T_NATIVE_DOUBLE, temp); CHKERRQ(ierr);
			ierr = ri2c(temp, (void *) gi->s_factor_orig[axis][gt], gi->N[axis]); CHKERRQ(ierr);
			ierr = init_c(gi->s_factor[axis][gt], gi->N[axis]); CHKERRQ(ierr);
		}
	}

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

	/** Set the flag mu. */
	htri_t is_mu;
	ierr = PetscStrcpy(datasetname, "/mu"); CHKERRQ(ierr);
	is_mu = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	if (is_mu && is_mu >=0) {
		gi->has_mu = PETSC_TRUE;
	} else {
		gi->has_mu = PETSC_FALSE;
	}

	/** Set the flag for epsNode. */
	/*
	   htri_t is_epsNode;
	   ierr = PetscStrcpy(datasetname, "/eps_node"); CHKERRQ(ierr);
	   is_epsNode = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	   if (is_epsNode && is_epsNode >=0) {
	   gi->has_epsNode = PETSC_TRUE;
	   } else {
	   gi->has_epsNode = PETSC_FALSE;
	   }
	 */

	/** Set the flag for the initial guess solution. */
	/*
	   htri_t isE0;
	   ierr = PetscStrcpy(datasetname, "/E0"); CHKERRQ(ierr);
	   isE0 = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	   if (isE0 && isE0 >=0) {
	   gi->has_x0 = PETSC_TRUE;
	   } else {
	   gi->has_x0 = PETSC_FALSE;
	   }
	 */
	/*
	   ierr = PetscStrcpy(file_name, gi->input_name); CHKERRQ(ierr);
	   ierr = PetscStrcat(file_name, ".F0"); CHKERRQ(ierr);
	   file = fopen(file_name, "r");  // in a project directory
	   if (file) {
	   gi->has_x0 = PETSC_TRUE;
	   fclose(file);
	   } else {
	   gi->has_x0 = PETSC_FALSE;
	   }
	 */

	/** Set the flag for the reference solution. */
	/*
	   htri_t isEref; 
	   ierr = PetscStrcpy(datasetname, "/Eref"); CHKERRQ(ierr);
	   isEref = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	   if (isEref && isEref >= 0) {
	   gi->has_xref = PETSC_TRUE;
	   ierr = createVecHDF5(&gi->xref, datasetname, *gi); CHKERRQ(ierr);
	   } else {
	   gi->has_xref = PETSC_FALSE;
	   }
	 */
	ierr = PetscStrcpy(file_name, gi->input_name); CHKERRQ(ierr);
	ierr = PetscStrcat(file_name, ".Fref"); CHKERRQ(ierr);
	file = fopen(file_name, "r");  // in a project directory
	if (file) {
		gi->has_xref = PETSC_TRUE;
		ierr = createVecPETSc(&gi->xref, "Fref", *gi);
		fclose(file);
	} else {
		gi->has_xref = PETSC_FALSE;
	}

	status = H5Fclose(inputfile_id);

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

	/*
	const char *opt_name;
	char opt_str[PETSC_MAX_PATH_LEN];
	PetscBool has_opt, is_target_value, has_opt_val;
	PetscInt i;
	PetscInt opt_int;
	PetscReal opt_real;
	*/

	/** Set flags with default values.  If the relevant fields are not set in the input file or 
	  fd3d_config file, these default values are used. */
	gi->cell_type = CELL_SD;  // SC-PML
	gi->sym_type = SYM_SQRTAL;  // diagonal-preserving symmetrization
	gi->pc_type = PRECOND_1;  // identity preconditioner
	gi->krylov_type = KRYLOV_BICG;
	gi->add_conteq = PETSC_FALSE;
	gi->factor_conteq = -1.0;
	gi->snapshot_interval = -1;
	gi->use_ksp = PETSC_FALSE;
	gi->output_mat_and_vec = PETSC_FALSE;
	gi->solve_eigen = PETSC_FALSE;
	gi->solve_singular = PETSC_FALSE;
	gi->verbose_level = VB_DETAIL;
	gi->output_relres = PETSC_FALSE;

	/** Below, PetscOptionsGetString() null-terminates opt_str, I guess. */
	ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "FD3D options", ""); CHKERRQ(ierr);
	{
		ierr = PetscOptionsInt("-fd3d_maxit", "Maximum number of iteration", "", gi->max_iter, &gi->max_iter, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsReal("-fd3d_tol", "Tolerance in relative residual norm", "", gi->tol, &gi->tol, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsEnum("-fd3d_x0", "Initial guess for x of Ax = b", "", X0Types, (PetscEnum)gi->x0_type, (PetscEnum*)&gi->x0_type, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsEnum("-fd3d_field", "Field to solve for", "", FieldTypes, (PetscEnum)gi->x_type, (PetscEnum*)&gi->x_type, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsEnum("-fd3d_cell", "Cell size type", "", CellTypes, (PetscEnum)gi->cell_type, (PetscEnum*)&gi->cell_type, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsEnum("-fd3d_pc", "Preconditioner", "", PrecondTypes, (PetscEnum)gi->pc_type, (PetscEnum*)&gi->pc_type, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsEnum("-fd3d_sym", "Symmetry type", "", SymTypes, (PetscEnum)gi->sym_type, (PetscEnum*)&gi->sym_type, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsEnum("-fd3d_krylov", "Krylov subspace method", "", KrylovTypes, (PetscEnum)gi->krylov_type, (PetscEnum*)&gi->krylov_type, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsReal("-fd3d_add_conteq", "Add continuity equation", "", gi->factor_conteq, &gi->factor_conteq, &gi->add_conteq); CHKERRQ(ierr);
		ierr = PetscOptionsBool("-fd3d_output_relres", "Output relative residual norms", "", gi->output_relres, &gi->output_relres, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsBool("-fd3d_output_mat_and_vec", "Output matrices and vectors such as A and b", "", gi->output_mat_and_vec, &gi->output_mat_and_vec, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsInt("-fd3d_snapshot_interval", "Snapshot interval", "", gi->snapshot_interval, &gi->snapshot_interval, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsBool("-fd3d_use_ksp", "Use KSP", "", gi->use_ksp, &gi->use_ksp, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsBool("-fd3d_solve_eigen", "Solve eigenvalue equation", "", gi->solve_eigen, &gi->solve_eigen, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsBool("-fd3d_solve_singular", "Solve singular value equation", "", gi->solve_singular, &gi->solve_singular, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscOptionsEnum("-fd3d_verbose", "Verbose level", "", VBTypes, (PetscEnum)gi->verbose_level, (PetscEnum*)&gi->verbose_level, PETSC_NULL); CHKERRQ(ierr);
	}
	ierr = PetscOptionsEnd(); CHKERRQ(ierr);


	/*
	   opt_name = "-fd3d_add_conteq";
	   ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &gi->add_conteq); CHKERRQ(ierr);
	   if (gi->add_conteq) {
	   PetscBool has_opt_val;
	   ierr = PetscOptionsGetReal(PETSC_NULL, opt_name, &gi->factor_conteq, &has_opt_val); CHKERRQ(ierr);
	   if (!has_opt_val) {
	   gi->factor_conteq = -1.0;
	   }
	   }

	   opt_name = "-fd3d_take_snapshots";
	   ierr = PetscOptionsHasName(PETSC_NULL, opt_name, &has_opt); CHKERRQ(ierr);
	   if (has_opt) {
	   ierr = PetscOptionsGetInt(PETSC_NULL, opt_name, (PetscInt*) &gi->snapshot_interval, &has_opt_val); CHKERRQ(ierr);
	   if (!has_opt_val) {
	   gi->snapshot_interval = 1;
	   }
	   }
	 */

	PetscFunctionReturn(0);
}

/**
 * stretch_s
 * ---------
 * Stretches sx, sy, sz with given factors.
 * Note that this function can be written to take GridInfo instead of GridInfo* because s_factor is 
 * a pointer variable; even if GridInfo were used and the argument gi is delivered as a 
 * copy, the pointer value s_factor is the same as the original, so modifying s_factor[axis][gt][n] 
 * modifies the original elements.
 * However, to make sure that users understand that the contents of gi change in this function, this
 * function is written to take GridInfo*.
 */
#undef __FUNCT__
#define __FUNCT__ "stretch_s"
PetscErrorCode stretch_s(GridInfo *gi, const PetscScalar *factor[Naxis][Ngrid])
{
	PetscFunctionBegin;

	/** Stretch gi.s_factor by gi.dl. */
	PetscInt axis, gt, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (gt = 0; gt < Ngrid; ++gt) {
			for (n = 0; n < gi->N[axis]; ++n) {
				gi->s_factor[axis][gt][n] *= factor[axis][gt][n];
			}
		}
	}

	PetscFunctionReturn(0);
}

/**
 * stretch_d
 * ---------
 * Stretches dx, dy, dz with given factors.
 * Note that this function can be written to take GridInfo instead of GridInfo* because dl is 
 * a pointer variable; even if GridInfo were used and the argument gi is delivered as a 
 * copy, the pointer value dl is the same as the original, so modifying dl[axis][gt][n] modifies 
 * the original elements.
 * However, to make sure that users understand that the contents of gi change in this function, this
 * function is written to take GridInfo*.
 */
#undef __FUNCT__
#define __FUNCT__ "stretch_d"
PetscErrorCode stretch_d(GridInfo *gi, const PetscScalar *factor[Naxis][Ngrid])
{
	PetscFunctionBegin;

	/** Stretch gi.dl by gi.s_factor. */
	PetscInt axis, gt, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (gt = 0; gt < Ngrid; ++gt) {
			for (n = 0; n < gi->N[axis]; ++n) {
				gi->dl[axis][gt][n] *= factor[axis][gt][n];
			}
		}
	}

	PetscFunctionReturn(0);
}

/**
 * init_s_d
 * ---------
 * Initialized s and d.
 */
#undef __FUNCT__
#define __FUNCT__ "init_s_d"
PetscErrorCode init_s_d(GridInfo *gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	switch(gi->cell_type) {
		case CELL_SD:  // d = s*d, s = 1
			ierr = stretch_d(gi, gi->s_factor_orig); CHKERRQ(ierr);
			ierr = stretch_d(gi, gi->dl_orig); CHKERRQ(ierr);
			break;
		case CELL_D:  // d = d, s = s
			ierr = stretch_d(gi, gi->dl_orig); CHKERRQ(ierr);
			ierr = stretch_s(gi, gi->s_factor_orig); CHKERRQ(ierr);
			break;
		case CELL_S:  // d = s, s = d
			ierr = stretch_d(gi, gi->s_factor_orig); CHKERRQ(ierr);
			ierr = stretch_s(gi, gi->dl_orig); CHKERRQ(ierr);
			break;
		case CELL_1:  // d = 1, s = s*d
			ierr = stretch_s(gi, gi->s_factor_orig); CHKERRQ(ierr);
			ierr = stretch_s(gi, gi->dl_orig); CHKERRQ(ierr);
			break;
		default:  // should not happen
			assert(PETSC_FALSE);
	}

	PetscFunctionReturn(0);
}

