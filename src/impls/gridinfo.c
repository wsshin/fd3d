#include "gridinfo.h"
#include "vec.h"

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
	PetscInt axis;
	
	inputfile_id = H5Fopen(gi->inputfile_name, H5F_ACC_RDONLY, H5P_DEFAULT);

	/** Import values defined in the input file. */
	ierr = h5get_data(inputfile_id, "/lambda", H5T_NATIVE_DOUBLE, &gi->lambda); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/omega", H5T_NATIVE_DOUBLE, &gi->omega); CHKERRQ(ierr);  // if gi->omega is PetscScalar, its imaginary part is garbage, and can be different between processors, which generates an error
	ierr = h5get_data(inputfile_id, "/maxit", H5T_NATIVE_INT, &gi->max_iter); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/tol", H5T_NATIVE_DOUBLE, &gi->tol); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/N", H5T_NATIVE_INT, gi->N); CHKERRQ(ierr);
	ierr = h5get_data(inputfile_id, "/bc", H5T_NATIVE_INT, gi->bc); CHKERRQ(ierr);

	PetscReal e_ikL[Naxis][Nri];
	ierr = h5get_data(inputfile_id, "/e_ikL", H5T_NATIVE_DOUBLE, e_ikL); CHKERRQ(ierr);
	ierr = ri2c(e_ikL, gi->exp_neg_ikL, Naxis); CHKERRQ(ierr);

ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "N = [%d, %d, %d]\n", gi->N[Xx], gi->N[Yy], gi->N[Zz]); CHKERRQ(ierr);

	/** Import values defined in the input file. */
/*
	pFunc = PyObject_GetAttrString(gi->pSim, "get_snapshot_interval");
	pValue = PyObject_CallFunction(pFunc, NULL);
	Py_DECREF(pFunc);
	gi->snapshot_interval = PyInt_AsLong(pValue);
	if (gi->snapshot_interval < 0) gi->snapshot_interval = PETSC_MAX_INT;
	Py_DECREF(pValue);
*/

	const char *w = "xyz";
	char datasetname[PETSC_MAX_PATH_LEN];
	for (axis = 0; axis < Naxis; ++axis) {
		PetscReal temp[gi->N[axis] * Nri];
		ierr = PetscMalloc6(
			gi->N[axis], PetscScalar, &gi->d_prim[axis],
			gi->N[axis], PetscScalar, &gi->d_dual[axis],
			gi->N[axis], PetscScalar, &gi->s_prim[axis],
			gi->N[axis], PetscScalar, &gi->s_dual[axis],
			gi->N[axis], PetscScalar, &gi->d_prim_orig[axis],
			gi->N[axis], PetscScalar, &gi->d_dual_orig[axis]); CHKERRQ(ierr);

		ierr = PetscStrcpy(datasetname, "/d"); CHKERRQ(ierr);
		ierr = append_char(datasetname, w[axis]); CHKERRQ(ierr);
		ierr = PetscStrcat(datasetname, "_prim"); CHKERRQ(ierr);
		ierr = h5get_data(inputfile_id, datasetname, H5T_NATIVE_DOUBLE, temp); CHKERRQ(ierr);
		ierr = ri2c(temp, gi->d_prim[axis], gi->N[axis]); CHKERRQ(ierr);
		ierr = ri2c(temp, gi->d_prim_orig[axis], gi->N[axis]); CHKERRQ(ierr);

		ierr = PetscStrcpy(datasetname, "/d"); CHKERRQ(ierr);
		ierr = append_char(datasetname, w[axis]); CHKERRQ(ierr);
		ierr = PetscStrcat(datasetname, "_dual"); CHKERRQ(ierr);
		ierr = h5get_data(inputfile_id, datasetname, H5T_NATIVE_DOUBLE, temp); CHKERRQ(ierr);
		ierr = ri2c(temp, gi->d_dual[axis], gi->N[axis]); CHKERRQ(ierr);
		ierr = ri2c(temp, gi->d_dual_orig[axis], gi->N[axis]); CHKERRQ(ierr);

		ierr = PetscStrcpy(datasetname, "/s"); CHKERRQ(ierr);
		ierr = append_char(datasetname, w[axis]); CHKERRQ(ierr);
		ierr = PetscStrcat(datasetname, "_prim"); CHKERRQ(ierr);
		ierr = h5get_data(inputfile_id, datasetname, H5T_NATIVE_DOUBLE, temp); CHKERRQ(ierr);
		ierr = ri2c(temp, gi->s_prim[axis], gi->N[axis]); CHKERRQ(ierr);

		ierr = PetscStrcpy(datasetname, "/s"); CHKERRQ(ierr);
		ierr = append_char(datasetname, w[axis]); CHKERRQ(ierr);
		ierr = PetscStrcat(datasetname, "_dual"); CHKERRQ(ierr);
		ierr = h5get_data(inputfile_id, datasetname, H5T_NATIVE_DOUBLE, temp); CHKERRQ(ierr);
		ierr = ri2c(temp, gi->s_dual[axis], gi->N[axis]); CHKERRQ(ierr);
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
	htri_t is_epsNode;
	ierr = PetscStrcpy(datasetname, "/eps_node"); CHKERRQ(ierr);
	is_epsNode = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	if (is_epsNode && is_epsNode >=0) {
		gi->has_epsNode = PETSC_TRUE;
	} else {
		gi->has_epsNode = PETSC_FALSE;
	}

	/** Set the flag for the initial guess solution. */
	htri_t isE0;
	ierr = PetscStrcpy(datasetname, "/E0"); CHKERRQ(ierr);
	isE0 = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	if (isE0 && isE0 >=0) {
		gi->has_x0 = PETSC_TRUE;
	} else {
		gi->has_x0 = PETSC_FALSE;
	}

	/** Set the flag for the reference solution. */
	htri_t isEref; 
	ierr = PetscStrcpy(datasetname, "/Eref"); CHKERRQ(ierr);
	isEref = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	if (isEref && isEref >= 0) {
		gi->has_xref = PETSC_TRUE;
		ierr = createVecHDF5(&gi->xref, datasetname, *gi); CHKERRQ(ierr);
	} else {
		gi->has_xref = PETSC_FALSE;
	}

	/** Set the flag for the incident field distribution for TF/SF. */
	htri_t isEinc;
	ierr = PetscStrcpy(datasetname, "/Einc"); CHKERRQ(ierr);
	isEinc = H5Lexists(inputfile_id, datasetname, H5P_DEFAULT);
	if (isEinc && isEinc >= 0) {
		gi->has_xinc = PETSC_TRUE;
	} else {
		gi->has_xinc = PETSC_FALSE;
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
