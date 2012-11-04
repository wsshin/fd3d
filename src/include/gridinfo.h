#ifndef GUARD_gridinfo_h
#define GUARD_gridinfo_h

#include "petsc.h"
#include "hdf5.h"
#include "type.h"
#include "h5.h"

typedef struct {
	char input_name[PETSC_MAX_PATH_LEN];
	char inputfile_name[PETSC_MAX_PATH_LEN];
	char output_name[PETSC_MAX_PATH_LEN];
	DM da;  // distributed array
	PetscInt N[Naxis];  // # of grid points in x, y, z
	PetscInt Ntot;  // total # of unknowns
	PetscInt Nlocal_tot;  // total # of local unknowns
	PetscInt Nlocal[Naxis];  // # of local grid points in x, y, z
	PetscInt start[Naxis]; // local starting points in x, y, z
	PetscInt Nlocal_g[Naxis];  // # of local grid points in x, y, z including ghost points
	PetscInt start_g[Naxis]; // local starting points in x, y, z including ghost points
	BC bc[Naxis][Nsign];  // boundary conditions at +-x, +-y, +-z
	PetscScalar exp_neg_ikL[Naxis];  // exp(-ik Lx), exp(-ik Ly), exp(-ik Lz)
	PetscScalar *s_prim[Naxis];  // sx, sy, sz parameters of UPML at primary grid positions (at integral indices)
	PetscScalar *s_dual[Naxis];  //  sx, sy, sz parameters of UPML at dual grid positions (at half integral indices)
	PetscScalar *d_prim[Naxis];  // dx_prim, dy_prim, dz_prim
	PetscScalar *d_dual[Naxis];  // dx_dual, dy_dual, dz_dual
	PetscScalar *d_prim_orig[Naxis];  // original dx_prim, dy_prim, dz_prim before stretched
	PetscScalar *d_dual_orig[Naxis];  // original dx_dual, dy_dual, dz_dual before stretched
	PetscScalar lambda;  // normalized wavelength
	PetscScalar omega;  // normalized angular frequency

	PetscBool has_x0;  // PETSC_TRUE if it has x0; PETSC_FALSE otherwise
	Vec x0;  // guess solution
	PetscBool has_xref;  // PETSC_TRUE if it has xref; PETSC_FALSE otherwise
	Vec xref;  // reference solution
	PetscBool has_xinc;  // PETSC_TRUE if it has xinc; PETSC_FALSE otherwise
	Vec xinc;  // incident field
	PetscReal norm_xref;  // infinity norm of xref
	PetscInt max_iter;  // maximum number of iteration of BiCG
	PetscReal tol;  // tolerance of BiCG
	PetscInt snapshot_interval;  // number of BiCG iterations between snapshots of approximate solutions
	PetscBool bg_only;  // PETSC_TRUE to account for the background objects only; PETSC_FALSE to account for the foreground objects together

	Vec vecTemp; // template vector.  Also used as a temporary storage of a vector
	ISLocalToGlobalMapping map;  // local-to-global index mapping
	FieldType x_type;
	PMLType pml_type;
	KrylovType krylov_type;
	PrecondType pc_type;
	PetscBool is_symmetric;
	PetscBool add_conteq;
	PetscReal factor_conteq;  // factor multiplied to the continuity equation before adding the eq
	PetscBool use_ksp;
	PetscBool output_mat_and_vec;
	PetscBool solve_eigen;
	PetscBool solve_singular;
	VerboseLevel verbose_level; 
	PetscBool output_relres;  // output the norms of relative residual vectors to a file
	FILE *relres_file;
} GridInfo;

/**
 * setGridInfo
 * -----------
 * Set up the grid info.
 */
PetscErrorCode setGridInfo(GridInfo *gi);

/**
 * setOptions
 * -----------
 * Set options in the grid info.
 */
PetscErrorCode setOptions(GridInfo *gi);

#endif
