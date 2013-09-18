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
	BC bc[Naxis];  // boundary conditions at -x, -y, -z boundaries
	GridType ge;  // grid type of the E-field grid
	PetscScalar exp_neg_ikL[Naxis];  // exp(-ik Lx), exp(-ik Ly), exp(-ik Lz)
	PetscScalar *s_factor[Naxis][Ngt];  // sx, sy, sz parameters of PML at primary and dual grid locations
	PetscScalar *dl[Naxis][Ngt];  // dx, dy, dz at primary and dual grid locations
	PetscScalar *dl_orig[Naxis][Ngt];  // original dl
	PetscReal lambda;  // normalized wavelength
	PetscReal omega;  // normalized angular frequency

	PetscBool has_mu;  // PETSC_TRUE if it has mu; PETSC_FALSE otherwise
	PetscBool has_xref;  // PETSC_TRUE if it has xref; PETSC_FALSE otherwise
	Vec xref;  // reference solution
	PetscReal norm_xref;  // infinity norm of xref
	PetscInt max_iter;  // maximum number of iteration of BiCG
	PetscReal tol;  // tolerance of BiCG
	PetscInt snapshot_interval;  // number of BiCG iterations between snapshots of approximate solutions

	Vec vecTemp; // template vector.  Also used as a temporary storage of a vector
	ISLocalToGlobalMapping map;  // local-to-global index mapping
	FieldType x_type;  // field type of the solution of the equation to formulate
	F0Type x0_type;  // how to generate x0
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
