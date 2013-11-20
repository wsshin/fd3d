#ifndef GUARD_output_h
#define GUARD_output_h

#include "gridinfo.h"

#include "petsc.h"

typedef struct {
	PetscReal norm_b;
	GridInfo *gi;
} MonitorCtx;

typedef PetscErrorCode (*MonitorIteration)(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat HE, const Vec conjParam, const Vec conjSrc, GridInfo *gi);

/**
 * output
 * ------
 * Output the E and H fields.
 */
PetscErrorCode output(char *output_name, const Vec x, const Mat GF, const Vec conjParam, const Vec conjSrc, const GridInfo gi);

/**
 * output_singular
 * ------
 * Output the left and right singular vectors.
 */
PetscErrorCode output_singular(char *output_name, const Vec u, const Vec v);

/**
 * output_mat_and_vec
 * ------
 * Output the matrices and vectors that can be used in MATLAB to solve various problems.
 */
PetscErrorCode output_mat_and_vec(const Mat A, const Vec b, const Vec right_precond, const Mat HE, const GridInfo gi);

/**
 * monitor_relres
 * --------------
 * Function to monitor the relative value of residual (= norm(r)/norm(b)).
 */
PetscErrorCode monitor_relres(KSP ksp, PetscInt n, PetscReal norm_r, void *ctx);

PetscErrorCode monitorRelres(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat HE, const Vec conjParam, const Vec conjSrc, GridInfo *gi);

PetscErrorCode monitorRelerr(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat HE, const Vec conjParam, const Vec conjSrc, GridInfo *gi);

PetscErrorCode monitorSnapshot(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat HE, const Vec conjParam, const Vec conjSrc, GridInfo *gi);

PetscErrorCode monitorAll(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat HE, const Vec conjParam, const Vec conjSrc, GridInfo *gi);

#endif
