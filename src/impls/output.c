#include <assert.h>
#include "output.h"
#include "vec.h"
#include "assert.h"

#undef __FUNCT__
#define __FUNCT__ "output"
/**
 * output
 * ------
 * Output the E and H fields.
 */
PetscErrorCode output(char *output_name, const Vec x, const Mat CF, const Vec conjParam, const Vec conjSrc, const GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	char output_name_prefixed[PETSC_MAX_PATH_LEN];
	//const char *prefix = "/out/";
	const char *h_extension = ".H.h5";
	const char *e_extension = ".E.h5";

	//ierr = PetscStrcpy(output_name_prefixed, getenv("FD3D_ROOT")); CHKERRQ(ierr);
	//ierr = PetscStrcat(output_name_prefixed, prefix); CHKERRQ(ierr);
	//ierr = PetscStrcat(output_name_prefixed, output_name); CHKERRQ(ierr);
	ierr = PetscStrcpy(output_name_prefixed, output_name); CHKERRQ(ierr);

	char h_file[PETSC_MAX_PATH_LEN];
	char e_file[PETSC_MAX_PATH_LEN];

	ierr = PetscStrcpy(h_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(h_file, h_extension); CHKERRQ(ierr);
	ierr = PetscStrcpy(e_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(e_file, e_extension); CHKERRQ(ierr);

	Vec y;  // H field vector if x_type == Etype
	ierr = VecDuplicate(gi.vecTemp, &y); CHKERRQ(ierr);
	ierr = VecCopy(conjSrc, y); CHKERRQ(ierr);
	if (gi.x_type==Etype) {
		ierr = MatMultAdd(CF, x, y, y); CHKERRQ(ierr);
		ierr = VecScale(y, -1.0/PETSC_i/gi.omega); CHKERRQ(ierr);
	} else {
		ierr = VecScale(y, -1.0); CHKERRQ(ierr);
		ierr = MatMultAdd(CF, x, y, y); CHKERRQ(ierr);
		ierr = VecScale(y, 1.0/PETSC_i/gi.omega); CHKERRQ(ierr);
	}
	ierr = VecPointwiseDivide(y, y, conjParam);

	PetscViewer viewer;

	//viewer = PETSC_VIEWER_STDOUT_WORLD;
	//ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, h_file, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);

	/** Write the E-field file. */
	ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, e_file, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerHDF5PushGroup(viewer, "/");
	if (gi.x_type==Etype) {
		ierr = PetscObjectSetName((PetscObject) x, "E"); CHKERRQ(ierr);
		ierr = VecView(x, viewer); CHKERRQ(ierr);
	} else {
		assert(gi.x_type==Htype);
		ierr = PetscObjectSetName((PetscObject) y, "E"); CHKERRQ(ierr);
		ierr = VecView(y, viewer); CHKERRQ(ierr);
	}
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	/** Write the H-field file. */
	ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, h_file, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerHDF5PushGroup(viewer, "/");
	if (gi.x_type==Etype) {
		ierr = PetscObjectSetName((PetscObject) y, "H"); CHKERRQ(ierr);
		ierr = VecView(y, viewer); CHKERRQ(ierr);
	} else {
		assert(gi.x_type==Htype);
		ierr = PetscObjectSetName((PetscObject) x, "H"); CHKERRQ(ierr);
		ierr = VecView(x, viewer); CHKERRQ(ierr);
	}
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	ierr = VecDestroy(&y); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "output_singular"
/**
 * output_singular
 * ------
 * Output the left and right singular vectors.
 */
PetscErrorCode output_singular(char *output_name, const Vec u, const Vec v)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	char output_name_prefixed[PETSC_MAX_PATH_LEN];
	//const char *prefix = "/out/";
	const char *u_extension = ".U";
	const char *v_extension = ".V";

	//ierr = PetscStrcpy(output_name_prefixed, getenv("FD3D_ROOT")); CHKERRQ(ierr);
	//ierr = PetscStrcat(output_name_prefixed, prefix); CHKERRQ(ierr);
	//ierr = PetscStrcat(output_name_prefixed, output_name); CHKERRQ(ierr);
	ierr = PetscStrcpy(output_name_prefixed, output_name); CHKERRQ(ierr);

	char u_file[PETSC_MAX_PATH_LEN];
	char v_file[PETSC_MAX_PATH_LEN];

	ierr = PetscStrcpy(u_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(u_file, u_extension); CHKERRQ(ierr);
	ierr = PetscStrcpy(v_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(v_file, v_extension); CHKERRQ(ierr);

	PetscViewer viewer;

	//viewer = PETSC_VIEWER_STDOUT_WORLD;
	//ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, h_file, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);

	/** Write the left singular vector u. */
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
	ierr = PetscViewerBinarySkipInfo(viewer); CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(viewer, u_file); CHKERRQ(ierr);
	/*
	   ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, e_file, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
	 */
	ierr = VecView(u, viewer); CHKERRQ(ierr);

	/** Write the right singular vector v. */
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
	ierr = PetscViewerBinarySkipInfo(viewer); CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(viewer, v_file); CHKERRQ(ierr);
	/*
	   ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, h_file, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
	 */
	ierr = VecView(v, viewer); CHKERRQ(ierr);

	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "output_mat_and_vec"
/**
 * output_mat_and_vec
 * ------
 * Output the matrices and vectors that can be used in MATLAB to solve various problems.
 */
PetscErrorCode output_mat_and_vec(const Mat A, const Vec b, const Vec right_precond, const Mat CF, const GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	char output_name_prefixed[PETSC_MAX_PATH_LEN];
	//const char *prefix = "/out/";
	const char *ind_extension = "_ind";
	const char *A_extension = "_A";
	const char *b_extension = "_b";
	const char *precond_extension = "_pR";
	const char *CF_extension = "_CF";

	//ierr = PetscStrcpy(output_name_prefixed, getenv("FD3D_ROOT")); CHKERRQ(ierr);
	//ierr = PetscStrcat(output_name_prefixed, prefix); CHKERRQ(ierr);
	//ierr = PetscStrcat(output_name_prefixed, gi.output_name); CHKERRQ(ierr);
	ierr = PetscStrcpy(output_name_prefixed, gi.output_name); CHKERRQ(ierr);

	char ind_file[PETSC_MAX_PATH_LEN];
	char A_file[PETSC_MAX_PATH_LEN];
	char b_file[PETSC_MAX_PATH_LEN];
	char precond_file[PETSC_MAX_PATH_LEN];
	char CF_file[PETSC_MAX_PATH_LEN];

	ierr = PetscStrcpy(ind_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(ind_file, ind_extension); CHKERRQ(ierr);
	ierr = PetscStrcpy(A_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(A_file, A_extension); CHKERRQ(ierr);
	ierr = PetscStrcpy(b_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(b_file, b_extension); CHKERRQ(ierr);
	ierr = PetscStrcpy(precond_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(precond_file, precond_extension); CHKERRQ(ierr);
	ierr = PetscStrcpy(CF_file, output_name_prefixed); CHKERRQ(ierr);
	ierr = PetscStrcat(CF_file, CF_extension); CHKERRQ(ierr);

	PetscViewer viewer;

	/** It turns out that VecView() shows the DA vector in natural order.  Therefore, even though 
	  indApp is constructed in application order, it is shown in natural order by VecView().  
	  On the other hand, indNat reorder indApp in natural order and then distribute the vector to 
	  processors.  However, because VecView() reorder a vector before it prints out the content of 
	  the vector, indNat is shown messy by VecView(). 
	  Inconsistently, MatView() does not reorder the matrix elements into natural order before it 
	  shows the matrix.  Therefore, when the binaries of matrices and vectors are imported in MATLAB,
	  I need to reorder the matrices but not the vectors. */
	Vec indApp;
	//ierr = create_index(&indApp, gi); CHKERRQ(ierr);
	ierr = createFieldArray(&indApp, set_index_at, gi); CHKERRQ(ierr);
	//ierr = VecView(indApp, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	Vec indNat;
	ierr = DMDACreateNaturalVector(gi.da, &indNat); CHKERRQ(ierr);
	ierr = VecCopy(indApp, indNat); CHKERRQ(ierr);
	ierr = VecDestroy(&indApp); CHKERRQ(ierr);
	//ierr = VecView(indNat, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	/** Write the index vector ind_app. */
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(viewer, ind_file); CHKERRQ(ierr);
	ierr = VecView(indNat, viewer); CHKERRQ(ierr);
	ierr = VecDestroy(&indNat); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	/** Write the coefficient matrix A. */
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(viewer, A_file); CHKERRQ(ierr);
	ierr = MatView(A, viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	/** Write the RHS vector b. */
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(viewer, b_file); CHKERRQ(ierr);
	ierr = VecView(b, viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	/** Write the right preconditioner vector pR. */
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(viewer, precond_file); CHKERRQ(ierr);
	ierr = VecView(right_precond, viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	/** Write the E-to-H converter matrix CF. */
	ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer); CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERBINARY); CHKERRQ(ierr);
	ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
	ierr = PetscViewerFileSetName(viewer, CF_file); CHKERRQ(ierr);
	ierr = MatView(CF, viewer); CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "monitor_relres"
/**
 * monitor_relres
 * --------------
 * Function to monitor the relative value of residual (= norm(r)/norm(b)).
 */
PetscErrorCode monitor_relres(KSP ksp, PetscInt n, PetscReal norm_r, void *ctx)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	MonitorCtx *monitor_ctx = (MonitorCtx *) ctx;
	PetscReal norm_b = monitor_ctx->norm_b;
	GridInfo *gi = monitor_ctx->gi;
	PetscReal rel_res = norm_r/norm_b;

	if (gi->output_relres) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, gi->relres_file, "%d,%e\n", n, rel_res); CHKERRQ(ierr);
	}

	ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\tnumiter: %d\tnorm(r)/norm(b) = %e\n", n, rel_res); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "monitorRelres"
PetscErrorCode monitorRelres(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat CF, const Vec conjParam, const Vec conjSrc, GridInfo *gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	if (gi->output_relres) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, gi->relres_file, "%d,%e\n", num_iter, rel_res); CHKERRQ(ierr);
	}

	if (gi->verbose_level >= vl) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\tnumiter: %d\trelres: %e\n", num_iter, rel_res); CHKERRQ(ierr);
	}

	/*
	   PetscInt axis, n;
	   for (axis = 0; axis < Naxis; ++axis) {
	   for (n = 0; n < gi->N[axis]; ++n) {
	   gi->d_dual[axis][n] *= gi->s_prim[axis][n];
	   gi->d_prim[axis][n] *= gi->s_dual[axis][n];
	   }
	   }

	   Mat DivE;
	   ierr = createDivE(&DivE, *gi); CHKERRQ(ierr);

	   Vec epsE0;
	   ierr = create_jSrc(&epsE0, *gi); CHKERRQ(ierr);
	   ierr = VecScale(epsE0, -1.0/PETSC_i/gi->omega); CHKERRQ(ierr);

	   Vec y;
	   ierr = VecDuplicate(gi->vecTemp, &y); CHKERRQ(ierr);

	   ierr = VecAYPX(epsE0, -1.0, x); CHKERRQ(ierr);
	   ierr = MatMult(DivE, epsE0, y);

	   PetscReal norm_x, norm_y;
	   ierr = VecNorm(x, NORM_INFINITY, &norm_x);
	   ierr = VecNorm(y, NORM_INFINITY, &norm_y);

	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\ttransversality: %e\n", num_iter, norm_y/norm_x); CHKERRQ(ierr);
	//ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\ttransversality: %e\n", num_iter, norm_y); CHKERRQ(ierr);

	for (axis = 0; axis < Naxis; ++axis) {
	for (n = 0; n < gi->N[axis]; ++n) {
	gi->d_dual[axis][n] /= gi->s_prim[axis][n];
	gi->d_prim[axis][n] /= gi->s_dual[axis][n];
	}
	}

	ierr = MatDestroy(DivE); CHKERRQ(ierr);
	ierr = VecDestroy(y); CHKERRQ(ierr);
	 */

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "monitorRelerr"
PetscErrorCode monitorRelerr(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat CF, const Vec conjParam, const Vec conjSrc, GridInfo *gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	if (gi->verbose_level >= vl && gi->has_xref) {
		Vec dx = gi->vecTemp;
		ierr = VecPointwiseDivide(dx, x, right_precond); CHKERRQ(ierr);
		ierr = VecAXPY(dx, -1.0, gi->xref); CHKERRQ(ierr);

		PetscReal norm_dx;
		ierr = VecNorm(dx, NORM_INFINITY, &norm_dx); CHKERRQ(ierr);
		PetscReal rel_err = norm_dx / gi->norm_xref;

		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\trelres: %e, \trelerr: %e\n", num_iter, rel_res, rel_err); CHKERRQ(ierr);
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "monitorSnapshot"
PetscErrorCode monitorSnapshot(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat CF, const Vec conjParam, const Vec conjSrc, GridInfo *gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	char snapshot_name[PETSC_MAX_PATH_LEN];
	char num_iter_str[PETSC_MAX_PATH_LEN];
	Vec x_snapshot = gi->vecTemp;

	if (gi->verbose_level >= vl && gi->snapshot_interval > 0 && num_iter >= 0 && num_iter % gi->snapshot_interval == 0) {
		ierr = PetscStrcpy(snapshot_name, gi->output_name); CHKERRQ(ierr);
		ierr = PetscStrcat(snapshot_name, "."); CHKERRQ(ierr);
		//sprintf(num_iter_str, "%d", num_iter);
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "%d", num_iter); CHKERRQ(ierr);
		ierr = PetscStrcat(snapshot_name, num_iter_str); CHKERRQ(ierr);
		ierr = VecCopy(x, x_snapshot); CHKERRQ(ierr);
		ierr = VecPointwiseDivide(x_snapshot, x_snapshot, right_precond); CHKERRQ(ierr);
		ierr = output(snapshot_name, x_snapshot, CF, conjParam, conjSrc, *gi); CHKERRQ(ierr);
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "monitorAll"
PetscErrorCode monitorAll(const VerboseLevel vl, const Vec x, const Vec right_precond, const PetscInt num_iter, const PetscReal rel_res, const Mat CF, const Vec conjParam, const Vec conjSrc, GridInfo *gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;


	if (gi->has_xref) {
		ierr = monitorRelerr(vl, x, right_precond, num_iter, rel_res, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);
	} else {
		ierr = monitorRelres(vl, x, right_precond, num_iter, rel_res, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);
	}

	ierr = monitorSnapshot(vl, x, right_precond, num_iter, rel_res, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
