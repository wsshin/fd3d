#include "petsc.h"
#include "hdf5.h"
#include "type.h"
#include "gridinfo.h"
#include "logging.h"
#include "vec.h"
#include "mat.h"
#include "solver.h"
#include "output.h"
#include <assert.h>

//ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "I'm here!\n"); CHKERRQ(ierr);
//ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//ierr = MatView(mat, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

/** Function names beginning with a capital character (e.g. VecDuplicate()) are PETSc 
  functions; others are custom functions. */
/** Below, my naming convention for Mat, Vec, array is: 
  all-capital notation (e.g. CEH) for Mat 
  Hungarian notation with the first letter capitalized (e.g. DivE) for Mat 
  Hungarian notation (e.g. negW2Mu) for Vec 
  notation with _ (e.g. neg_w2mu) for array */

const char *OPTION_FILE_NAME = "fd3d_config";

#undef __FUNCT__
#define __FUNCT__ "cleanup"
PetscErrorCode cleanup(Mat A, Vec b, Vec right_precond, Mat CF, Vec conjParam, Vec conjSrc, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Destroy matrices and vectotrs used in BiCG. */
	ierr = MatDestroy(&A); CHKERRQ(ierr);  // destroy A == CGF
	ierr = VecDestroy(&b); CHKERRQ(ierr);
	ierr = VecDestroy(&right_precond); CHKERRQ(ierr);
	ierr = MatDestroy(&CF); CHKERRQ(ierr);  // destroy CF == CH or CE
	ierr = VecDestroy(&conjParam); CHKERRQ(ierr);
	ierr = VecDestroy(&conjSrc); CHKERRQ(ierr);

	/** Finalize the program. */
	ierr = DMDestroy(&gi.da); CHKERRQ(ierr);
	ierr = VecDestroy(&gi.vecTemp); CHKERRQ(ierr);
	if (gi.has_xref) {
		ierr = VecDestroy(&gi.xref); CHKERRQ(ierr);
	}

	// Need to move this to gridinfo.c
	PetscInt axis, sign;
	for (axis = 0; axis < Naxis; ++axis) {
		for (sign = 0; sign < Nsign; ++sign) {
			ierr = PetscFree3(gi.dl[axis][sign], gi.s_factor[axis][sign], gi.dl_orig[axis][sign]); CHKERRQ(ierr);
		}
	}
	//ISLocalToGlobalMappingDestroy(gi.map); CHKERRQ(ierr);

	/** Print the summary of the program execution. */
	if (gi.verbose_level >= VBMedium) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "fd3d finished.\n"); CHKERRQ(ierr);
	}

	ierr = PetscFinalize(); CHKERRQ(ierr);  // finalize PETSc.
	PetscFunctionReturn(0);
}

static char help[] = "Usage: <job launcher with # of proc (ex. mpirun -np 4)> ./fd3d -i <input file> [-o <output file>]\n\
					  File names should be written without file name extensions.\n";

#undef __FUNCT__
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char **argv)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Initialize PETSc. */
	char option_file_name[PETSC_MAX_PATH_LEN];
	strcpy(option_file_name, OPTION_FILE_NAME);

	FILE *option_file;
	option_file = fopen(option_file_name, "r");  // in a project directory
	if (!option_file) {  // in the parent directory of each project directory
		option_file_name[0] = '\0';
		strcpy(option_file_name, "..");
		strcat(option_file_name, "/");
		strcat(option_file_name, OPTION_FILE_NAME);
		option_file = fopen(option_file_name, "r");
	}

	if (!option_file) {  // in $FD3D_ROOT directory
		option_file_name[0] = '\0';
		strcpy(option_file_name, getenv("FD3D_ROOT"));
		strcat(option_file_name, "/");
		strcat(option_file_name, OPTION_FILE_NAME);
		option_file = fopen(option_file_name, "r");
	}

	if (option_file) {
		fclose(option_file);
		ierr = PetscInitialize(&argc, &argv, option_file_name, help); CHKERRQ(ierr);
	} else {
		ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help); CHKERRQ(ierr);
	}

	GridInfo gi;
	PetscBool flg;
	ierr = PetscOptionsGetString(PETSC_NULL, "-i", gi.input_name, PETSC_MAX_PATH_LEN-1, &flg); CHKERRQ(ierr);
	if (!flg) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, help); CHKERRQ(ierr);
		PetscFunctionReturn(0);
	}
	ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "fd3d launched.\n"); CHKERRQ(ierr);
	ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "input file: %s.h5\n", gi.input_name); CHKERRQ(ierr);

	/** Initialize the time stamp. */
	TimeStamp ts;
	ierr = initTimeStamp(&ts); CHKERRQ(ierr);

	/** Set up grid info. */
	const char *h5_ext = ".h5";
	ierr = PetscStrcpy(gi.inputfile_name, gi.input_name); CHKERRQ(ierr);
	ierr = PetscStrcat(gi.inputfile_name, h5_ext); CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL, "-o", gi.output_name, PETSC_MAX_PATH_LEN-1, &flg); CHKERRQ(ierr);
	if (!flg) {
		ierr = PetscStrcpy(gi.output_name, gi.input_name); CHKERRQ(ierr);
	}
	ierr = setGridInfo(&gi); CHKERRQ(ierr);
	if (gi.verbose_level >= VBMedium) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\n", gi.input_name); CHKERRQ(ierr);
	}
	ierr = updateTimeStamp(VBDetail, &ts, "grid info construction", gi); CHKERRQ(ierr);

	/** Set FD3D options in GridInfo. */
	ierr = setOptions(&gi); CHKERRQ(ierr);

	Mat A, CF;
	Vec b, right_precond, conjParam, conjSrc;

	/** Create A and b according to the options. */
	ierr = create_A_and_b4(&A, &b, &right_precond, &CF, &conjParam, &conjSrc, gi, &ts); CHKERRQ(ierr);

	/** Output A and b. */
	if (gi.output_mat_and_vec) {
		ierr = output_mat_and_vec(A, b, right_precond, CF, gi); CHKERRQ(ierr);
		ierr = cleanup(A, b, right_precond, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);

		PetscFunctionReturn(0);  // finish the program
	}

	/** Prepare the initial guess .*/
	/** eq to solve: 
	  diag(1/left_precond) A0 diag(1/right_precond) y = diag(1/left_precond) b
	  , where
	  x = diag(1/right_precond) y
	  y = diag(right_precond) x
	 */
	Vec x;  // actually y in the above equation
	if (gi.x0_type == GEN_GIVEN) {
		ierr = createVecPETSc(&x, "F0", gi); CHKERRQ(ierr);
		ierr = VecPointwiseMult(x, x, right_precond); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, &ts, "supplied x0 initialization", gi); CHKERRQ(ierr);
	} else {  // in this case, not preconditioning x is better
		ierr = VecDuplicate(gi.vecTemp, &x); CHKERRQ(ierr);
		if (gi.x0_type == GEN_ZERO) {
			ierr = VecSet(x, 0.0); CHKERRQ(ierr);
			ierr = updateTimeStamp(VBDetail, &ts, "zero x0 initialization", gi); CHKERRQ(ierr);
		} else {
			assert(gi.x0_type == GEN_RAND);
			ierr = VecSetRandom(x, PETSC_NULL); CHKERRQ(ierr);  // for random initial x
			ierr = updateTimeStamp(VBDetail, &ts, "random x0 initialization", gi); CHKERRQ(ierr);
		}
//		ierr = VecPointwiseDivide(x, x, right_precond); CHKERRQ(ierr);

		/** For initial condition trick */
/*
		ierr = VecCopy(b, x); CHKERRQ(ierr);
		ierr = VecScale(x, PETSC_i/gi.omega); CHKERRQ(ierr);
*/
	}
	//ierr = VecCopy(e0, x); CHKERRQ(ierr);
	//ierr = VecCopy(epsE0, x); CHKERRQ(ierr);
	//A = GD;
	//ierr = MatAXPY(A, -1.0, GD, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

	/** Create the normal equation. */
	/*
	   Mat C;
	   ierr = MatHermitianTranspose(B, MAT_INITIAL_MATRIX, &C); CHKERRQ(ierr);
	   ierr = MatMatMult(C, B, MAT_INITIAL_MATRIX, 1, &A); CHKERRQ(ierr);
	   ierr = MatSetOption(A, MAT_SYMMETRIC, PETSC_FALSE); CHKERRQ(ierr);
	   ierr = MatSetOption(A, MAT_HERMITIAN, PETSC_TRUE); CHKERRQ(ierr);
	   ierr = VecDuplicate(gi.vecTemp, &b); CHKERRQ(ierr);
	   ierr = MatMult(C, c, b); CHKERRQ(ierr);
	   ierr = MatDestroy(&C); CHKERRQ(ierr);
	   ierr = MatDestroy(&B); CHKERRQ(ierr);
	   ierr = VecDestroy(&c); CHKERRQ(ierr);
	 */

	/** Solve A x = b. */
	if (gi.use_ksp) {  // Use an algorithm in KSP.
		if (gi.verbose_level >= VBMedium) {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Iterative solver starts.\n"); CHKERRQ(ierr);
		}

		/** Create a KSP object. */
		KSP ksp;
		ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
		ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
		ierr = KSPSetTolerances(ksp, gi.tol, 0.0, PETSC_MAX_REAL, gi.max_iter); CHKERRQ(ierr);

		/** Allow a nonzero initial guess. */
		ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRQ(ierr);

		/** Set the iterative method. */
		PetscBool isSymmetric;
		ierr = MatIsSymmetric(A, 0.0, &isSymmetric); CHKERRQ(ierr);
		if (isSymmetric) {  // symmetric
			ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
			ierr = KSPCGSetType(ksp, KSP_CG_SYMMETRIC); CHKERRQ(ierr);
			if (gi.verbose_level >= VBMedium) {
				ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "default algorithm: KSP CG symmetric, "); CHKERRQ(ierr);
			}
		} else {
			PetscBool isHermitian;
			ierr = MatIsHermitian(A, 0.0, &isHermitian); CHKERRQ(ierr);
			if (isHermitian) {  // nonsymmetric but Hermitian
				ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
				ierr = KSPCGSetType(ksp, KSP_CG_HERMITIAN); CHKERRQ(ierr);
				if (gi.verbose_level >= VBMedium) {
					ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "default algorithm: KSP CG Hermitian, "); CHKERRQ(ierr);
				}
			} else {  // neither symmetric nor Hermitian
				ierr = KSPSetType(ksp, KSPBICG); CHKERRQ(ierr);
				if (gi.verbose_level >= VBMedium) {
					ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "default algorithm: KSP BICG, "); CHKERRQ(ierr);
				}
			}
		}

		/** Create a PC object. */
		PC pc;
		ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
		ierr = PCSetType(pc, PCNONE); CHKERRQ(ierr);
		ierr = KSPSetPC(ksp, pc); CHKERRQ(ierr);
		ierr = KSPSetNormType(ksp, KSP_NORM_PRECONDITIONED); CHKERRQ(ierr);

		/** Override any KSP settings (method, PC, monitor, etc) with options described in "option_file", if any. */ 
		ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
		const KSPType type;
		ierr = KSPGetType(ksp, &type); CHKERRQ(ierr);
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "actual algorithm: KSP %s\n", type); CHKERRQ(ierr);
		/** Set up the monitor of the iteration process. */
		ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);

		PetscReal norm_b;
		Vec b_precond;
		ierr = VecDuplicate(gi.vecTemp, &b_precond); CHKERRQ(ierr);
		ierr = PCApply(pc, b, b_precond); CHKERRQ(ierr);  // for KSP_NORM_PRECONDITIONED above
		ierr = VecNorm(b_precond, NORM_2, &norm_b); CHKERRQ(ierr);
		ierr = VecDestroy(&b_precond); CHKERRQ(ierr);
		ierr = KSPMonitorCancel(ksp); CHKERRQ(ierr);  // cancel the default monitor
		MonitorCtx monitor_ctx;
		monitor_ctx.norm_b = norm_b;
		monitor_ctx.gi = &gi;
		ierr = KSPMonitorSet(ksp, monitor_relres, &monitor_ctx, PETSC_NULL); CHKERRQ(ierr);  // compare with the default monitor function that shows the absolute residual norm

		if (gi.output_relres) {
			char relres_file_name[PETSC_MAX_PATH_LEN];
			ierr = PetscStrcpy(relres_file_name, gi.output_name); CHKERRQ(ierr);
			ierr = PetscStrcat(relres_file_name, ".res"); CHKERRQ(ierr);
			gi.relres_file = fopen(relres_file_name, "w");  // in a project directory
		}

		/** Solve the equation, and retrieve convergence information. */
		ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

		if (gi.output_relres) {
			fclose(gi.relres_file);
		}

		if (gi.verbose_level >= VBMedium) {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Iterative solver finished.\n"); CHKERRQ(ierr);
		}

		KSPConvergedReason reason;
		ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "converged reason: %d\n", reason); CHKERRQ(ierr);
/*
		PetscInt num_iter;
		PetscReal rel_res;

		ierr = KSPGetIterationNumber(ksp, &num_iter); CHKERRQ(ierr);
		ierr = KSPGetResidualNorm(ksp, &rel_res); CHKERRQ(ierr);
		rel_res /= norm_b;
*/

		/** Clean up KSP objects. */
		//ierr = DiagonalShellPCDestroy(&pc); CHKERRQ(ierr);
		ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
	} else {  // Use my algorithm.
		PetscBool calc_singular = PETSC_FALSE;
		if (calc_singular) {
			Mat Adag;
			ierr = MatHermitianTranspose(A, MAT_INITIAL_MATRIX, &Adag); CHKERRQ(ierr);
			//ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &Adag); CHKERRQ(ierr);

			Vec x1, x2, b1, b2;
			ierr = VecDuplicate(gi.vecTemp, &x1); CHKERRQ(ierr);
			ierr = VecDuplicate(gi.vecTemp, &x2); CHKERRQ(ierr);
			ierr = VecDuplicate(gi.vecTemp, &b1); CHKERRQ(ierr);
			ierr = VecDuplicate(gi.vecTemp, &b2); CHKERRQ(ierr);

			PetscReal smin_inv, smax, sprev;
			ierr = VecSetRandom(b1, PETSC_NULL); CHKERRQ(ierr);
			//ierr = VecCopy(b1, b2); CHKERRQ(ierr);
			ierr = VecSetRandom(b2, PETSC_NULL); CHKERRQ(ierr);
			ierr = vecNormalize(b1, b2, PETSC_NULL); CHKERRQ(ierr);
			sprev = 0.0;
			smax = 1.0;
			PetscInt nmax;
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Calculate smax.\n"); CHKERRQ(ierr);

			for (nmax = 1; PetscAbsScalar((smax-sprev)/smax) > 1e-11; ++nmax) {
				sprev = smax;

				ierr = multAandAdag(A, Adag, b1, b2, x1, x2); CHKERRQ(ierr);
				ierr = vecNormalize(x1, x2, &smax); CHKERRQ(ierr);
				ierr = VecCopy(x1, b1); CHKERRQ(ierr);
				ierr = VecCopy(x2, b2); CHKERRQ(ierr);
				ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\tsmax: %e\n", nmax, smax); CHKERRQ(ierr);
			}

			ierr = VecSetRandom(b1, PETSC_NULL); CHKERRQ(ierr);
			ierr = VecSetRandom(b2, PETSC_NULL); CHKERRQ(ierr);
			//ierr = VecCopy(b1, b2); CHKERRQ(ierr);
			ierr = vecNormalize(b1, b2, PETSC_NULL); CHKERRQ(ierr);
			ierr = VecSet(x1, 0.0); CHKERRQ(ierr);
			ierr = VecSet(x2, 0.0); CHKERRQ(ierr);
			sprev = 0.0;
			smin_inv = 1.0;
			PetscInt nmin;
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nCalculate smin.\n"); CHKERRQ(ierr);
			for (nmin = 1; PetscAbsScalar((smin_inv-sprev)/smin_inv) > 1e-11; ++nmin) {
				sprev = smin_inv;

				ierr = bicgAandAdag(A, Adag, x1, x2, b1, b2, right_precond, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);
				ierr = vecNormalize(x1, x2, &smin_inv); CHKERRQ(ierr);
				ierr = VecCopy(x1, b1); CHKERRQ(ierr);
				ierr = VecCopy(x2, b2); CHKERRQ(ierr);

				ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\tsmin: %e\n", nmin, 1.0/smin_inv); CHKERRQ(ierr);
			}

			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nnmax = %d, smax = %e, \tnmin = %d, smin = %e, \tcond = %e\n", nmax, smax, nmin, 1.0/smin_inv, smax*smin_inv); CHKERRQ(ierr);
		} else {
			if (gi.verbose_level >= VBMedium) {
				ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Iterative solver starts.\n"); CHKERRQ(ierr);
			}

			IterativeSolver solver;

			PetscBool isSymmetric;
			ierr = MatIsSymmetric(A, 0.0, &isSymmetric); CHKERRQ(ierr);
			if (gi.krylov_type == QMR) {
				if (isSymmetric) {
					solver = qmrSymmetric;
				} else {
					solver = qmr;
				}
			} else {
				if (isSymmetric) {
					solver = bicgSymmetric;
				} else {
					solver = bicg;
				}
			}

			if (gi.output_relres) {
				char relres_file_name[PETSC_MAX_PATH_LEN];
				ierr = PetscStrcpy(relres_file_name, gi.output_name); CHKERRQ(ierr);
				ierr = PetscStrcat(relres_file_name, ".res"); CHKERRQ(ierr);
				gi.relres_file = fopen(relres_file_name, "w");  // in a project directory
			}

			ierr = solver(A, x, b, right_precond, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);
			//ierr = solver(GD, x, b, right_precond, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);

			//ierr = bicg_component(x, gi, &ts); CHKERRQ(ierr);

			/** Iterative refinement using the initial condition trick. */
/*
			PetscInt i_outer;
			Vec x0, rk;
			ierr = VecDuplicate(gi.vecTemp, &x0); CHKERRQ(ierr);
			ierr = VecDuplicate(gi.vecTemp, &rk); CHKERRQ(ierr);
			ierr = VecZeroEntries(x); CHKERRQ(ierr);
			gi.tol = 1e-6;
			//gi.max_iter = 10;
			for (i_outer = 0; i_outer < 1; ++i_outer) {
				ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "i_outer = %d\n", i_outer); CHKERRQ(ierr);
				ierr = MatMult(A, x, rk); CHKERRQ(ierr);  // rk = A*x
				ierr = VecAXPY(rk, -(gi.omega*gi.omega), x); CHKERRQ(ierr);
				ierr = VecAYPX(rk, -1.0, b); CHKERRQ(ierr);  // rk = b - rk
				ierr = VecCopy(rk, x0); CHKERRQ(ierr);
				ierr = VecScale(x0, -1/(gi.omega*gi.omega)); CHKERRQ(ierr);
				ierr = solver(A, x0, rk, right_precond, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);  // now x0 is solution
				//ierr = solver(GD, x, b, right_precond, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);
				ierr = VecAXPY(x, 1.0, x0); CHKERRQ(ierr);  // x = x + x0
			}
*/

			if (gi.output_relres) {
				fclose(gi.relres_file);
			}
		}

		if (gi.verbose_level >= VBMedium) {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Iterative solver finished.\n"); CHKERRQ(ierr);
		}
	}

	/** Check the directly calculated residual norm, which is not a by-product of the 
	  iterative solver. */
	Vec r;  // residual vector for x
	ierr = VecDuplicate(x, &r); CHKERRQ(ierr);
	ierr = MatMult(A, x, r); CHKERRQ(ierr);
	ierr = VecAYPX(r, -1.0, b); CHKERRQ(ierr);  // r = b - A*x

	PetscReal norm_r, norm_b;
	ierr = VecNorm(r, NORM_INFINITY, &norm_r); CHKERRQ(ierr);
	ierr = VecNorm(b, NORM_INFINITY, &norm_b); CHKERRQ(ierr);
	PetscReal rel_res = norm_r / norm_b;

	if (gi.verbose_level >= VBMedium) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\ttrue relres: %e\n", rel_res); CHKERRQ(ierr);
	}

	/** Recover the solution for the original matrix A from the solution for the 
	  symmetrized matrix A. */
	ierr = VecPointwiseDivide(x, x, right_precond); CHKERRQ(ierr);

	/** Output the solution. */
	//ierr = VecAXPY(x, 1.0, e0); CHKERRQ(ierr);
	ierr = output(gi.output_name, x, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, &ts, "iterative solver", gi); CHKERRQ(ierr);

	/** Clean up. */
	ierr = VecDestroy(&x); CHKERRQ(ierr);
	ierr = cleanup(A, b, right_precond, CF, conjParam, conjSrc, gi); CHKERRQ(ierr);

	PetscFunctionReturn(0);  // finish the program
}
