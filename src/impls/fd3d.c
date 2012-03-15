#include "Python.h"  // this line should be included at the very first 

#include "gridinfo.h"
#include "type.h"
#include "logging.h"
#include "vec.h"
#include "mat.h"
#include "solver.h"
#include "output.h"

#include "petsc.h"

#if USE_SLEPC!=0
#include "slepceps.h"
#include "slepcsvd.h"
#endif

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
PetscErrorCode cleanup(Mat A, Vec b, Vec right_precond, Mat HE, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Destroy matrices and vectotrs used in BiCG. */
	ierr = MatDestroy(&A); CHKERRQ(ierr);  // destroy A == CEH
	ierr = VecDestroy(&b); CHKERRQ(ierr);
	ierr = VecDestroy(&right_precond); CHKERRQ(ierr);
	ierr = MatDestroy(&HE); CHKERRQ(ierr);  // destroy EH == CH

	/** Finalize the program. */
	Py_DECREF(gi.pSim);
	ierr = DMDestroy(&gi.da); CHKERRQ(ierr);
	ierr = VecDestroy(&gi.vecTemp); CHKERRQ(ierr);

	// Need to move this to gridinfo.c
	PetscInt axis;
	for (axis = 0; axis < Naxis; ++axis) {
		ierr = PetscFree6(gi.d_prim[axis], gi.d_dual[axis], gi.s_prim[axis], gi.s_dual[axis], gi.d_prim_orig[axis], gi.d_dual_orig[axis]); CHKERRQ(ierr);
	}
	//ISLocalToGlobalMappingDestroy(gi.map); CHKERRQ(ierr);

	/** Print the summary of the program execution. */
	if (gi.verbose_level >= VBMedium) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "fd3d finished.\n"); CHKERRQ(ierr);
	}

	Py_Finalize();
#if USE_SLEPC==0
	ierr = PetscFinalize(); CHKERRQ(ierr);  // finalize PETSc.
#else
	ierr = SlepcFinalize(); CHKERRQ(ierr);  // finalize SLEPc.
#endif

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
		//strcpy(option_file_name, getenv("FD3D_ROOT"));
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
#if USE_SLEPC==0
		ierr = PetscInitialize(&argc, &argv, option_file_name, help); CHKERRQ(ierr);
#else
		ierr = SlepcInitialize(&argc, &argv, option_file_name, help); CHKERRQ(ierr);
#endif
	} else {
#if USE_SLEPC==0
		ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help); CHKERRQ(ierr);
#else
		ierr = SlepcInitialize(&argc, &argv, PETSC_NULL, help); CHKERRQ(ierr);
#endif
	}

	char input_name[PETSC_MAX_PATH_LEN];
	PetscBool flg;
	ierr = PetscOptionsGetString(PETSC_NULL, "-i", input_name, PETSC_MAX_PATH_LEN-1, &flg); CHKERRQ(ierr);
	if (!flg) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, help); CHKERRQ(ierr);
		PetscFunctionReturn(0);
	}

	//ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "fd3d launched.\n"); CHKERRQ(ierr);

	/** Initialize Python interpreter. */
	Py_Initialize(); if (PyErr_Occurred()) PyErr_Print();

	/** Add the present working directory to the module search path of the Python interpreter. */
	PyRun_SimpleString("import sys"); if (PyErr_Occurred()) PyErr_Print();
	//PyRun_SimpleString("sys.path.append('/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages')");
	//PyRun_SimpleString("from scipy.io import loadmat");
	PyRun_SimpleString("import os"); if (PyErr_Occurred()) PyErr_Print();

	/** By some reason, the order of insertion below is important; /in should be inserted later
	  than /bin, i.e. /in should come earlier in the path than /bin. */
	PyRun_SimpleString("sys.path.insert(0, os.path.expandvars('$FD3D_ROOT')+'/bin')"); if (PyErr_Occurred()) PyErr_Print();
	//PyRun_SimpleString("sys.path.insert(0, os.path.expandvars('$FD3D_ROOT')+'/in')");
	//PyRun_SimpleString("sys.path.insert(0, '.')");
	PyRun_SimpleString("sys.path.insert(0, '.')"); if (PyErr_Occurred()) PyErr_Print();  // for the project directory where the input is
	//PyRun_SimpleString("sys.path.insert(0, '../in')");

	/** Force regeneration of .pyc files. */
	/*
	   PyRun_SimpleString("import compileall");
	   PyRun_SimpleString("import re");
	   PyRun_SimpleString("compileall.compile_dir('./', rx=re.compile('/[.]svn'), force=True, quiet=True)");
	   char compile_input[PETSC_MAX_PATH_LEN];
	   PyRun_SimpleString("import py_compile");
	   ierr = PetscStrcpy(compile_input, "py_compile.compile('../in/"); CHKERRQ(ierr);
	   ierr = PetscStrcat(compile_input, input_name); CHKERRQ(ierr);
	   ierr = PetscStrcat(compile_input, ".py')"); CHKERRQ(ierr);
	   PyRun_SimpleString(compile_input);
	 */
	//PyRun_SimpleString("import os");
	//PyRun_SimpleString("os.system('rm ../in/*.pyc')");
	//PyRun_SimpleString("os.system('rm *.pyc')");

	/** Initialize the time stamp. */
	TimeStamp ts;
	ierr = initTimeStamp(&ts); CHKERRQ(ierr);

	/** Set up grid info. */
	GridInfo gi;
	ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "input file: %s.py", input_name); CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL, "-o", gi.output_name, PETSC_MAX_PATH_LEN-1, &flg); CHKERRQ(ierr);
	if (!flg) {
		ierr = PetscStrcpy(gi.output_name, input_name); CHKERRQ(ierr);
	}
	ierr = setGridInfo(&gi, input_name); CHKERRQ(ierr);
	if (gi.verbose_level >= VBMedium) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\n", input_name); CHKERRQ(ierr);
	}
	ierr = updateTimeStamp(VBDetail, &ts, "grid info construction", gi); CHKERRQ(ierr);

	/** Set FD3D options in GridInfo. */
	ierr = setOptions(&gi); CHKERRQ(ierr);

	Mat A, HE;
	Vec b, right_precond;

	/** Create A and b according to the options. */
	//ierr = create_A_and_b(&A, &b, &right_precond, &HE, gi, &ts); CHKERRQ(ierr);
	//ierr = create_A_and_b2(&A, &b, &right_precond, &HE, gi, &ts); CHKERRQ(ierr);
	//ierr = create_A_and_b3(&A, &b, &right_precond, &HE, gi, &ts); CHKERRQ(ierr);
	ierr = create_A_and_b4(&A, &b, &right_precond, &HE, gi, &ts); CHKERRQ(ierr);

	/*
	   if (gi.pml_type == SCPML) {
	   ierr = stretch_d(gi); CHKERRQ(ierr);
	   }
	   Mat EGrad, B;
	   ierr = createEGrad(&EGrad, gi); CHKERRQ(ierr);
	   if (gi.pml_type == SCPML) {
	   ierr = unstretch_d(gi); CHKERRQ(ierr);
	   }
	   ierr = MatMatMult(A, EGrad, MAT_INITIAL_MATRIX, 26.0/(13.0+2.0), &B); CHKERRQ(ierr); // GD = EGrad*invEpsNode*DivE

	   PetscReal normB;
	   ierr = MatNorm(B, NORM_INFINITY, &normB); CHKERRQ(ierr);
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nnorm(A EGrad) = %e\n", normB); CHKERRQ(ierr);
	 */

	/** TF/SF */
	/** TF/SF is currently supported only by the SC-PML. */
	if (gi.has_xinc) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nCreate the TF/SF source.\n"); CHKERRQ(ierr);
		Mat A_bg;
		gi.bg_only = PETSC_TRUE;
		ierr = create_A_and_b(&A_bg, &b, &right_precond, &HE, gi, &ts); CHKERRQ(ierr);
		Vec xInc;
		//ierr = create_xInc(&xInc, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&xInc, set_x_inc_at, gi); CHKERRQ(ierr);
		ierr = MatMult(A_bg, xInc, b); CHKERRQ(ierr);

		ierr = MatDestroy(&A_bg); CHKERRQ(ierr);
		ierr = VecDestroy(&xInc); CHKERRQ(ierr);
		gi.bg_only = PETSC_FALSE;
		ierr = updateTimeStamp(VBDetail, &ts, "TF/SF source", gi); CHKERRQ(ierr);
	}

	/** Output A and b. */
	if (gi.output_mat_and_vec) {
		ierr = output_mat_and_vec(A, b, right_precond, HE, gi); CHKERRQ(ierr);
		ierr = cleanup(A, b, right_precond, HE, gi); CHKERRQ(ierr);

		PetscFunctionReturn(0);  // finish the program
	}

#if USE_SLEPC!=0
	/** Solve the eigen problem A x = lambda x. */
	if (gi.solve_eigen) {
		EPS eps;
		const EPSType type;
		PetscReal tol;
		PetscInt n = 1, nev, its, nconv, maxit;
		PetscScalar ev, ev_img;  // eigenvalue
		Vec x, x_img;  // eigenvector

		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nEigen solver starts.\n"); CHKERRQ(ierr);

		/** Create the eigensolver context. */
		ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);

		/** Set operators for the standard eigenvalue problem. */
		ierr = EPSSetOperators(eps, A, PETSC_NULL); CHKERRQ(ierr);
		ierr = EPSSetProblemType(eps, EPS_NHEP); CHKERRQ(ierr);

		/** Set solver parameters at runtime. */
		ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);

		/** Solve the eigen problem. */
		ierr = EPSSolve(eps); CHKERRQ(ierr);

		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Eigen solver finished.\n"); CHKERRQ(ierr);

		/** Gather information and display it. */
		ierr = EPSGetIterationNumber(eps, &its); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of iterations of the method: %d\n", its); CHKERRQ(ierr);
		ierr = EPSGetType(eps, &type); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution method: %s\n", type); CHKERRQ(ierr);
		ierr = EPSGetDimensions(eps, &nev, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of requested eigenvalues: %d\n", nev); CHKERRQ(ierr);
		ierr = EPSGetTolerances(eps, &tol, &maxit); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Stopping condition: tol=%.4g, maxit=%d\n", tol, maxit); CHKERRQ(ierr);


		/** Display the solution. */
		ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of converged eigenpairs: %d\n", nconv); CHKERRQ(ierr);
		if (nconv > 0) {
			/** Display eigenvalues and relative errors. */
			PetscReal error;
			PetscInt i;
			char output_name_eigen[PETSC_MAX_PATH_LEN];

			ierr = VecDuplicate(gi.vecTemp, &x); CHKERRQ(ierr);
			ierr = VecDuplicate(gi.vecTemp, &x_img); CHKERRQ(ierr);

			ierr = PetscPrintf(PETSC_COMM_WORLD,
					"        k          ||Ax-kx||/||kx||\n" 
					"----------------- ------------------\n" ); CHKERRQ(ierr);
			for (i = 0; i < nconv; i++) {
				ierr = EPSGetEigenpair(eps, i, &ev, &ev_img, x, x_img); CHKERRQ(ierr);
				ierr = EPSComputeRelativeError(eps, i, &error); CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD," %f%+fi %12e\n", PetscRealPart(ev), PetscImaginaryPart(ev), error); CHKERRQ(ierr);

				PetscSNPrintf(output_name_eigen, PETSC_MAX_PATH_LEN, "%s_eigen%d", gi.output_name, i); CHKERRQ(ierr);
				ierr = output(output_name_eigen, gi.x_type, x, HE); CHKERRQ(ierr);
			}
			ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
		}

		/** Clean up. */
		ierr = EPSDestroy(&eps); CHKERRQ(ierr); 
		ierr = VecDestroy(&x); CHKERRQ(ierr); 
		ierr = VecDestroy(&x_img); CHKERRQ(ierr); 
		ierr = cleanup(A, b, right_precond, HE, gi); CHKERRQ(ierr);

		PetscFunctionReturn(0);  // finish the program
	}

	/** Carry out partial singular value decomposition. */
	if (gi.solve_singular) {
		SVD svd;
		const SVDType type;
		PetscReal tol;
		PetscInt n = 1, nsv, its, nconv, maxit;
		PetscReal sv;  // singular value
		Vec u, v;  // left, right singular vectors

		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nPartial SVD starts.\n"); CHKERRQ(ierr);

		/** Create the SVD context. */
		ierr = SVDCreate(PETSC_COMM_WORLD, &svd); CHKERRQ(ierr);

		/** Set the operator for SVD. */
		ierr = SVDSetOperator(svd, A); CHKERRQ(ierr);

		/** Set solver parameters at runtime. */
		ierr = SVDSetFromOptions(svd); CHKERRQ(ierr);

		/** Carry out SVD. */
		ierr = SVDSolve(svd); CHKERRQ(ierr);

		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Partial SVD finished.\n"); CHKERRQ(ierr);

		/** Gather information and display it. */
		ierr = SVDGetIterationNumber(svd, &its); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of iterations of the method: %d\n", its); CHKERRQ(ierr);
		ierr = SVDGetType(svd, &type); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Solution method: %s\n", type); CHKERRQ(ierr);
		ierr = SVDGetDimensions(svd, &nsv, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of requested singular values: %d\n", nsv); CHKERRQ(ierr);
		ierr = SVDGetTolerances(svd, &tol, &maxit); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Stopping condition: tol=%.4g, maxit=%d\n", tol, maxit); CHKERRQ(ierr);


		/** Display the solution. */
		ierr = SVDGetConverged(svd, &nconv); CHKERRQ(ierr);
		ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of converged singular triplets: %d\n", nconv); CHKERRQ(ierr);
		if (nconv > 0) {
			/** Display eigenvalues and relative errors. */
			PetscReal error;
			PetscInt i;
			char output_name_singular[PETSC_MAX_PATH_LEN];

			ierr = VecDuplicate(gi.vecTemp, &u); CHKERRQ(ierr);
			ierr = VecDuplicate(gi.vecTemp, &v); CHKERRQ(ierr);

			ierr = PetscPrintf(PETSC_COMM_WORLD,
					"       k       ||H(A)x-kx||/||kx||\n" 
					"-------------- -------------------\n" ); CHKERRQ(ierr);
			for (i = 0; i < nconv; i++) {
				ierr = SVDGetSingularTriplet(svd, i, &sv, u, v); CHKERRQ(ierr);
				ierr = SVDComputeRelativeError(svd, i, &error); CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD," %e      %12e\n", sv, error); CHKERRQ(ierr);

				PetscSNPrintf(output_name_singular, PETSC_MAX_PATH_LEN, "%s_singular%d", gi.output_name, i); CHKERRQ(ierr);
				ierr = output_singular(output_name_singular, u, v); CHKERRQ(ierr);
			}
			ierr = PetscPrintf(PETSC_COMM_WORLD, "\n"); CHKERRQ(ierr);
		}

		/** Clean up. */
		ierr = SVDDestroy(&svd); CHKERRQ(ierr); 
		ierr = VecDestroy(&u); CHKERRQ(ierr); 
		ierr = VecDestroy(&v); CHKERRQ(ierr); 
		ierr = cleanup(A, b, right_precond, HE, gi); CHKERRQ(ierr);

		PetscFunctionReturn(0);  // finish the program
	}
#endif

	/** Prepare the initial guess .*/
	/** eq to solve: 
	  diag(1/left_precond) A0 diag(1/right_precond) y = diag(1/left_precond) b
	  , where
	  x = diag(1/right_precond) y
	  y = diag(right_precond) x
	 */
	Vec x;  // actually y in the above equation
	ierr = VecDuplicate(gi.vecTemp, &x); CHKERRQ(ierr);
	if (gi.has_x0) {
		ierr = VecCopy(gi.x0, x); CHKERRQ(ierr);
		ierr = VecPointwiseMult(x, x, right_precond); CHKERRQ(ierr);
	} else {  // in this case, not preconditioning x is better
		ierr = VecSet(x, 0.0); CHKERRQ(ierr);
		//ierr = VecSetRandom(x, PETSC_NULL); CHKERRQ(ierr);  // for random initial x
		//ierr = VecPointwiseDivide(x, x, right_precond); CHKERRQ(ierr);
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
		PetscInt num_iter;
		PetscReal rel_res;

		ierr = KSPGetIterationNumber(ksp, &num_iter); CHKERRQ(ierr);
		ierr = KSPGetResidualNorm(ksp, &rel_res); CHKERRQ(ierr);
		rel_res /= norm_b;

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

				ierr = bicgAandAdag(A, Adag, x1, x2, b1, b2, right_precond, HE, gi); CHKERRQ(ierr);
				ierr = vecNormalize(x1, x2, &smin_inv); CHKERRQ(ierr);
				ierr = VecCopy(x1, b1); CHKERRQ(ierr);
				ierr = VecCopy(x2, b2); CHKERRQ(ierr);

				ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\tsmin: %e\n", nmin, 1.0/smin_inv); CHKERRQ(ierr);
			}

			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nnmax = %d, smax = %e, \tnmin = %d, smin = %e, \tcond = %e\n", nmax, smax, nmin, 1.0/smin_inv, smax*smin_inv); CHKERRQ(ierr);
		} else {
			/*
			   if (calc_singular) {
			   PetscReal smin_inv, smax, sprev;
			   ierr = VecSetRandom(b, PETSC_NULL); CHKERRQ(ierr);
			//ierr = VecSet(b, 1.0*PETSC_i); CHKERRQ(ierr);
			ierr = VecNormalize(b, PETSC_NULL); CHKERRQ(ierr);
			sprev = 0.0;
			smax = 1.0;
			PetscInt nmax;
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Calculate smax.\n"); CHKERRQ(ierr);

			for (nmax = 1; PetscAbsScalar((smax-sprev)/smax) > 1e-11; ++nmax) {
			sprev = smax;

			ierr = MatMult(A, b, x); CHKERRQ(ierr);
			ierr = VecNormalize(x, &smax); CHKERRQ(ierr);
			ierr = VecCopy(x, b); CHKERRQ(ierr);

			smax = PetscSqrtScalar(smax);
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\tsmax: %e\n", nmax, smax); CHKERRQ(ierr);
			}

			ierr = VecSetRandom(b, PETSC_NULL); CHKERRQ(ierr);
			ierr = VecNormalize(b, PETSC_NULL); CHKERRQ(ierr);
			ierr = VecSet(x, 0.0); CHKERRQ(ierr);
			sprev = 0.0;
			smin_inv = 1.0;
			PetscInt nmin;
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nCalculate smin.\n"); CHKERRQ(ierr);
			for (nmin = 1; PetscAbsScalar((smin_inv-sprev)/smin_inv) > 1e-11; ++nmin) {
			sprev = smin_inv;

			ierr = bicg(A, x, b, right_precond, HE, gi); CHKERRQ(ierr);
			ierr = VecNormalize(x, &smin_inv); CHKERRQ(ierr);
			ierr = VecCopy(x, b); CHKERRQ(ierr);

			smin_inv = PetscSqrtScalar(smin_inv);
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\t%d\t\tsmin: %e\n", nmin, 1.0/smin_inv); CHKERRQ(ierr);
			}

			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nnmax = %d, smax = %e, \tnmin = %d, smin = %e, \tcond = %e\n", nmax, smax, nmin, 1.0/smin_inv, smax*smin_inv); CHKERRQ(ierr);
			} else {
			 */
			if (gi.verbose_level >= VBMedium) {
				ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Iterative solver starts.\n"); CHKERRQ(ierr);
			}

		IterativeSolver solver;
		/*
		   PetscBool flgBloch;
		   ierr = hasBloch(&flgBloch, gi); CHKERRQ(ierr);
		   if (!flgBloch) solver = bicgSymmetric;
		   else solver = bicg;
		 */

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

		/*
		   gi.tol = 1.0e-6;
		   ierr = solver(A, x, b, right_precond, HE, gi); CHKERRQ(ierr);
		   ierr = VecPointwiseDivide(x, x, right_precond); CHKERRQ(ierr);
		   ierr = VecAXPY(x, 1.0, e0); CHKERRQ(ierr);

		   gi.tol = 1.0e-6;
		   ierr = MatDestroy(&A); CHKERRQ(ierr);
		   ierr = MatDestroy(&HE); CHKERRQ(ierr);
		   ierr = VecDestroy(&b); CHKERRQ(ierr);
		   ierr = VecDestroy(&right_precond); CHKERRQ(ierr);

		   ierr = create_stretched_A(&A, &b, &right_precond, &HE, gi, &ts); CHKERRQ(ierr);
		 */
		if (gi.output_relres) {
			char relres_file_name[PETSC_MAX_PATH_LEN];
			ierr = PetscStrcpy(relres_file_name, gi.output_name); CHKERRQ(ierr);
			ierr = PetscStrcat(relres_file_name, ".res"); CHKERRQ(ierr);
			gi.relres_file = fopen(relres_file_name, "w");  // in a project directory
		}

		ierr = solver(A, x, b, right_precond, HE, gi); CHKERRQ(ierr);
		//ierr = solver(GD, x, b, right_precond, HE, gi); CHKERRQ(ierr);

		if (gi.output_relres) {
			fclose(gi.relres_file);
		}
		//}


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
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Iterative solver finished.\n"); CHKERRQ(ierr);
		}

		/** Recover the solution for the original matrix A from the solution for the 
		  symmetrized matrix A. */
		ierr = VecPointwiseDivide(x, x, right_precond); CHKERRQ(ierr);

		/** Output the solution. */
		//ierr = VecAXPY(x, 1.0, e0); CHKERRQ(ierr);
		ierr = output(gi.output_name, gi.x_type, x, HE); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, &ts, "iterative solver", gi); CHKERRQ(ierr);

		/** Clean up. */
		ierr = VecDestroy(&x); CHKERRQ(ierr);
		ierr = cleanup(A, b, right_precond, HE, gi); CHKERRQ(ierr);

		PetscFunctionReturn(0);  // finish the program
	}

	PetscFunctionReturn(0);  // finish the program
}
