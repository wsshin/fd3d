#ifndef GUARD_mat_h
#define GUARD_mat_h

#include "gridinfo.h"
#include "type.h"
#include "logging.h"
#include "vec.h"

#include "petsc.h"

#define MATRIX_TYPE MATAIJ
#define MATRIX_SYM_TYPE MATSBAIJ
//#define MATRIX_TYPE MATMPIAIJ
//#define MATRIX_SYM_TYPE MATMPISBAIJ
//#define MATRIX_TYPE MATSEQAIJ
//#define MATRIX_SYM_TYPE MATSEQSBAIJ


/**
 * setDpOnDivF_at
 * ------------
 * Take the div(F) operator matrix DivF, and set up the elements for d/d(p) on it, where 
 * F = E, H, and p = x, y, z, at a given location coord[].
 */
PetscErrorCode setDpOnDivF_at(Mat DivF, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi);

/**
 * setDivF
 * -----
 * Set up the div(F) operator matrix DivF.  
 * DivF is an N x 3N matrix expanded to 3N x 3N.
 */
PetscErrorCode setDivF(Mat DivF, FieldType ftype, GridInfo gi);

/**
 * createDivE
 * --------
 * Create the matrix DivE, the divergence operator on E fields.
 */
PetscErrorCode createDivE(Mat *DivE, GridInfo gi);

/**
 * setDpOnFGrad_at
 * ------------
 * Take the F = grad(phi) operator matrix FGrad, and set up the elements for d/d(p) on it, where 
 * F = E, H, and p = x, y, z, at a given location coord[].
 */
PetscErrorCode setDpOnFGrad_at(Mat FGrad, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi);

/**
 * setFGrad
 * -----
 * Set up the F = grad(phi) operator matrix FGrad.  
 * FGrad is a 3N x N matrix expanded to 3N x 3N.
 */
PetscErrorCode setFGrad(Mat FGrad, FieldType ftype, GridInfo gi);

/**
 * createEGrad
 * --------
 * Create the matrix EGrad, the gradient operator generating E fields.
 */
PetscErrorCode createEGrad(Mat *EGrad, GridInfo gi);

/**
 * setDpOnCF_at
 * ------------
 * Take the curl(F) operator matrix CF for given F == E or H, and set up the elements
 * for d/d(p) on it, where p = x, y, z, at a given location coord[].
 */
PetscErrorCode setDpOnCF_at(Mat CF, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi, PetscBool assumeField);

/**
 * setCF
 * -----
 * Set up the curl(F) operator matrix CF for given F == E or H.
 */
PetscErrorCode setCF(Mat CF, FieldType ftype, GridInfo gi, PetscBool useBC);

/**
 * createCH
 * --------
 * Create the matrix CH, the curl operator on H fields.
 */
PetscErrorCode createCH(Mat *CH, GridInfo gi, PetscBool useBC);

/**
 * createCE
 * --------
 * Create the matrix CE, the curl operator on E fields.
 */
PetscErrorCode createCE(Mat *CE, GridInfo gi, PetscBool useBC);

/**
 * createCHE
 * -------
 * Create the matrix CHE, the curl(mu^-1 curl) operator on E fields.
 */
PetscErrorCode createCHE(Mat *CHE, Mat CH, Mat HE, GridInfo gi);

/**
 * setAtemplate_at
 * ------------
 * Set the elements of Atemplate zeros.
 */
PetscErrorCode setAtemplate_at(Mat Atemplate, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi);

/**
 * setAtemplate
 * -----
 * Set up Atemplate matrix that has zeros preset for the nonzero elements of A.
 */
PetscErrorCode setAtemplate(Mat Atemplate, FieldType ftype, GridInfo gi);

/**
 * createAtemplate
 * --------
 * Create the template matrix of A.  It has an appropriate nonzero element pattern predetermined.
 */
PetscErrorCode createAtemplate(Mat *Atemplate, GridInfo gi);

/**
 * createGD
 * -------
 * Create the matrix GD, the grad(eps^-1 div) operator on E fields.
 */
PetscErrorCode createGD(Mat *GD, GridInfo gi);

/**
 * createDG
 * -------
 * Create the matrix DG, the div(eps grad) operator on a scalar potential.
 */
PetscErrorCode createDG(Mat *DG, GridInfo gi);

/**
 * hasPMC
 * ------
 * Set flgPMC PETSC_TRUE if some boundary is PMC; PETSC_FALSE otherwise.
 */
PetscErrorCode hasPMC(PetscBool *flgPMC, GridInfo gi);

/**
 * hasBloch
 * ------
 * Set flgBloch PETSC_TRUE if some boundary is Bloch and associated k_Bloch is nonzero; 
 * PETSC_FALSE otherwise.
 */
PetscErrorCode hasBloch(PetscBool *flgBloch, GridInfo gi);

/**
 * numSymmetrize
 * ------
 * Numerically symmetrize a given matrix.  It also evaluate the relative error between
 * A and A^T.
 */
PetscErrorCode numSymmetrize(Mat A);

PetscErrorCode stretch_d(GridInfo gi);

PetscErrorCode unstretch_d(GridInfo gi);

PetscErrorCode create_A_and_b(Mat *A, Vec *b, Vec *right_precond, Mat *HE, GridInfo gi, TimeStamp *ts);

#endif
