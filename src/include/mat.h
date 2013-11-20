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

PetscErrorCode create_A_and_b4(Mat *A, Vec *b, Vec *right_precond, Mat *HE, Vec *conjParam, Vec *conjSrc, GridInfo gi, TimeStamp *ts);

PetscErrorCode createDivE(Mat *DivE, GridInfo gi);

PetscErrorCode stretch_d(GridInfo *gi);

PetscErrorCode unstretch_d(GridInfo *gi);

PetscErrorCode createCH(Mat *CH, GridInfo gi);

PetscErrorCode createCE(Mat *CE, GridInfo gi);


#endif
