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

PetscErrorCode create_A_and_b4(Mat *A, Vec *b, Vec *right_precond, Mat *HE, GridInfo gi, TimeStamp *ts);

#endif
