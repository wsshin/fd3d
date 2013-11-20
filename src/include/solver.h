#ifndef GUARD_solver_h
#define GUARD_solver_h

#include "gridinfo.h"
#include "output.h"
#include "logging.h"

#include "petsc.h"

typedef PetscErrorCode (*IterativeSolver)(const Mat A, Vec x, const Vec b, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode bicgSymmetric(const Mat A, Vec x, const Vec b, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode cgs(const Mat A, Vec x, const Vec b, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode bicg(const Mat A, Vec x, const Vec b, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode qmr(const Mat A, Vec x, const Vec b, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode qmrSymmetric(const Mat A, Vec x, const Vec b, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode cgAandAdag(const Mat A, const Mat Adag, Vec x1, Vec x2, const Vec b1, const Vec b2, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode bicgAandAdag(const Mat A, const Mat Adag, Vec x1, Vec x2, const Vec b1, const Vec b2, const Vec right_precond, const Mat HE, const Vec conjPararm, const Vec conjSrc, GridInfo gi);

PetscErrorCode vecNormalize(Vec x1, Vec x2, PetscReal *val);

PetscErrorCode multAandAdag(const Mat A, const Mat Adag, const Vec x1, const Vec x2, Vec y1, Vec y2);

PetscErrorCode bicg_component(Vec x, GridInfo gi, TimeStamp *ts);

#endif
