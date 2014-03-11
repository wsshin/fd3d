#ifndef GUARD_type_h
#define GUARD_type_h

typedef enum {Neg, Pos, Nsign} Sign;
typedef enum {PEC, PMC, Bloch} BC;

typedef enum {Rr, Ii, Nri} RI;

typedef enum {Xx, Yy, Zz, Naxis} Axis;
static const char * const AxisName[] = {"x", "y", "z"};

typedef enum {VB_COMPACT, VB_MEDIUM, VB_DETAIL} VBType;  // verbose level
static const char * const VBTypes[] = {"compact", "medium", "detail", "VBType", "VB_", 0};

typedef enum {GRID_PRIMARY, GRID_DUAL, Ngrid} GridType;
static const char * const GridTypes[] = {"primary", "dual", "GridType", "GRID_", 0};

typedef enum {CELL_SD, CELL_D, CELL_S, CELL_1} CellType;
static const char * const CellTypes[] = {"sd", "d", "s", "1", "CellType", "CELL_", 0};

typedef enum {FIELD_E, FIELD_H} FieldType;
static const char * const FieldTypes[] = {"E", "H", "FieldType", "FIELD_", 0};

typedef enum {SYM_NON, SYM_1, SYM_AL, SYM_L, SYM_A, SYM_SQRTAL} SymType;  // right preconditioners
static const char * const SymTypes[] = {"non", "1", "al", "l", "a", "sqrtal", "SymType", "SYM_", 0};

typedef enum {PRECOND_1, PRECOND_JACOBI, PRECOND_S, PRECOND_MATPARAM} PrecondType;
static const char * const PrecondTypes[] = {"1", "jacobi", "s", "matparam", "PrecondType", "PRECOND_", 0};

typedef enum {X0_ZERO, X0_RANDOM, X0_GIVEN} X0Type;
static const char * const X0Types[] = {"zero", "random", "given", "X0Type", "X0_", 0};

typedef enum {KRYLOV_BICG, KRYLOV_QMR} KrylovType;
static const char * const KrylovTypes[] = {"BiCG", "QMR", "KrylovType", "KRYLOV_", 0};

#endif
