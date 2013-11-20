#ifndef GUARD_type_h
#define GUARD_type_h

typedef enum {VBCompact, VBMedium, VBDetail} VerboseLevel;
typedef enum {Xx, Yy, Zz, Naxis} Axis;
typedef enum {Neg, Pos, Nsign} Sign;
typedef enum {Prim, Dual, Ngt} GridType;
typedef enum {Etype, Htype, Nft} FieldType;
typedef enum {GEN_ZERO, GEN_RAND, GEN_GIVEN} F0Type;
typedef enum {PEC, PMC, Bloch} BC;
typedef enum {SCPML, UPML} PMLType;
typedef enum {PCIdentity, PCSfactor, PCParam, PCJacobi} PrecondType;
typedef enum {BiCG, QMR} KrylovType;
typedef enum {Rr, Ii, Nri} RI;

#endif
