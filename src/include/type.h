#ifndef GUARD_type_h
#define GUARD_type_h

typedef enum {VBCompact, VBMedium, VBDetail} VerboseLevel;
typedef enum {Xx, Yy, Zz, Naxis} Axis;
typedef enum {Neg, Pos, Nsign} Sign;
typedef enum {Etype, Htype} FieldType;
typedef enum {PEC, PMC, Bloch} BC;
typedef enum {SCPML, UPML} PMLType;
typedef enum {PCIdentity, PCSparam, PCEps, PCJacobi} PrecondType;
typedef enum {BiCG, QMR} KrylovType;
typedef enum {Rr, Ii, Nri} RI;

#endif
