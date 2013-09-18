#include <assert.h>
#include "mat.h"

const char * const FieldTypeName[] = {"E", "H"};
const char * const PMLTypeName[] = {"SC-PML", "UPML"};
const char * const PCTypeName[] = {"identity", "s-factor", "eps", "Jacobi"};

//const char * const KrylovTypeName[] = {"BiCG", "QMR"};

#undef __FUNCT__
#define __FUNCT__ "setDpOnDivF_at"
/**
 * setDpOnDivF_at
 * ------------
 * Take the div(F) operator matrix DivF, and set up the elements for d/d(p) on it, where 
 * F = E, H, and p = x, y, z, at a given location coord[].
 */
PetscErrorCode setDpOnDivF_at(Mat DivF, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** In theory DivF is an N x 3N matrix, but it is much easier to make it a 3N x 3N square
	  matrix with the distributted array (DA) of PETSc.  To that end we leave every 2nd and 3rd rows
	  empty. 
	  In other words, even though div(F) is a scalar, we make it a vector quantity such that 
	  [div(F)]x = div(F)
	  [div(F)]y = 0
	  [div(F)]z = 0
	 */

	/** For general ftype and Pp, I'm going to set up (d/dp) operation of the following equation:
	      [div(F)]x = (d/dp)Fp + (d/dq)Fq + (d/dr)Fr
	  If ftype==Etype and Pp==Zz, this means that I'm going to set up (d/dz) operations of:
	      [div(E)]x = (d/dx)Ex + (d/dy)Ey + (d/dz)Ez
	  For the future notation I define a vector Gx= [div(F)]x.  Then the equation is
	      Gx = (d/dp)Fp + (d/dq)Fq + (d/dr)Fr
	 */

	/** For (d/dp) terms in Gx */
	MatStencil indGx;  // grid point indices of Gx (mapped to row index of DivF)
	MatStencil indFp[2];  // current and next grid point indices of Fp (mapped to column indices of DivF).  The next grid point can be either in +p dir or in -p dir according to ftype.

	/** Determine Qq and Rr from the given Pp. */
	Axis Qq = (Axis)((Pp+1) % Naxis);  // if Pp==Xx, this is Yy    
	Axis Rr = (Axis)((Pp+2) % Naxis);  // if Pp==Xx, this is Zz

	/** Set MatStencil.c, which is the degree-of-freedom (dof) indices of PETSc.  In FD3D this is 
	  used to indicate the direction of the field component. */
	indGx.c = Xx; 
	indFp[0].c = Pp; indFp[1].c = Pp;

	/** Set Np. */
	PetscInt Np = gi.N[Pp];

	/** Set (x,y,z) indices for the current grid point. */
	indGx.i = i; indGx.j = j; indGx.k = k;
	indFp[0].i = i; indFp[0].j = j; indFp[0].k = k;

	/** Below, I'm going to set up the indices of the next grid point, which is 
	  either in +p direction or -p direction.  
	  If ftype==Etype, then we are calculating div(E), and we compute the difference 
	  between the current grid point and the next in +p direction.  
	  If ftype==Htype, then we are calculating div(H), and we compute the difference 
	  between the current grid point and the next in -p direction. */
	PetscInt coord_next[] = {i, j, k};  // will be updated according to whether the next grid point is in the +p dir or -p dir
	PetscInt p = coord_next[Pp];  // current p-coordinate, not the next
	PetscInt q = coord_next[Qq];  // current q-coordinate, not the next
	PetscInt r = coord_next[Rr];  // current r-coordinate, not the next

	/** Below, I'm going to set the two matrix elements +1/dp and -1/dp at the locations 
	  corresponding to the current Fp and the next.  If the next Fp is at the boundary (i.e. i==0 
	  for ftype==Htype), ignore -1/dp because no Fp is available there. */
	PetscScalar dp;
	PetscScalar dFp[2];
	PetscInt num_dFp = 2;
	if (ftype==Htype) {
		dp = gi.d_prim[Pp][p];
		--coord_next[Pp];

		indFp[1].i = coord_next[Xx];
		indFp[1].j = coord_next[Yy];
		indFp[1].k = coord_next[Zz];

		/** Two matrix elements in a single row to be set at once. */
		dFp[0] = 1.0/dp; dFp[1] = -1.0/dp;  // used for +(d/dp)Fp

		/** Handle boundary conditions. */
		if (p==0 && gi.bc[Pp][Neg]==PEC) {  // p==0 plane
			/** The tangential component of the E field on PEC is zero, and this effectively 
			  forces the normal component of the H field zero.  Therefore, the normal component 
			  of the H field inside PEC should be antisymmetric to that outside PEC. */
			/** When we deal with a field component a half grid behind the boundary, we need to
			  deal with it as if there is a symmetric or antisymmetric field behind the 
			  boundary.  With a real PMC or PEC, all field components inside PMC or PEC are 
			  zeros, and the surface current or charges support the field patterns in the 
			  problem domain.  But we don't want to have extra surface currents other than the 
			  driving source current.  Therefore we use fictitous symmetric or antisymmetric 
			  field components instead of the surface charge or currents to support the field in 
			  the problem domain. This way, the charge and current distribution with the 
			  boundary remain the same as the those of the symmetric or antisymmetric field 
			  distribution without the boundary. */
			num_dFp = 1;  // dFq[1] and dFr[1] are beyond the matrix index boundary
			dFp[0] = 2.0/dp;
		}
		if (p==0 && gi.bc[Pp][Neg]==PMC) {  // p==0 plane
			/** PMC is usually used to simulate a whole structure with only a half structure when
			  the field distribution is symmetric such that the normal component of the H field 
			  is continuous while the tangential component of the H field is zero.  In the 
			  current case of PMC at p==0, it simulates the continuous Hp and Hq==Hr==0.  
			  But it doesn't mean that (d/dp)Hp = 0.  To simulate the whole structure with only a
			  half structure, the image charge is formed on the PMC, and Hp inside PMC is 
			  essentially zero. 
			  But when we deal with a field component a half grid behind the boundary, we need to
			  deal with it as if there is a symmetric or antisymmetric field behind the 
			  boundary.  
			  With a real PMC or PEC, all field components inside PMC or PEC are zeros, and the 
			  surface current or charges support the field patterns in the problem domain.  But 
			  we don't want to have extra surface currents other than the driving source 
			  current.  Therefore we use fictitous symmetric or antisymmetric field components 
			  instead of the surface charge or currents to support the field in the problem 
			  domain. This way, the charge and current distribution with the boundary remain the 
			  same as the those of the symmetric or antisymmetric field distribution without 
			  the boundary. */
			num_dFp = 1;  // dFq[1] and dFr[1] are beyond the matrix index boundary
			dFp[0] = 0.0;
		}
		if (p==0 && gi.bc[Pp][Pos]==Bloch) {  // p==0 plane
			/** num_dFq==2, num_dFr==2 would access the array elements out of bounds, 
			  but this is OK because MatSetValuesStencil() below supports periodic 
			  indexing. */
			//dFq[1] = gi.exp_neg_ikL[Pp]/Sr;
			//dFr[1] = -gi.exp_neg_ikL[Pp]/Sq
			PetscScalar scale = gi.exp_neg_ikL[Pp];
			dFp[1] /= scale;
		}

		if (q==0 && gi.bc[Qq][Neg]==PEC) {  // q==0 plane
			//dFp[0] = 0.0; dFp[1] = 0.0;
		}
		if (q==0 && gi.bc[Qq][Neg]==PMC) {  // q==0 plane
			/** This effectively forces H components tangential to PMC zero. */
			/** The below is mathematically the same as doing num_dFq = 0, because 
			  num_dFq = 0 keeps matrix elements untouched, which are initially zeros.
			  The difference is in the nonzero pattern of the matrix.  PETSc thinks 
			  whatever elements set are nonzeros, even though we set zeros.  So if we set
			  0.0 as matrix elements, they are actually added to the nonzero pattern of 
			  the matrix, while they aren't in case of num_dFq = 0.  
			  I need them added to the nonzero pattern, because otherwise when I create
			  a matrix A = CE*INV_EPS*C_LH - w^2*mu*S/L, CE*INV_EPS*C_LH does not have 
			  all diagonal elements in the nonzero pattern while w^2*mu*S/L does, which 
			  prevents me from applying MatAXPY with SUBSET_NONZERO_PATTERN to subtract 
			  w^2*mu*S/L from CE*INV_EPS*C_LH in createA() function. */
			dFp[0] = 0.0; dFp[1] = 0.0;
		}

		if (r==0 && gi.bc[Rr][Neg]==PEC) {  // r==0 plane
			//dFp[0] = 0.0; dFp[1] = 0.0;
		}
		if (r==0 && gi.bc[Rr][Neg]==PMC) {  // r==0 plane
			/** This effectively forces H components tangential to PMC zero. */
			dFp[0] = 0.0; dFp[1] = 0.0;
		}
	} else {
		assert(ftype==Etype);

		dp = gi.d_dual[Pp][p];
		++coord_next[Pp];

		indFp[1].i = coord_next[Xx];
		indFp[1].j = coord_next[Yy];
		indFp[1].k = coord_next[Zz];

		/** Two matrix elements in a single row to be set at once. */
		dFp[0] = -1.0/dp; dFp[1] = 1.0/dp;  // used for +(d/dp)Fp

		/** Handle boundary conditions. */
		if (p==0 && gi.bc[Pp][Neg]==PEC) {  // p==0 plane
		}
		if (p==0 && gi.bc[Pp][Neg]==PMC) {  // p==0 plane
			/** The tangential component of the H field on PMC is zero, and this effectively 
			  forces the normal component of the E field zero. */
			dFp[0] = 0.0;
		}
		if (p==Np-1 && gi.bc[Pp][Pos]==Bloch) {  // p==0 plane
			/** num_dFq==2, num_dFr==2 would access the array elements out of bounds, 
			  but this is OK because MatSetValuesStencil() below supports periodic 
			  indexing. */
			//dFq[1] = gi.exp_neg_ikL[Pp]/Sr;
			//dFr[1] = -gi.exp_neg_ikL[Pp]/Sq
			PetscScalar scale = gi.exp_neg_ikL[Pp];
			dFp[1] *= scale;
		}
		if (p==Np-1 && gi.bc[Pp][Pos]!=Bloch) {  // p==0 plane
			/** num_dFq==2, num_dFr==2 would access the array elements out of bounds. */
			/** bc[Pp][Pos]==PMC, so the normal component of the E-field is zero. */
			num_dFp = 1;
		}

		if (q==0 && gi.bc[Qq][Neg]==PEC) {  // q==0 plane
			//dFp[0] = 0.0; dFp[1] = 0.0;
		}
		if (q==0 && gi.bc[Qq][Neg]==PMC) {  // q==0 plane
		}

		if (r==0 && gi.bc[Rr][Neg]==PEC) {  // r==0 plane
			//dFp[0] = 0.0; dFp[1] = 0.0;
		}
		if (r==0 && gi.bc[Rr][Neg]==PMC) {  // r==0 plane
		}
	}

	/** Below, ADD_VALUES is used instead of INSERT_VALUES to deal with cases of 
	  gi.bc[Pp][Neg]==gi.bc[Pp][Pos]==Bloch and Np==1.  In such a case, p==0 and 
	  p==Np-1 coincide, so inserting dFq[1] after dFq[0] overwrites dFq[0], which is
	  not what we want.  On the other hand, if we add dFq[1] to dFq[0], it is 
	  equivalent to insert -1.0/dp + 1.0/dp = 0.0, and this is what should be done
	  because when Np==1, the E fields at p==0 and p==1(==Np) are the same, and the 
	  line integrals on p==0 and p==1 cancel one another. */
	ierr = MatSetValuesStencil(DivF, 1, &indGx, num_dFp, indFp, dFp, ADD_VALUES); CHKERRQ(ierr);  // Gx =  (d/dp)Fp + (d/dq)Fq + (d/dr)Fr

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setDivF"
/**
 * setDivF
 * -----
 * Set up the div(F) operator matrix DivF.  
 * DivF is an N x 3N matrix expanded to 3N x 3N.
 */
PetscErrorCode setDivF(Mat DivF, FieldType ftype, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {
					ierr = setDpOnDivF_at(DivF, ftype, (Axis)axis, i, j, k, gi);
				}
			}
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createDivE"
/**
 * createDivE
 * --------
 * Create the matrix DivE, the divergence operator on E fields.
 */
PetscErrorCode createDivE(Mat *DivE, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = MatCreate(PETSC_COMM_WORLD, DivE); CHKERRQ(ierr);
	ierr = MatSetSizes(*DivE, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*DivE, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*DivE);
	ierr = MatMPIAIJSetPreallocation(*DivE, 6, PETSC_NULL, 3, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*DivE, 6, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*DivE, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*DivE, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setDivF(*DivE, Etype, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*DivE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*DivE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setDpOnFGrad_at"
/**
 * setDpOnFGrad_at
 * ------------
 * Take the F = grad(phi) operator matrix FGrad, and set up the elements for d/d(p) on it, where 
 * F = E, H, and p = x, y, z, at a given location coord[].
 */
PetscErrorCode setDpOnFGrad_at(Mat FGrad, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** In theory FGrad is a 3N x N matrix, but it is much easier to make it a 3N x 3N square
	  matrix with the distributted array (DA) of PETSc.  To that end we leave every 2nd and 3rd 
	  columns of FGrad empty.  This is also consistent with the operator composition such as 
	  EGrad * DivE for grad(div(E)), because DivE is constructed so that every 2nd and 3rd rows are 
	  empty.  
	  In other words, even though phi in F = grad(phi) is a scalar, we take a vector G such that 
	  Gx = phi
	  Gy = 0
	  Gz = 0
	 */

	/** Below, I'm going to set up (d/dz) operations of the following equation:
	  (for ftype==Etype, Pp==Zz):
	  Ez = (d/dz)phi
	  In general, I'm going to set up (d/dp) operation of the following:
	  Fp = (d/dp)Gx
	 */

	/** Indices for Fp */
	MatStencil indFp;  // grid point indices of Fp (mapped to row index of FGrad)
	MatStencil indGx[2];  // current and next grid point indices of Gx (mapped to column indices of FGrad).  The next grid point can be either in +p dir or in -p dir according to ftype.

	/** Determine Qq and Rr from the given Pp. */
	Axis Qq = (Axis)((Pp+1) % Naxis);  // if Pp==Xx, this is Yy    
	Axis Rr = (Axis)((Pp+2) % Naxis);  // if Pp==Xx, this is Zz

	/** Set MatStencil.c, which is the degree-of-freedom (dof) indices of PETSc.  In FD3D this is 
	  used to indicate the direction of the field component. */
	indFp.c = Pp;
	indGx[0].c = Xx; indGx[1].c = Xx; 

	/** Set Np. */
	PetscInt Np = gi.N[Pp];

	/** Set (x,y,z) indices for the current grid point. */
	indFp.i = i; indFp.j = j; indFp.k = k;
	indGx[0].i = i; indGx[0].j = j; indGx[0].k = k;
	indGx[1].i = i; indGx[1].j = j; indGx[1].k = k;

	/** Below, I'm going to set up the indices of the next grid point, which is 
	  either in +p direction or -p direction.  
	  If ftype==Htype, then we are calculating E = grad(phi) where phi is defined at the lower left
	  front node of Yee's cell.  So we compute the difference between phi at the current grid point 
	  and the next in +p direction.  
	  If ftype==Etype, then we are calculating H = grad(phi) where phi is defined at the center of 
	  Yee's cell.  So we compute the difference between the current grid point and the next in -p 
	  direction. */
	PetscInt coord_next[] = {i, j, k};  // will be updated according to whether the next grid point is in the +p dir or -p dir
	PetscInt p = coord_next[Pp];  // current p-coordinate, not the next
	PetscInt q = coord_next[Qq];  // current q-coordinate, not the next
	PetscInt r = coord_next[Rr];  // current r-coordinate, not the next

	/** Below, I'm going to set the two matrix elements -1/dp and +1/dp at the locations 
	  corresponding to the current Fp and the next.  If the next Fp is at the boundary (i.e. i+1==Nx 
	  for ftype==Htype), ignore +1/dp because no Fp is available there. */
	PetscScalar dp;
	PetscScalar dGx[2];
	PetscInt num_dGx = 2;
	if (ftype==Htype) {
		dp = gi.d_dual[Pp][p];
		++coord_next[Pp];

		indGx[1].i = coord_next[Xx];
		indGx[1].j = coord_next[Yy];
		indGx[1].k = coord_next[Zz];

		/** Two matrix elements in a single row to be set at once. */
		dGx[0] = -1.0/dp; dGx[1] = 1.0/dp;  // used for +(d/dp)Fp

		/** Handle boundary conditions. */
		if (p==0 && gi.bc[Pp][Neg]==PEC) {  // p==0 plane
		}
		if (p==0 && gi.bc[Pp][Neg]==PMC) {  // p==0 plane
		}
		if (p==Np-1 && gi.bc[Pp][Pos]==PMC) {  // p==Np plane
			/** Assume that phi on PMC is zero, which is the case when phi = div(esp E). */
			num_dGx = 1;  // dFq[1] and dFr[1] are beyond the matrix index boundary
		}
		if (p==Np-1 && gi.bc[Pp][Pos]==Bloch) {  // p==Np plane
			/** num_dGx==2, num_dGx==2 would access the array elements out of bounds, 
			  but this is OK because MatSetValuesStencil() below supports periodic 
			  indexing. */
			//dFq[1] = gi.exp_neg_ikL[Pp]/Sr;
			//dFr[1] = -gi.exp_neg_ikL[Pp]/Sq
			PetscScalar scale = gi.exp_neg_ikL[Pp];
			dGx[1] *= scale;
		}

		if (q==0 && gi.bc[Qq][Neg]==PEC) {  // q==0 plane
		}
		if (q==0 && gi.bc[Qq][Neg]==PMC) {  // q==0 plane
		}

		if (r==0 && gi.bc[Rr][Neg]==PEC) {  // r==0 plane
		}
		if (r==0 && gi.bc[Rr][Neg]==PMC) {  // r==0 plane
		}
		/*
		   if (indGx[0].i==0 || indGx[0].j==0 || indGx[0].k==0) {
		   dGx[0] = 0.0;
		   }
		   if (indGx[1].i==0 || indGx[1].j==0 || indGx[1].k==0) {
		   dGx[1] = 0.0;
		   }
		 */
	} else {
		assert(ftype==Etype);

		dp = gi.d_prim[Pp][p];
		--coord_next[Pp];

		indGx[1].i = coord_next[Xx];
		indGx[1].j = coord_next[Yy];
		indGx[1].k = coord_next[Zz];

		/** Two matrix elements in a single row to be set at once. */
		dGx[0] = 1.0/dp; dGx[1] = -1.0/dp;  // used for +(d/dp)Fp

		/** Handle boundary conditions. */
		if (p==0 && gi.bc[Pp][Neg]==PEC) {  // p==0 plane
			/** The divergences at symmetric points are the same, which makes the gradient along 
			the surface normal direction zero. */
			dGx[0] = 0.0; dGx[1] = 0.0;
		}
		if (p==0 && gi.bc[Pp][Neg]==PMC) {  // p==0 plane
			/** The divergences at symmetric points are the same, which makes the gradient along 
			the surface normal direction zero. */
			dGx[0] = 0.0; dGx[1] = 0.0;
		}
		if (p==0 && gi.bc[Pp][Pos]==Bloch) {  // p==Np plane
			/** num_dGx==2, num_dGx==2 would access the array elements out of bounds, 
			  but this is OK because MatSetValuesStencil() below supports periodic 
			  indexing. */
			//dFq[1] = gi.exp_neg_ikL[Pp]/Sr;
			//dFr[1] = -gi.exp_neg_ikL[Pp]/Sq
			PetscScalar scale = gi.exp_neg_ikL[Pp];
			dGx[1] /= scale;
		}

		if (q==0 && gi.bc[Qq][Neg]==PEC) {  // q==0 plane
		}
		if (q==0 && gi.bc[Qq][Neg]==PMC) {  // q==0 plane
		}

		if (r==0 && gi.bc[Rr][Neg]==PEC) {  // r==0 plane
		}
		if (r==0 && gi.bc[Rr][Neg]==PMC) {  // r==0 plane
		}
		/*
		   if (indGx[0].i==0 || indGx[0].j==0 || indGx[0].k==0) {
		   dGx[0] = 0.0;
		   }
		   if (indGx[1].i==0 || indGx[1].j==0 || indGx[1].k==0) {
		   dGx[1] = 0.0;
		   }
		 */
	}

	/** Below, ADD_VALUES is used instead of INSERT_VALUES to deal with cases of 
	  gi.bc[Pp][Neg]==gi.bc[Pp][Pos]==Bloch and Np==1.  In such a case, p==0 and 
	  p==Np-1 coincide, so inserting dFq[1] after dFq[0] overwrites dFq[0], which is
	  not what we want.  On the other hand, if we add dFq[1] to dFq[0], it is 
	  equivalent to insert -1.0/dp + 1.0/dp = 0.0, and this is what should be done
	  because when Np==1, the E fields at p==0 and p==1(==Np) are the same, and the 
	  line integrals on p==0 and p==1 cancel one another. */
	ierr = MatSetValuesStencil(FGrad, 1, &indFp, num_dGx, indGx, dGx, ADD_VALUES); CHKERRQ(ierr);  // Gx =  (d/dp)Fp + (d/dq)Fq + (d/dr)Fr

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setFGrad"
/**
 * setFGrad
 * -----
 * Set up the F = grad(phi) operator matrix FGrad.  
 * FGrad is a 3N x N matrix expanded to 3N x 3N.
 */
PetscErrorCode setFGrad(Mat FGrad, FieldType ftype, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {
					ierr = setDpOnFGrad_at(FGrad, ftype, (Axis)axis, i, j, k, gi);
				}
			}
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createEGrad"
/**
 * createEGrad
 * --------
 * Create the matrix EGrad, the gradient operator generating E fields.
 */
PetscErrorCode createEGrad(Mat *EGrad, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = MatCreate(PETSC_COMM_WORLD, EGrad); CHKERRQ(ierr);
	ierr = MatSetSizes(*EGrad, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*EGrad, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*EGrad);
	ierr = MatMPIAIJSetPreallocation(*EGrad, 2, PETSC_NULL, 1, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*EGrad, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*EGrad, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*EGrad, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setFGrad(*EGrad, Etype, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*EGrad, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*EGrad, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setDpOnCF_at"
/**
 * setDpOnCF_at
 * ------------
 * Take the curl(F) operator matrix CF for given F == E or H, and set up the elements
 * for d/d(p) on it, where p = x, y, z, at a given location coord[].
 */
PetscErrorCode setDpOnCF_at(Mat CF, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Notation: 
	    F: field to differentiate.  Either E or H.  Given by ftype.
	    G: if F==E, then G==H, and if F==H, then G==E.
	    (p,q,r): cyclic permutation of (x,y,z)
	    (Pp,Qq,Rr): cyclic permutation of (Xx,Yy,Zz)
	  Therefore, if Gq==Hy, then Fp==Ex and dr==dx, and so on. */

	/** For general ftype and Pp, I'm going to set up (d/dp) operations of:
	      -i w mu Gr = [curl(F)]r = (d/dp)Fq - (d/dq)Fp
	      -i w mu Gq = [curl(F)]q = (d/dr)Fp - (d/dp)Fr
	  if (F,G)==(E,H), or 
	      i w eps Gr = [curl(F)]r = (d/dp)Fq - (d/dq)Fp
	      i w eps Gq = [curl(F)]q = (d/dr)Fp - (d/dp)Fr
	  if (F,G)==(H,E).  (p,q,r) is a cyclic permutation of (x,y,z). 
	  If ftype==Etype and Pp==Xx, this means that I'm going to set up (d/dx) operations of the 
	  following two equations:
	      -i w mu Hz = [curl(E)]z = (d/dx)Ey - (d/dy)Ex 
	      -i w mu Hy = [curl(E)]y = (d/dz)Ex - (d/dx)Ez 
	  In both (F,G)==(E,H) and (H,E) cases, we set up [curl(F)]r matrix elements for Gr, 
	  or [curl(F)]q matrix elements for Gq.  I will denote these as 
	      Gr <-- [curl(F)]r
	      Gq <-- [curl(F)]q
	 */

	/** For (d/dp) term in Gr <-- [curl(F)]r */
	MatStencil indGr;  // grid point indices of Gr (mapped to row index of CF)
	MatStencil indFq[2];  // current and next grid point indices of Fq (mapped to column indices of CF).  The next grid point can be either in +p dir or in -p dir.

	/** For (d/dp) term in Gq <-- [curl(F)]q */
	MatStencil indGq;  // grid point indices of Gq (mapped to row index of CF)
	MatStencil indFr[2];  // current and next grid point indices of Fr (mapped to column indices of CF).  The next grid point can be either in +p dir or in -p dir.

	/** Determine Qq and Rr from the given Pp. */
	Axis Qq = (Axis)((Pp+1) % Naxis);  // if Pp==Xx, this is Yy
	Axis Rr = (Axis)((Pp+2) % Naxis);  // if Pp==Xx, this is Zz

	/** Set MatStencil.c, which is the degree-of-freedom (dof) indices of PETSc.  In FD3D this is 
	  used to indicate the direction of the field component. */
	indGr.c = Rr; indFq[0].c = Qq; indFq[1].c = Qq;
	indGq.c = Qq; indFr[0].c = Rr; indFr[1].c = Rr;

	/** Set Np. */
	PetscInt Np = gi.N[Pp];

	/** Set (x,y,z) indices for the current grid point. */
	indGr.i = i; indGr.j = j; indGr.k = k;
	indFq[0].i = i; indFq[0].j = j; indFq[0].k = k;

	indGq.i = i; indGq.j = j; indGq.k = k;
	indFr[0].i = i; indFr[0].j =j; indFr[0].k = k;

	/** Below, I'm going to set up the indices of the next grid point, which is 
	  either in +p direction or -p direction.  
	  If ftype==Htype, then we are calculating curl[E], and we compute the difference 
	  between the current grid point and the next in +p direction.  
	  If ftype==Etype, then we are calculating curl[H], and we compute the difference 
	  between the current grid point and the next in -p direction. */
	PetscInt coord_next[] = {i, j, k};  // will be updated according to whether the next grid point is in the +p dir or -p dir
	PetscInt p = coord_next[Pp];  // current p-coordinate, not the next
	PetscInt q = coord_next[Qq];  // current q-coordinate, not the next
	PetscInt r = coord_next[Rr];  // current r-coordinate, not the next

	/** Below, I'm going to set the two matrix elements -1/dp and +1/dp at the 
	  locations corresponding to the current Fr (or Fq) and the next.  If the next Fr 
	  (or Fq) is at the boundary (i.e. i+1==Nx for ftype==Htype, Pp==Xx), ignore +1/dp
	  because no Fr (or Fq) is available there. */
	PetscScalar dp;
	PetscScalar dFq[2];
	PetscScalar dFr[2];
	PetscInt num_dFq = 2;
	PetscInt num_dFr = 2;
	if (ftype==Htype) {
		//PetscScalar Sr = gi.d_prim[Pp][p] * gi.d_prim[Qq][q];
		//PetscScalar Sq = gi.d_prim[Rr][r] * gi.d_prim[Pp][p];
		dp = gi.d_dual[Pp][p];
		++coord_next[Pp];

		indFr[1].i = coord_next[Xx]; 
		indFr[1].j = coord_next[Yy]; 
		indFr[1].k = coord_next[Zz];

		indFq[1].i = coord_next[Xx]; 
		indFq[1].j = coord_next[Yy]; 
		indFq[1].k = coord_next[Zz];

		/** Two matrix elements in a single row to be set at once. */
		//PetscScalar dFq[] = {-1.0/Sr, 1.0/Sr};  // used for +(d/dp)Fq = +(Lq Fq)/Sr
		//PetscScalar dFr[] = {1.0/Sq, -1.0/Sq};  // used for -(d/dp)Fr = -(Lr Fr)/Sq
		/** Here, we cannot just use dFq[] = {-1.0, 1.0} and dFr[] = {1.0, -1.0} and multiply some
		  diagonal matrix later to CE.  dFq and dFr are divided by dp evendually, which means that 
		  two different elements of a vector supplied to CE should be scaled by the same amount. This
		  is equivalent to scaling one element of the vector should scaled by different values 
		  depending on matrix elements. (This is also obvious on the nonuniform Yee's grid.) However, 	   multiplying a diagonal matrix to CE is equivalent to scaling a vector supplied to CE 
		  elementwise, which scale each vector element by a unique amount. */
		dFq[0] = -1.0/dp; dFq[1] = 1.0/dp;  // used for +(d/dp)Fq
		dFr[0] = 1.0/dp; dFr[1] = -1.0/dp;  // used for -(d/dp)Fr

		/** Handle boundary conditions. */
		if (p==0 && gi.bc[Pp][Neg]==PEC) {  // p==0 plane
			/*
			   dFq[0] = -2.0/Sr; dFr[0] = 2.0/Sq; 
			   num_dFq = 1; num_dFr = 1;
			 */
			/*
			   dFq[0] = -1.0/(3.0*Sr); dFq[1] = 1.0/(3.0*Sr);
			   dFr[0] = -1.0/(3.0*Sq); dFr[1] = 1.0/(3.0*Sq);

			   indFr[0].i = coord_next[Xx]; 
			   indFr[0].j = coord_next[Yy]; 
			   indFr[0].k = coord_next[Zz];

			   indFq[0].i = coord_next[Xx]; 
			   indFq[0].j = coord_next[Yy]; 
			   indFq[0].k = coord_next[Zz];

			   ++coord_next[Pp];

			   indFr[1].i = coord_next[Xx]; 
			   indFr[1].j = coord_next[Yy]; 
			   indFr[1].k = coord_next[Zz];

			   indFq[1].i = coord_next[Xx]; 
			   indFq[1].j = coord_next[Yy]; 
			   indFq[1].k = coord_next[Zz];
			 */
		}
		if (p==0 && gi.bc[Pp][Neg]==PMC) {  // p==0 plane
			/** This effectively forces H components tangential to PMC zero. */
			dFq[0] = 0.0; 
			dFr[0] = 0.0;
		}
		if (p==Np-1 && gi.bc[Pp][Pos]==Bloch) {  // p==Np plane
			/** num_dFq==2, num_dFr==2 would access the array elements out of bounds, 
			  but this is OK because MatSetValuesStencil() below supports periodic 
			  indexing. */
			//dFq[1] = gi.exp_neg_ikL[Pp]/Sr;
			//dFr[1] = -gi.exp_neg_ikL[Pp]/Sq
			PetscScalar scale = gi.exp_neg_ikL[Pp];
			dFq[1] *= scale;
			dFr[1] *= scale;
		}
		if (p==Np-1 && gi.bc[Pp][Pos]!=Bloch) {  // p==Np plane
			/** num_dFq==2, num_dFr==2 would access the array elements out of bounds. */ 
			num_dFq = 1;
			num_dFr = 1;
		}

		if (q==0 && gi.bc[Qq][Neg]==PEC) {  // q==0 plane
			//dFr[0] = 0.0; dFr[1] = 0.0;
		}
		if (q==0 && gi.bc[Qq][Neg]==PMC) {  // q==0 plane
			/** This effectively forces H components tangential to PMC zero. */
			/** The below is mathematically the same as doing num_dFq = 0, because 
			  num_dFq = 0 keeps matrix elements untouched, which are initially zeros.
			  The difference is in the nonzero pattern of the matrix.  PETSc thinks 
			  whatever elements set are nonzeros, even though we set zeros.  So if we set
			  0.0 as matrix elements, they are actually added to the nonzero pattern of 
			  the matrix, while they aren't in case of num_dFq = 0.  
			  I need them added to the nonzero pattern, because otherwise when I create
			  a matrix A = CE*INV_EPS*C_LH - w^2*mu*S/L, CE*INV_EPS*C_LH does not have 
			  all diagonal elements in the nonzero pattern while w^2*mu*S/L does, which 
			  prevents me from applying MatAXPY with SUBSET_NONZERO_PATTERN to subtract 
			  w^2*mu*S/L from CE*INV_EPS*C_LH in createA() function. */
			dFr[0] = 0.0; dFr[1] = 0.0;
		}

		if (r==0 && gi.bc[Rr][Neg]==PEC) {  // r==0 plane
			//dFq[0] = 0.0; dFq[1] = 0.0;
		}
		if (r==0 && gi.bc[Rr][Neg]==PMC) {  // r==0 plane
			/** This effectively forces H components tangential to PMC zero. */
			dFq[0] = 0.0; dFq[1] = 0.0;
		}
	} else {
		assert(ftype==Etype);
		//PetscScalar dq = gi.d_dual[Qq][q];
		//PetscScalar dr = gi.d_dual[Rr][r];
		dp = gi.d_prim[Pp][p];
		--coord_next[Pp];  // It measn coord_next[Pp] is actually the "previous" coordinate in +p direction.

		indFr[1].i = coord_next[Xx]; 
		indFr[1].j = coord_next[Yy]; 
		indFr[1].k = coord_next[Zz];

		indFq[1].i = coord_next[Xx]; 
		indFq[1].j = coord_next[Yy]; 
		indFq[1].k = coord_next[Zz];

		/** Two matrix elements in a single row to be set at once.  Note the sign 
		  differences from ftype==Htype case, we have done --coord_next[Pp] instead of
		  ++coord_next[Pp]. */
		//PetscScalar dFq[] = {dq, -dq};  // used for +(d/dp)Fq = +(Lq Fq)/Sr -> Gr
		//PetscScalar dFr[] = {-dr, dr};  // used for -(d/dp)Fr = -(Lr Fr)/Sq -> Gq
		/** Here, we cannot just use dFq[] = {-1.0, 1.0} and dFr[] = {1.0, -1.0} and multiply some
		  diagonal matrix later to CE.  dFq and dFr are divided by dp evendually, which means that 
		  two different elements of a vector supplied to CE should be scaled by the same amount. This
		  is equivalent to scaling one element of the vector should scaled by different values 
		  depending on matrix elements. (This is also obvious on the nonuniform Yee's grid.) However, 	   multiplying a diagonal matrix to CE is equivalent to scaling a vector supplied to CE 
		  elementwise, which scale each vector element by a unique amount. */
		dFq[0] = 1.0/dp; dFq[1] = -1.0/dp;  // used for +(d/dp)Fq
		dFr[0] = -1.0/dp; dFr[1] = 1.0/dp;  // used for -(d/dp)Fr

		/** Handle boundary conditions. */
		if (p==0 && gi.bc[Pp][Neg]==PEC) {  // p==0 plane
			/** PEC is usually used to simulate a whole structure with only a half structure 
			  when the field distribution is symmetric such that the normal component of the E 
			  field is continuous and the tangential component of the E field is zero.  In the 
			  current case of PEC at p==0, it simulates the continuous Ep and Eq==Er==0, which 
			  leads to the antisymmetric distribution of Eq and Er around p==0 plane.  Then it 
			  seems like (d/dp)Eq = (2.0/dp)*Eq and (d/dp)Er = (2.0/dp)*Er at p==0.
			  But in reality (d/dp)Eq and (d/dp)Er at p==0 are not like that.  To simulate the 
			  whole structure with only a half structure, the image current is formed on PEC, 
			  and Eq and Er inside PEC is essentially zero. Therefore, the correct formulation of
			  Maxwell's equations is to leave (d/dp)Eq = (1.0/dp)*Eq and (d/dp)Er = (1.0/dp)*Er,
			  and put the additional magnetic surface current on PEC.  But this leads to a 
			  technical difficulty.  First, in the continuous (not distretized) case Eq and Er on
			  PEC are zeros, and the tangential component of curl(E) is zero on PEC.  This means 
			  that the tangential component of the H field is completely generated by the surface
			  magnetic current without any curl(E).  In other words, PEC can support any value of
			  the tangential component of the H field; an appropriate surface current will be 
			  readily generated.
			  So instead of this "real" PEC, it is better to model PEC as a layer a half-grid 
			  under the interface that actively generates an antisymmetric tangential component 
			  of the H field.
			 */
			num_dFq = 1; num_dFr = 1;  // dFq[1] and dFr[1] are beyond the matrix index boundary
			//dFq[0] = 2.0*dq; dFr[0] = -2.0*dr;
			//dFq[0] = 1.0/dp; dFr[0] = -1.0/dp;
			dFq[0] = 2.0/dp; dFr[0] = -2.0/dp;
		}
		if (p==0 && gi.bc[Pp][Neg]==PMC) {  // p==0 plane
			/** This is the Neumann condition for PMC.  Basically, this makes the 
			  resulting H field components Hq and Hr zero on p == 0 PMC.  If the source 
			  components Mq and Mr are not zero on p == 0 PMC, I think we need to force 
			  them zero when constructing the source vector. */
			/** Note that the following is the same as num_dFq = 2, num_dFr = 2, dFq =
			  {0.0, 0.0}, dFr = {0.0, 0.0}.  But then the nonzero element pattern becomes
			  different.  So, when we perform MatAXPY(dA, -1.0, A, strct), we cannot use 
			  strct == SAME_NONZERO_PATTERN anymore. */
			num_dFq = 1; num_dFr = 1;  // dFq[1] and dFr[1] are beyond the matrix index boundary
			dFq[0] = 0.0;
			dFr[0] = 0.0;
		}
		if (p==0 && gi.bc[Pp][Neg]==Bloch) {  // p==0 plane
			/** num_dFq==2, num_dFr==2 would access the array elements out of bounds, 
			  but this is OK because MatSetValuesStencil() below supports periodic 
			  indexing. */
			//dFq[1] = -dq/gi.exp_neg_ikL[Pp];
			//dFr[1] = dr/gi.exp_neg_ikL[Pp];
			PetscScalar scale = gi.exp_neg_ikL[Pp];
			dFq[1] /= scale;
			dFr[1] /= scale;
		}

		if (q==0 && gi.bc[Qq][Neg]==PEC) {  // q==0 plane
		}
		if (q==0 && gi.bc[Qq][Neg]==PMC) {  // q==0 plane
			/** This is the Neumann condition for PMC.  Because the H field components Hr and
			  Hp are zeros on q == 0 PMC, and Eq that is normal to the PMC should be zero. */ 
			dFq[0] = 0.0; dFq[1] = 0.0;
		}

		if (r==0 && gi.bc[Rr][Neg]==PEC) {  // r==0 plane
		}
		if (r==0 && gi.bc[Rr][Neg]==PMC) {  // r==0 plane
			/** This is the Neumann condition for PMC.  Because the H field components Hp and
			  Hq are zeros on r == 0 PMC, Er that is normal to the PMC should be zero. */ 
			dFr[0] = 0.0; dFr[1] = 0.0;
		}
	}

	/** Below, ADD_VALUES is used instead of INSERT_VALUES to deal with cases of 
	  gi.bc[Pp][Neg]==gi.bc[Pp][Pos]==Bloch and Np==1.  In such a case, p==0 and 
	  p==Np-1 coincide, so inserting dFq[1] after dFq[0] overwrites dFq[0], which is
	  not what we want.  On the other hand, if we add dFq[1] to dFq[0], it is 
	  equivalent to insert -1.0/dp + 1.0/dp = 0.0, and this is what should be done
	  because when Np==1, the E fields at p==0 and p==1(==Np) are the same, and the 
	  line integrals on p==0 and p==1 cancel one another. */
	ierr = MatSetValuesStencil(CF, 1, &indGr, num_dFq, indFq, dFq, ADD_VALUES); CHKERRQ(ierr);  // Gr <-- [curl(F)]r = (d/dp)Fq - (d/dq)Fp
	ierr = MatSetValuesStencil(CF, 1, &indGq, num_dFr, indFr, dFr, ADD_VALUES); CHKERRQ(ierr);  // Gq <-- [curl(F)]q = (d/dr)Fp - (d/dp)Fr

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setCF"
/**
 * setCF
 * -----
 * Set up the curl(F) operator matrix CF for given F == E or H.
 */
PetscErrorCode setCF(Mat CF, FieldType ftype, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {
					ierr = setDpOnCF_at(CF, ftype, (Axis)axis, i, j, k, gi);
				}
			}
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createCH"
/**
 * createCH
 * --------
 * Create the matrix CH, the curl operator on H fields.
 */
PetscErrorCode createCH(Mat *CH, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = MatCreate(PETSC_COMM_WORLD, CH); CHKERRQ(ierr);
	ierr = MatSetSizes(*CH, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*CH, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*CH);
	ierr = MatMPIAIJSetPreallocation(*CH, 4, PETSC_NULL, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*CH, 4, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*CH, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*CH, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setCF(*CH, Htype, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*CH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*CH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createCE"
/**
 * createCE
 * --------
 * Create the matrix CE, the curl operator on E fields.
 */
PetscErrorCode createCE(Mat *CE, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = MatCreate(PETSC_COMM_WORLD, CE); CHKERRQ(ierr);
	ierr = MatSetSizes(*CE, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*CE, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*CE);
	ierr = MatMPIAIJSetPreallocation(*CE, 4, PETSC_NULL, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*CE, 4, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*CE, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*CE, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setCF(*CE, Etype, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*CE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*CE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createCHE"
/**
 * createCHE
 * -------
 * Create the matrix CHE, the curl(mu^-1 curl) operator on E fields.
 */

PetscErrorCode createCHE(Mat *CHE, Mat CH, Mat HE, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/*
	   ierr = MatCreate(PETSC_COMM_WORLD, CHE); CHKERRQ(ierr);
	   ierr = MatSetSizes(*CHE, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	   ierr = MatSetType(*CHE, MATRIX_TYPE); CHKERRQ(ierr);
	   ierr = MatSetFromOptions(*CHE);
	 */

	/** Below, 13 is the maximum number of nonzero elements in a diagonal portion of 
	  a local submatrix. e.g., if CHE is 9-by-9 and distributed among 3 processors, 
	  entire row (0,1,2) compose a submatrix in processsor 0, and row (3,4,5) in 
	  processor 1, row (6,7,8) in processor 2.  In processor 1, the diagonal portion 
	  means 3-by-3 square matrix at the diagonal, which is composed of row (3,4,5) and 
	  column (3,4,5).  Each row of CHE corresponds to one of Ex, Ey, Ez component of 
	  some cell in the grid.

	  When CHE is multiplied to a vector x, which has 3*Nx*Ny*Nz E field components, a 
	  row of CHE generates an output E field component out of 13 input E field 
	  components; each output E field component is involved in 4 curl loops, in each of 
	  which 3 extra E field components are introduced.  Therefore, an output E field 
	  component is the result of interactions between 1(itself) + 4(# of curl loops) * 
	  3(# of extra E field components in each loop) = 13 input E field components.

	  If the cell containing the output E field component is in interior of a local 
	  portion of the Yee's grid, then all 4 curl loops lie inside the local grid.  
	  Therefore, at most 13 E field components can be in the diagonal portion of a local 
	  submatrix.

	  On the other hand, if the cell is at a boundary of a local grid, then some of 4 
	  curl loops lie outside the local grid.  As an extreme case, if the local grid is
	  composed of only one cell, then only 3 E field components are in the local grid, 
	  and therefore 10 E field components are in the off-diagonal portion of the 
	  submatrix. */
	/*
	   ierr = MatMPIAIJSetPreallocation(*CHE, 13, PETSC_NULL, 10, PETSC_NULL); CHKERRQ(ierr);
	   ierr = MatSeqAIJSetPreallocation(*CHE, 13, PETSC_NULL); CHKERRQ(ierr);
	   ierr = MatSetLocalToGlobalMapping(*CHE, gi.map, gi.map); CHKERRQ(ierr);
	   ierr = MatSetStencil(*CHE, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	 */

	/** Set up the matrix CHE. */
	/** Below, we use 15.0 instead of 13.0, because some row has explicit zeros, which
	  take memories as if they are nonzeros. */
	ierr = MatMatMult(CH, HE, MAT_INITIAL_MATRIX, 13.0/(4.0+4.0), CHE); CHKERRQ(ierr); // CHE = CH*invMu*CE
	//ierr = MatMatMult(CH, HE, MAT_INITIAL_MATRIX, PETSC_DEFAULT, CHE); CHKERRQ(ierr); // CHE = CH*invMu*CE

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setAtemplate_at"
/**
 * setAtemplate_at
 * ------------
 * Set the elements of Atemplate zeros.
 */
PetscErrorCode setAtemplate_at(Mat Atemplate, FieldType ftype, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Notation: 
w: subscript indicating one of x, y, z
F: input field type.  Either E or H.  Given by ftype.
G: output field type.  Has the same type as F.
(p,q,r): cyclic permutation of (x,y,z)
(Pp,Qq,Rr): cyclic permutation of (Xx,Yy,Zz)
Therefore, if Gq==Ey, then Fp==Ex and dr==dx, and so on. */

	/** Below, I'm going to set 15 input field components for one output field component Gp. 
	  The 15 elements are from the 4 curl loops along which Gp lies, and 2 divergences at the two end
	  points of Gp. */
	MatStencil indGp;  // grid point indices of Gp (mapped to row index of Atemplate)
	MatStencil indFw[15];  // grid point indices of Fw (mapped to column indices of Atemplate)

	/** Determine Qq and Rr from the given Pp. */
	Axis Qq = (Axis)((Pp+1) % Naxis);  // if Pp==Xx, this is Yy
	Axis Rr = (Axis)((Pp+2) % Naxis);  // if Pp==Xx, this is Zz

	if (ftype==Htype) {
		/** Set (x,y,z,axis) indices for the output field component Gp. */
		/** indGp.c is the degree-of-freedom (dof) index; in FD3D used to indicate the direction of 
		  the field component. */
		indGp.i = i; indGp.j = j; indGp.k = k; indGp.c = Pp; 

		/** Set (x,y,z,axis) indices for the input field components Fw's. */
		PetscInt coordFw[3];  // coordinate of an input field component

		/** Fw[0]: the same as the output field component */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		indFw[0].i = coordFw[Xx]; indFw[0].j = coordFw[Yy]; indFw[0].k = coordFw[Zz]; indFw[0].c = Pp;

		/** Fw[1~2]: Fp at (p,q,r) = (p+-1,q,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Pp];
		indFw[1].i = coordFw[Xx]; indFw[1].j = coordFw[Yy]; indFw[1].k = coordFw[Zz]; indFw[1].c = Pp;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Pp];
		indFw[2].i = coordFw[Xx]; indFw[2].j = coordFw[Yy]; indFw[2].k = coordFw[Zz]; indFw[2].c = Pp;

		/** Fw[3~6]: Fp at (p,q,r) = (p,q+-1,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Qq];
		indFw[3].i = coordFw[Xx]; indFw[3].j = coordFw[Yy]; indFw[3].k = coordFw[Zz]; indFw[3].c = Pp;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Qq];
		indFw[4].i = coordFw[Xx]; indFw[4].j = coordFw[Yy]; indFw[4].k = coordFw[Zz]; indFw[4].c = Pp;

		/** Fw[3~6]: Fp at (p,q,r) = (p,q,r+-1) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Rr];
		indFw[5].i = coordFw[Xx]; indFw[5].j = coordFw[Yy]; indFw[5].k = coordFw[Zz]; indFw[5].c = Pp;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Rr];
		indFw[6].i = coordFw[Xx]; indFw[6].j = coordFw[Yy]; indFw[6].k = coordFw[Zz]; indFw[6].c = Pp;

		/** Fw[7~8]: Fq and Fr at (p,q,r) = (p,q,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		indFw[7].i = coordFw[Xx]; indFw[7].j = coordFw[Yy]; indFw[7].k = coordFw[Zz]; indFw[7].c = Qq;
		indFw[8].i = coordFw[Xx]; indFw[8].j = coordFw[Yy]; indFw[8].k = coordFw[Zz]; indFw[8].c = Rr;

		/** Fw[9~10]: Fq and Fr at (p,q,r) = (p+1,q,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Pp];
		indFw[9].i = coordFw[Xx]; indFw[9].j = coordFw[Yy]; indFw[9].k = coordFw[Zz]; indFw[9].c = Qq;
		indFw[10].i = coordFw[Xx]; indFw[10].j = coordFw[Yy]; indFw[10].k = coordFw[Zz]; indFw[10].c = Rr;

		/** Fw[11~12]: Fq at (p,q,r) = (p,q-1,r) and (p+1,q-1,r)*/
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Qq];
		indFw[11].i = coordFw[Xx]; indFw[11].j = coordFw[Yy]; indFw[11].k = coordFw[Zz]; indFw[11].c = Qq;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Pp]; --coordFw[Qq];
		indFw[12].i = coordFw[Xx]; indFw[12].j = coordFw[Yy]; indFw[12].k = coordFw[Zz]; indFw[12].c = Qq;

		/** Fw[13~14]: Fr at (p,q,r) = (p,q,r-1) and (p+1,q,r-1)*/
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Rr];
		indFw[13].i = coordFw[Xx]; indFw[13].j = coordFw[Yy]; indFw[13].k = coordFw[Zz]; indFw[13].c = Rr;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Pp]; --coordFw[Rr];
		indFw[14].i = coordFw[Xx]; indFw[14].j = coordFw[Yy]; indFw[14].k = coordFw[Zz]; indFw[14].c = Rr;
	} else {
		assert(ftype==Etype);

		/** Set (x,y,z,axis) indices for the output field component Gp. */
		/** indGp.c is the degree-of-freedom (dof) index; in FD3D used to indicate the direction of 
		  the field component. */
		indGp.i = i; indGp.j = j; indGp.k = k; indGp.c = Pp; 

		/** Set (x,y,z,axis) indices for the input field components Fw's. */
		PetscInt coordFw[3];  // coordinate of an input field component

		/** Fw[0]: the same as the output field component */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		indFw[0].i = coordFw[Xx]; indFw[0].j = coordFw[Yy]; indFw[0].k = coordFw[Zz]; indFw[0].c = Pp;

		/** Fw[1~2]: Fp at (p,q,r) = (p+-1,q,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Pp];
		indFw[1].i = coordFw[Xx]; indFw[1].j = coordFw[Yy]; indFw[1].k = coordFw[Zz]; indFw[1].c = Pp;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Pp];
		indFw[2].i = coordFw[Xx]; indFw[2].j = coordFw[Yy]; indFw[2].k = coordFw[Zz]; indFw[2].c = Pp;

		/** Fw[3~6]: Fp at (p,q,r) = (p,q+-1,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Qq];
		indFw[3].i = coordFw[Xx]; indFw[3].j = coordFw[Yy]; indFw[3].k = coordFw[Zz]; indFw[3].c = Pp;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Qq];
		indFw[4].i = coordFw[Xx]; indFw[4].j = coordFw[Yy]; indFw[4].k = coordFw[Zz]; indFw[4].c = Pp;

		/** Fw[3~6]: Fp at (p,q,r) = (p,q,r+-1) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Rr];
		indFw[5].i = coordFw[Xx]; indFw[5].j = coordFw[Yy]; indFw[5].k = coordFw[Zz]; indFw[5].c = Pp;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Rr];
		indFw[6].i = coordFw[Xx]; indFw[6].j = coordFw[Yy]; indFw[6].k = coordFw[Zz]; indFw[6].c = Pp;

		/** Fw[7~8]: Fq and Fr at (p,q,r) = (p,q,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		indFw[7].i = coordFw[Xx]; indFw[7].j = coordFw[Yy]; indFw[7].k = coordFw[Zz]; indFw[7].c = Qq;
		indFw[8].i = coordFw[Xx]; indFw[8].j = coordFw[Yy]; indFw[8].k = coordFw[Zz]; indFw[8].c = Rr;

		/** Fw[9~10]: Fq and Fr at (p,q,r) = (p-1,q,r) */
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Pp];
		indFw[9].i = coordFw[Xx]; indFw[9].j = coordFw[Yy]; indFw[9].k = coordFw[Zz]; indFw[9].c = Qq;
		indFw[10].i = coordFw[Xx]; indFw[10].j = coordFw[Yy]; indFw[10].k = coordFw[Zz]; indFw[10].c = Rr;

		/** Fw[11~12]: Fq at (p,q,r) = (p,q+1,r) and (p-1,q+1,r)*/
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Qq];
		indFw[11].i = coordFw[Xx]; indFw[11].j = coordFw[Yy]; indFw[11].k = coordFw[Zz]; indFw[11].c = Qq;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Pp]; ++coordFw[Qq];
		indFw[12].i = coordFw[Xx]; indFw[12].j = coordFw[Yy]; indFw[12].k = coordFw[Zz]; indFw[12].c = Qq;

		/** Fw[13~14]: Fr at (p,q,r) = (p,q,r+1) and (p-1,q,r+1)*/
		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		++coordFw[Rr];
		indFw[13].i = coordFw[Xx]; indFw[13].j = coordFw[Yy]; indFw[13].k = coordFw[Zz]; indFw[13].c = Rr;

		coordFw[Xx] = i; coordFw[Yy] = j; coordFw[Zz] = k;
		--coordFw[Pp]; ++coordFw[Rr];
		indFw[14].i = coordFw[Xx]; indFw[14].j = coordFw[Yy]; indFw[14].k = coordFw[Zz]; indFw[14].c = Rr;
	}

	/** Below, I'm going to set zeros at (indGp, indFw[]). */
	/** 15 matrix elements in a single row to be set at once. */
	PetscScalar Fw[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	PetscInt num_Fw = 15;

	/** Below, INSERT_VALUES is used instead of ADD_VALUES to force the elements to be zeros. */
	ierr = MatSetValuesStencil(Atemplate, 1, &indGp, num_Fw, indFw, Fw, INSERT_VALUES); CHKERRQ(ierr);  

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setAtemplate"
/**
 * setAtemplate
 * -----
 * Set up Atemplate matrix that has zeros preset for the nonzero elements of A.
 */
PetscErrorCode setAtemplate(Mat Atemplate, FieldType ftype, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {
					ierr = setAtemplate_at(Atemplate, ftype, (Axis)axis, i, j, k, gi);
				}
			}
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createAtemplate"
/**
 * createAtemplate
 * --------
 * Create the template matrix of A.  It has an appropriate nonzero element pattern predetermined.
 */
PetscErrorCode createAtemplate(Mat *Atemplate, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	ierr = MatCreate(PETSC_COMM_WORLD, Atemplate); CHKERRQ(ierr);
	ierr = MatSetSizes(*Atemplate, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*Atemplate, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*Atemplate);
	ierr = MatMPIAIJSetPreallocation(*Atemplate, 15, PETSC_NULL, 12, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*Atemplate, 15, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*Atemplate, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*Atemplate, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setAtemplate(*Atemplate, Etype, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*Atemplate, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*Atemplate, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createGD"
/**
 * createGD
 * -------
 * Create the matrix GD, the grad(eps^-1 div) operator on E fields.
 */
PetscErrorCode createGD(Mat *GD, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Set up the matrix DivE, the divergence operator on E fields. */
	Mat DivE;
	ierr = createDivE(&DivE, gi); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nDivE\n"); CHKERRQ(ierr);
	   ierr = MatView(DivE, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Set up the matrix EGrad, the gradient operator generating E fields. */
	Mat EGrad;
	ierr = createEGrad(&EGrad, gi); CHKERRQ(ierr);

	/** Set the inverse of the permittivity vector at nodes, and left-scale DivE by invEpsNode. */
	Vec invEpsNode;
	//ierr = create_epsNode(&invEpsNode, gi); CHKERRQ(ierr);
	//ierr = createFieldArray(&invEpsNode, set_epsNode_at, gi); CHKERRQ(ierr);
	//ierr = createVecHDF5(&invEpsNode, "/eps_node", gi); CHKERRQ(ierr);
	ierr = createVecPETSc(&invEpsNode, "eps_node", gi); CHKERRQ(ierr);
	ierr = VecReciprocal(invEpsNode); CHKERRQ(ierr);
	ierr = MatDiagonalScale(DivE, invEpsNode, PETSC_NULL); CHKERRQ(ierr);
	ierr = VecDestroy(&invEpsNode); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nEGrad\n"); CHKERRQ(ierr);
	   ierr = MatView(EGrad, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Create the matrix GD. */
	/** Below, 11 is the maximum number of nonzero elements in a diagonal portion of 
	  a local submatrix. e.g., if GD is 9-by-9 and distributed among 3 processors, 
	  entire row (0,1,2) compose a submatrix in processsor 0, and row (3,4,5) in 
	  processor 1, row (6,7,8) in processor 2.  In processor 1, the diagonal portion 
	  means 3-by-3 square matrix at the diagonal, which is composed of row (3,4,5) and 
	  column (3,4,5).  Each row of GD corresponds to one of Ex, Ey, Ez component of 
	  some cell in the grid.

	  When GD is multiplied to a vector x, which has 3*Nx*Ny*Nz E field components, a 
	  row of GD generates an output E field component out of 11 input E field 
	  components; each output E field component is calculated as the gradient between the 
	  scalar potentials at two nodal points, and each scalar potential value is calculated 
	  as the divergence at a point, which involves 6 E field components.  So, the output E 
	  field component is calculated out of 6 + 6 input E field components, but one component
	  is shared between the two sets of 6 components, and therefore the output E field 
	  component comes from 6 + 6 - 1 = 11 input E field components.

	  If the cell containing the output E field component is in interior of a local 
	  portion of the Yee's grid, then all 11 input E field components can be in the diagonal
	  portion of a local submatrix.

	  On the other hand, if the cell is at a boundary of a local grid, then some of 11 
	  input E field components can lie outside the local grid.  It is obivous that 3 components
	  that are in the same cell as the output E field component are guaranteed to be in the same
	  local grid, but the rest 11 - 3 = 8 components can lie outside.  Therefore 8 input E field 
	  components can be in the off-diagonal portion of the submatrix. */

	/** Set up the matrix GD. */
	ierr = MatMatMult(EGrad, DivE, MAT_INITIAL_MATRIX, 11.0/(2.0+6.0), GD); CHKERRQ(ierr); // GD = EGrad*invEpsNode*DivE
	//ierr = MatMatMult(CH, HE, MAT_INITIAL_MATRIX, PETSC_DEFAULT, GD); CHKERRQ(ierr); // GD = CH*invMu*CE

	/** Destroy matrices and vectors. */
	ierr = MatDestroy(&DivE); CHKERRQ(ierr);
	ierr = MatDestroy(&EGrad); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createGD2"
/**
 * createGD2
 * -------
 * Create the matrix GD, the eps^-1 grad(div) operator on E fields.
 */
PetscErrorCode createGD2(Mat *GD, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Set up the matrix DivE, the divergence operator on E fields. */
	Mat DivE;
	ierr = createDivE(&DivE, gi); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nDivE\n"); CHKERRQ(ierr);
	   ierr = MatView(DivE, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Set up the matrix EGrad, the gradient operator generating E fields. */
	Mat EGrad;
	ierr = createEGrad(&EGrad, gi); CHKERRQ(ierr);

	/** Set the inverse of the permittivity vector at nodes, and left-scale DivE by invEps. */
	Vec invEps;
	//ierr = create_epsNode(&invEpsNode, gi); CHKERRQ(ierr);
	//ierr = createFieldArray(&invEps, set_eps_at, gi); CHKERRQ(ierr);
	ierr = createVecHDF5(&invEps, "/eps", gi); CHKERRQ(ierr);
	ierr = VecReciprocal(invEps); CHKERRQ(ierr);
	ierr = MatDiagonalScale(EGrad, invEps, PETSC_NULL); CHKERRQ(ierr);
	ierr = VecDestroy(&invEps); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nEGrad\n"); CHKERRQ(ierr);
	   ierr = MatView(EGrad, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Create the matrix GD. */
	/** Below, 11 is the maximum number of nonzero elements in a diagonal portion of 
	  a local submatrix. e.g., if GD is 9-by-9 and distributed among 3 processors, 
	  entire row (0,1,2) compose a submatrix in processsor 0, and row (3,4,5) in 
	  processor 1, row (6,7,8) in processor 2.  In processor 1, the diagonal portion 
	  means 3-by-3 square matrix at the diagonal, which is composed of row (3,4,5) and 
	  column (3,4,5).  Each row of GD corresponds to one of Ex, Ey, Ez component of 
	  some cell in the grid.

	  When GD is multiplied to a vector x, which has 3*Nx*Ny*Nz E field components, a 
	  row of GD generates an output E field component out of 11 input E field 
	  components; each output E field component is calculated as the gradient between the 
	  scalar potentials at two nodal points, and each scalar potential value is calculated 
	  as the divergence at a point, which involves 6 E field components.  So, the output E 
	  field component is calculated out of 6 + 6 input E field components, but one component
	  is shared between the two sets of 6 components, and therefore the output E field 
	  component comes from 6 + 6 - 1 = 11 input E field components.

	  If the cell containing the output E field component is in interior of a local 
	  portion of the Yee's grid, then all 11 input E field components can be in the diagonal
	  portion of a local submatrix.

	  On the other hand, if the cell is at a boundary of a local grid, then some of 11 
	  input E field components can lie outside the local grid.  It is obivous that 3 components
	  that are in the same cell as the output E field component are guaranteed to be in the same
	  local grid, but the rest 11 - 3 = 8 components can lie outside.  Therefore 8 input E field 
	  components can be in the off-diagonal portion of the submatrix. */

	/** Set up the matrix GD. */
	ierr = MatMatMult(EGrad, DivE, MAT_INITIAL_MATRIX, 11.0/(2.0+6.0), GD); CHKERRQ(ierr); // GD = invEps*EGrad*DivE
	//ierr = MatMatMult(CH, HE, MAT_INITIAL_MATRIX, PETSC_DEFAULT, GD); CHKERRQ(ierr); // GD = CH*invMu*CE

	/** Destroy matrices and vectors. */
	ierr = MatDestroy(&DivE); CHKERRQ(ierr);
	ierr = MatDestroy(&EGrad); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "createGD3"
/**
 * createGD3
 * -------
 * Create the matrix GD, the grad(eps^-2 div) operator on E fields.
 */
PetscErrorCode createGD3(Mat *GD, GridInfo gi)
{
/** Need to divide by (eps^2 mu) rather than eps^2. */
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Set up the matrix DivE, the divergence operator on E fields. */
	Mat DivE;
	ierr = createDivE(&DivE, gi); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nDivE\n"); CHKERRQ(ierr);
	   ierr = MatView(DivE, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Set up the matrix EGrad, the gradient operator generating E fields. */
	Mat EGrad;
	ierr = createEGrad(&EGrad, gi); CHKERRQ(ierr);

	/** Set the inverse of the permittivity vector at nodes, and left-scale DivE by invEpsNode. */
	Vec invEps2Node;
	//ierr = create_epsNode(&invEps2Node, gi); CHKERRQ(ierr);
	//ierr = createFieldArray(&invEps2Node, set_epsNode_at, gi); CHKERRQ(ierr);
	assert(gi.has_epsNode);
	//ierr = createVecHDF5(&invEps2Node, "/eps_node", gi); CHKERRQ(ierr);
	ierr = createVecPETSc(&invEps2Node, "eps_node", gi); CHKERRQ(ierr);
//ierr = VecView(invEps2Node, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	ierr = VecPointwiseMult(invEps2Node, invEps2Node, invEps2Node); CHKERRQ(ierr);
	ierr = VecReciprocal(invEps2Node); CHKERRQ(ierr);
	ierr = MatDiagonalScale(DivE, invEps2Node, PETSC_NULL); CHKERRQ(ierr);
	ierr = VecDestroy(&invEps2Node); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nEGrad\n"); CHKERRQ(ierr);
	   ierr = MatView(EGrad, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Create the matrix GD. */
	/** Below, 11 is the maximum number of nonzero elements in a diagonal portion of 
	  a local submatrix. e.g., if GD is 9-by-9 and distributed among 3 processors, 
	  entire row (0,1,2) compose a submatrix in processsor 0, and row (3,4,5) in 
	  processor 1, row (6,7,8) in processor 2.  In processor 1, the diagonal portion 
	  means 3-by-3 square matrix at the diagonal, which is composed of row (3,4,5) and 
	  column (3,4,5).  Each row of GD corresponds to one of Ex, Ey, Ez component of 
	  some cell in the grid.

	  When GD is multiplied to a vector x, which has 3*Nx*Ny*Nz E field components, a 
	  row of GD generates an output E field component out of 11 input E field 
	  components; each output E field component is calculated as the gradient between the 
	  scalar potentials at two nodal points, and each scalar potential value is calculated 
	  as the divergence at a point, which involves 6 E field components.  So, the output E 
	  field component is calculated out of 6 + 6 input E field components, but one component
	  is shared between the two sets of 6 components, and therefore the output E field 
	  component comes from 6 + 6 - 1 = 11 input E field components.

	  If the cell containing the output E field component is in interior of a local 
	  portion of the Yee's grid, then all 11 input E field components can be in the diagonal
	  portion of a local submatrix.

	  On the other hand, if the cell is at a boundary of a local grid, then some of 11 
	  input E field components can lie outside the local grid.  It is obivous that 3 components
	  that are in the same cell as the output E field component are guaranteed to be in the same
	  local grid, but the rest 11 - 3 = 8 components can lie outside.  Therefore 8 input E field 
	  components can be in the off-diagonal portion of the submatrix. */

	/** Set up the matrix GD. */
	ierr = MatMatMult(EGrad, DivE, MAT_INITIAL_MATRIX, 11.0/(2.0+6.0), GD); CHKERRQ(ierr); // GD = EGrad*invEpsNode*DivE
	//ierr = MatMatMult(CH, HE, MAT_INITIAL_MATRIX, PETSC_DEFAULT, GD); CHKERRQ(ierr); // GD = CH*invMu*CE

	/** Destroy matrices and vectors. */
	ierr = MatDestroy(&DivE); CHKERRQ(ierr);
	ierr = MatDestroy(&EGrad); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "createDG"
/**
 * createDG
 * -------
 * Create the matrix DG, the div(eps grad) operator on a scalar potential.
 */
PetscErrorCode createDG(Mat *DG, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Set up the matrix DivE, the divergence operator on E fields. */
	Mat DivE;
	ierr = createDivE(&DivE, gi); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nDivE\n"); CHKERRQ(ierr);
	   ierr = MatView(DivE, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Set the permittivity vector at edges, and right-scale DivE by eps. */
	Vec eps;
	//ierr = create_eps(&eps, gi); CHKERRQ(ierr);
	//ierr = createFieldArray(&eps, set_eps_at, gi); CHKERRQ(ierr);
	ierr = createVecHDF5(&eps, "/eps", gi); CHKERRQ(ierr);
	ierr = MatDiagonalScale(DivE, PETSC_NULL, eps); CHKERRQ(ierr);

	/** Set up the matrix EGrad, the gradient operator generating E fields. */
	Mat EGrad;
	ierr = createEGrad(&EGrad, gi); CHKERRQ(ierr);

	/*
	   ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\nEGrad\n"); CHKERRQ(ierr);
	   ierr = MatView(EGrad, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
	 */

	/** Create the matrix DG. */
	/*
	   ierr = MatCreate(PETSC_COMM_WORLD, DG); CHKERRQ(ierr);
	   ierr = MatSetSizes(*DG, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	   ierr = MatSetType(*DG, MATRIX_TYPE); CHKERRQ(ierr);
	   ierr = MatSetFromOptions(*DG);
	 */

	/** Below, 7 is the maximum number of nonzero elements in a diagonal portion of 
	  a local submatrix. e.g., if DG is 9-by-9 and distributed among 3 processors, 
	  entire row (0,1,2) compose a submatrix in processsor 0, and row (3,4,5) in 
	  processor 1, row (6,7,8) in processor 2.  In processor 1, the diagonal portion 
	  means 3-by-3 square matrix at the diagonal, which is composed of row (3,4,5) and 
	  column (3,4,5).  Each row of DG corresponds to one of Ex, Ey, Ez component of 
	  some cell in the grid.

	  When DG is multiplied to a vector x, which has scalar potential values at Nx*Ny*Nz 
	  spatial node, a row of DG generates an output scalar potential values out of 7 scalar
	  potential values.  

	  If the cell containing the output nodes is in interior of a local portion of the Yee's grid
	  , then all 7 input nodes can be in the diagonal portion of a local submatrix.

	  On the other hand, if the cell is at a boundary of a local grid, then some of 7 input 
	  spatial nodes can lie outside the local grid.  It is obivous that the center node
	  is guaranteed to be in the same local grid, but the rest 7 - 1 = 6 nodes can lie outside.  
	  Therefore 6 input spatial nodes can be in the off-diagonal portion of the submatrix. */
	/*
	   ierr = MatMPIAIJSetPreallocation(*DG, 7, PETSC_NULL, 6, PETSC_NULL); CHKERRQ(ierr);
	   ierr = MatSeqAIJSetPreallocation(*DG, 7, PETSC_NULL); CHKERRQ(ierr);
	   ierr = MatSetLocalToGlobalMapping(*DG, gi.map, gi.map); CHKERRQ(ierr);
	   ierr = MatSetStencil(*DG, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	 */

	/** Set up the matrix DG. */
	ierr = MatMatMult(DivE, EGrad, MAT_INITIAL_MATRIX, 1.0, DG); CHKERRQ(ierr); // DG = DivE*eps*EGrad

	/** Destroy matrices and vectors. */
	ierr = MatDestroy(&DivE); CHKERRQ(ierr);
	ierr = MatDestroy(&EGrad); CHKERRQ(ierr);
	ierr = VecDestroy(&eps); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "hasPEC"
/**
 * hasPEC
 * ------
 * Set flgPEC PETSC_TRUE if some boundary is PEC; PETSC_FALSE otherwise.
 */
PetscErrorCode hasPEC(PetscBool *flgPEC, GridInfo gi)
{
	PetscFunctionBegin;

	if (gi.bc[Xx][Neg]==PEC || gi.bc[Yy][Neg]==PEC || gi.bc[Zz][Neg]==PEC) *flgPEC = PETSC_TRUE;
	else *flgPEC = PETSC_FALSE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "hasBloch"
/**
 * hasBloch
 * ------
 * Set flgBloch PETSC_TRUE if some boundary is Bloch and associated k_Bloch is nonzero; 
 * PETSC_FALSE otherwise.
 */
PetscErrorCode hasBloch(PetscBool *flgBloch, GridInfo gi)
{
	PetscFunctionBegin;

	if ((gi.bc[Xx][Neg]==Bloch && gi.exp_neg_ikL[Xx]!=1.0) 
			|| (gi.bc[Yy][Neg]==Bloch && gi.exp_neg_ikL[Yy]!=1.0)
			|| (gi.bc[Zz][Neg]==Bloch && gi.exp_neg_ikL[Zz]!=1.0)) *flgBloch = PETSC_TRUE;
	else *flgBloch = PETSC_FALSE;

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "numSymmetrize"
/**
 * numSymmetrize
 * ------
 * Numerically symmetrize a given matrix.  It also evaluate the relative error between
 * A and A^T.
 */
PetscErrorCode numSymmetrize(Mat A)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Mat A_tr;
	ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &A_tr); CHKERRQ(ierr);
	ierr = MatAXPY(A_tr, -1.0, A, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
	//ierr = MatAXPY(A_tr, -1.0, A, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

	PetscReal normA, relerr;
	ierr = MatNorm(A, NORM_INFINITY, &normA); CHKERRQ(ierr);
	ierr = MatNorm(A_tr, NORM_INFINITY, &relerr); CHKERRQ(ierr);
	ierr = MatDestroy(&A_tr); CHKERRQ(ierr);
	relerr /= normA;
	ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "\tsymmetry test: norm(A^T - A)/norm(A) = %e\n",  relerr); CHKERRQ(ierr);

	PetscReal sym_tol = 1e-15;
	if (relerr > sym_tol) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONGSTATE, "ERROR: A is not symmetric enough");

	ierr = MatScale(A, 0.5); CHKERRQ(ierr);
	ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &A_tr); CHKERRQ(ierr);
	ierr = MatAXPY(A, 1.0, A_tr, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
	//ierr = MatAXPY(A, 1.0, A_tr, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

	ierr = MatDestroy(&A_tr); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/**
 * stretch_d
 * ---------
 * Stretches dx, dy, dz with s-parameters.
 * Note that this function can be written to take GridInfo instead of GridInfo* because d_prim and
 * d_dual are pointer variables; even if GridInfo were used and the argument gi is delivered as a 
 * copy, the pointer values d_prim and d_dual are the same as the original, so modifying 
 * d_prim[axis][n] and d_dual[axis][n] modifies the original d_prim and d_dual elements.
 * However, to make sure that users understand that the contents of gi change in this function, this
 * function is written to take GridInfo*.
 */
#undef __FUNCT__
#define __FUNCT__ "stretch_d"
PetscErrorCode stretch_d(GridInfo *gi)
{
	PetscFunctionBegin;

	/** Stretch gi.d_prim and gi.d_dual by gi.s_prim and gi.s_dual. */
	PetscInt axis, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (n = 0; n < gi->N[axis]; ++n) {
			gi->d_prim[axis][n] *= gi->s_prim[axis][n];
			gi->d_dual[axis][n] *= gi->s_dual[axis][n];
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "unstretch_d"
PetscErrorCode unstretch_d(GridInfo *gi)
{
	PetscFunctionBegin;

	/** Recover the original gi.d_prim and gi.d_dual. */
	PetscInt axis, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (n = 0; n < gi->N[axis]; ++n) {
			gi->d_prim[axis][n] = gi->d_prim_orig[axis][n];
			gi->d_dual[axis][n] = gi->d_dual_orig[axis][n];
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "make_d_one"
PetscErrorCode make_d_one(GridInfo *gi)
{
	PetscFunctionBegin;

	/** Stretch gi.d_prim and gi.d_dual by gi.s_prim and gi.s_dual. */
	PetscInt axis, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (n = 0; n < gi->N[axis]; ++n) {
			gi->d_prim[axis][n] = 1.0;
			gi->d_dual[axis][n] = 1.0;
		}
	}

	PetscFunctionReturn(0);
}

/**
 * Modify create_A_and_b() so that the added continuity equation is symmetric.
 */
#undef __FUNCT__
#define __FUNCT__ "create_A_and_b4"
PetscErrorCode create_A_and_b4(Mat *A, Vec *b, Vec *right_precond, Mat *HE, GridInfo gi, TimeStamp *ts)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Vec eps, mu, epsMask; 
	Vec inverse;  // store various inverse vectors
	Vec left_precond, precond;
	Mat CE, CH;  // curl operators on E and H
	Mat CHE; 

	if (gi.verbose_level >= VBMedium) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Create the matrix for %s with %s, preconditioned by %s.\n", FieldTypeName[gi.x_type], PMLTypeName[gi.pml_type], PCTypeName[gi.pc_type]); CHKERRQ(ierr);
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "The matrix is %s, continuity eq %s", (gi.is_symmetric ? "symmetric":"non-symmetric"), (gi.add_conteq ? "added":"not added")); CHKERRQ(ierr);
		if (gi.add_conteq) {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, " with factor %f", gi.factor_conteq); CHKERRQ(ierr);
		}
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, ".\n"); CHKERRQ(ierr);
	}

	ierr = VecDuplicate(gi.vecTemp, &inverse); CHKERRQ(ierr);

	/** Stretch gi.d_prim and gi.d_dual by gi.s_prim and gi.s_dual. */
	if (gi.pml_type == SCPML) {
		ierr = stretch_d(&gi); CHKERRQ(ierr);
	}

	/** Create the permittivity vector. */
	//ierr = create_eps(&eps, gi); CHKERRQ(ierr);
	//ierr = createFieldArray(&eps, set_eps_at, gi); CHKERRQ(ierr);
	//ierr = createVecHDF5(&eps, "/eps", gi); CHKERRQ(ierr);
	ierr = createVecPETSc(&eps, "eps", gi); CHKERRQ(ierr);
	ierr = VecDuplicate(gi.vecTemp, &epsMask); CHKERRQ(ierr);
	ierr = VecCopy(eps, epsMask); CHKERRQ(ierr);
	if (gi.pml_type == UPML) {
		Vec sparamEps;
		//ierr = create_sparamEps(&sparamEps, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&sparamEps, set_sparam_eps_at, gi); CHKERRQ(ierr);
		ierr = VecPointwiseMult(eps, eps, sparamEps); CHKERRQ(ierr);
		ierr = VecDestroy(&sparamEps); CHKERRQ(ierr);
	}
	//ierr = create_epsMask(&epsMask, gi); CHKERRQ(ierr);  // to handle TruePEC objects
	//ierr = createFieldArray(&epsMask, set_epsMask_at, gi); CHKERRQ(ierr);  // to handle TruePEC objects
	ierr = infMaskVec(epsMask, gi); CHKERRQ(ierr);  // to handle TruePEC objects
	ierr = maskInf2One(eps, gi); CHKERRQ(ierr);  // to handle TruePEC objects
	ierr = updateTimeStamp(VBDetail, ts, "eps vector", gi); CHKERRQ(ierr);


	/** Create the permeability vector. */
	//ierr = create_mu(&mu, gi); CHKERRQ(ierr);
	//ierr = createFieldArray(&mu, set_mu_at, gi); CHKERRQ(ierr);
	if (gi.has_mu) {
		ierr = createVecHDF5(&mu, "/mu", gi); CHKERRQ(ierr);
	} else {
		ierr = VecDuplicate(gi.vecTemp, &mu); CHKERRQ(ierr);
		ierr = VecSet(mu, 1.0); CHKERRQ(ierr);
	}
	
	if (gi.pml_type == UPML) {
		Vec sparamMu;
		//ierr = create_sparamMu(&sparamMu, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&sparamMu, set_sparam_mu_at, gi); CHKERRQ(ierr);
		ierr = VecPointwiseMult(mu, mu, sparamMu); CHKERRQ(ierr);
		ierr = VecDestroy(&sparamMu); CHKERRQ(ierr);
	}
	ierr = updateTimeStamp(VBDetail, ts, "mu vector", gi); CHKERRQ(ierr);

	/** Set up the matrix CE, the curl operator on E fields. */
	ierr = createCE(&CE, gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "CE matrix", gi); CHKERRQ(ierr);

	/** Set up the matrix CH, the curl operator on H fields. */
	ierr = createCH(&CH, gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "CH matrix", gi); CHKERRQ(ierr);

	if (gi.x_type == Htype) {
		Mat mat_temp;
		Vec vec_temp;

		mat_temp = CE; CE = CH; CH = mat_temp;
		vec_temp = eps; eps = mu; mu = vec_temp;
	}

	/** Set up the matrix HE, the operator giving H fields from E fields. */
	*HE = CE;
	ierr = VecSet(inverse, 1.0); CHKERRQ(ierr);
	ierr = VecPointwiseDivide(inverse, inverse, mu); CHKERRQ(ierr);
	ierr = MatDiagonalScale(*HE, inverse, PETSC_NULL); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "HE matrix", gi); CHKERRQ(ierr);

	/** Create the matrix CHE, the curl(mu^-1 curl) operator. */
	ierr = createCHE(&CHE, CH, *HE, gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "CHE matrix", gi); CHKERRQ(ierr);
	ierr = MatDestroy(&CH); CHKERRQ(ierr);

	if (!gi.add_conteq) {
		/** Below, isn't *A = CHE the same as A = &CHE?  No.  Remember that A is a return value.  
		  When this function is called, we do:
		  Mat B;
		  ...
		  ierr = create_XXX_A_YYY(&B, ...); CHKERRQ(ierr);
		  The intension of this function call is to fill the memory pointed by &B. *A = CHE fulfills 
		  this intension.
		  On the other hand, if the below line is A = &CHE, it is nothing but changing the value of 
		  the pointer variable A from &B to &CHE.  Therefore nothing is returned to B. */
		*A = CHE;

		/** Create b. */
		//ierr = create_jSrc(b, gi); CHKERRQ(ierr);
		//ierr = createFieldArray(b, set_src_at, gi); CHKERRQ(ierr);
		//ierr = createVecHDF5(b, "/J", gi); CHKERRQ(ierr);
		ierr = createVecPETSc(b, "J", gi); CHKERRQ(ierr);
		ierr = VecScale(*b, -PETSC_i*gi.omega); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "b vector", gi); CHKERRQ(ierr);
	} else {  // currently, add_conteq only works for x_type == Etype
		ierr = createAtemplate(A, gi); CHKERRQ(ierr);
		ierr = MatAXPY(*A, 1.0, CHE, SUBSET_NONZERO_PATTERN); CHKERRQ(ierr);
		ierr = MatDestroy(&CHE); CHKERRQ(ierr);

		/** Create the gradient-divergence operator. */
		Mat GD;
		//ierr = createGD(&GD, gi); CHKERRQ(ierr);
		//ierr = createGD2(&GD, gi); CHKERRQ(ierr);
		ierr = createGD3(&GD, gi); CHKERRQ(ierr);
//ierr = MatView(GD, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
		ierr = MatDiagonalScale(GD, eps, PETSC_NULL); CHKERRQ(ierr);  // GD = eps grad[eps^-2 div()]; only for createGD3
/*
ierr = VecSet(inverse, 1.0); CHKERRQ(ierr);
ierr = VecPointwiseDivide(inverse, inverse, eps); CHKERRQ(ierr);
ierr = MatDiagonalScale(GD, inverse, PETSC_NULL); CHKERRQ(ierr);
*/
		ierr = updateTimeStamp(VBDetail, ts, "GD matrix", gi); CHKERRQ(ierr);

		/** Create b. */
		Vec b_aug;
		ierr = VecDuplicate(gi.vecTemp, &b_aug); CHKERRQ(ierr);
		//ierr = create_jSrc(b, gi); CHKERRQ(ierr);
		//ierr = createFieldArray(b, set_src_at, gi); CHKERRQ(ierr);  // b = J
		//ierr = createVecHDF5(b, "/J", gi); CHKERRQ(ierr);  // b = J
		ierr = createVecPETSc(b, "J", gi); CHKERRQ(ierr);  // b = J
		ierr = VecCopy(*b, b_aug); CHKERRQ(ierr);  // b_aug = J
		ierr = VecScale(b_aug, gi.factor_conteq*PETSC_i/gi.omega); CHKERRQ(ierr);  // b_aug = s*(i/omega)*J
		ierr = VecScale(*b, -PETSC_i*gi.omega); CHKERRQ(ierr);  // b = -i*omega*J
		ierr = MatMultAdd(GD, b_aug, *b, *b); CHKERRQ(ierr);  // b = -i*omega*J + GD * s*(i/omega)*J
		ierr = VecDestroy(&b_aug); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "b vector", gi); CHKERRQ(ierr);

		ierr = MatDiagonalScale(GD, PETSC_NULL, eps); CHKERRQ(ierr);
		ierr = MatAXPY(*A, gi.factor_conteq, GD, SUBSET_NONZERO_PATTERN); CHKERRQ(ierr);
		ierr = MatDestroy(&GD); CHKERRQ(ierr);
	}

//ierr = MatView(*A, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

	ierr = MatDiagonalScale(*A, epsMask, epsMask); CHKERRQ(ierr);  // omega^2*mu*eps is not subtracted yet, so the diagonal entries will be nonzero
	ierr = VecPointwiseMult(*b, epsMask, *b); CHKERRQ(ierr);  // force E = 0 on TruePEC.  comment this line to allow source on TruePEC

	if (!gi.solve_eigen) {
		Vec negW2Eps = eps;
		ierr = VecScale(negW2Eps, -gi.omega*gi.omega); CHKERRQ(ierr);
		ierr = MatDiagonalSet(*A, negW2Eps, ADD_VALUES); CHKERRQ(ierr);
	}
	ierr = updateTimeStamp(VBDetail, ts, "A matrix", gi); CHKERRQ(ierr);

	/** Scale the matrix HE. */
	if (gi.x_type == Etype) {
		ierr = MatScale(*HE, -1/gi.omega/PETSC_i); CHKERRQ(ierr);  // HE = [(-i*omega)^-1] * invMu*CH
	} else {
		ierr = MatScale(*HE, 1/gi.omega/PETSC_i); CHKERRQ(ierr);  // HE = [(i*omega)^-1] * invEps * CH, where HE is in fact EH
	}
	ierr = updateTimeStamp(VBDetail, ts, "HE matrix scaling", gi); CHKERRQ(ierr);

	/** Create the left and right preconditioner. */
	/** Set the left preconditioner. */
	ierr = VecDuplicate(gi.vecTemp, &left_precond); CHKERRQ(ierr);
	ierr = VecSet(left_precond, 1.0); CHKERRQ(ierr);

	/** Set the right preconditioner. */
	ierr = VecDuplicate(gi.vecTemp, right_precond); CHKERRQ(ierr);
	ierr = VecSet(*right_precond, 1.0); CHKERRQ(ierr);

	if (gi.is_symmetric) {  // currently, is_symmetric only works for x_type == Etype
		/** original eq: A0 x = b.  The matrix 
		  diag(1/sqrt(Epec)) diag(sqrt(LS)) A0 diag(1/sqrt(LS)) diag(sqrt(Epec))
		  is symmetric. */

		/** Calculate the diagonal matrix to be multiplied to the left and right of the 
		  matrix A for symmetrizing A. */
		Vec sqrtLS, dS;
		//ierr = create_dLf(&sqrtLS, Etype, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&sqrtLS, set_dLe_at, gi); CHKERRQ(ierr);
		//ierr = create_dSf(&dS, Etype, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&dS, set_dSe_at, gi); CHKERRQ(ierr);
		ierr = VecPointwiseMult(sqrtLS, sqrtLS, dS); CHKERRQ(ierr);
		ierr = VecDestroy(&dS); CHKERRQ(ierr);
		ierr = sqrtVec(sqrtLS, gi); CHKERRQ(ierr);

		ierr = VecPointwiseMult(*right_precond, *right_precond, sqrtLS); CHKERRQ(ierr);
		ierr = VecDestroy(&sqrtLS); CHKERRQ(ierr);

		Vec sqrtScaleEpec;
		//ierr = create_scaleEpec(&sqrtScaleEpec, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&sqrtScaleEpec, set_scale_Epec_at, gi); CHKERRQ(ierr);
		ierr = sqrtVec(sqrtScaleEpec, gi); CHKERRQ(ierr);

		ierr = VecPointwiseDivide(*right_precond, *right_precond, sqrtScaleEpec); CHKERRQ(ierr);
		ierr = VecDestroy(&sqrtScaleEpec); CHKERRQ(ierr);
		ierr = VecPointwiseDivide(left_precond, left_precond, *right_precond); CHKERRQ(ierr);
	}

	/** Apply the preconditioner. Only one type of preconditioners is applied. */
	if (gi.pc_type == PCSparam) {  
		Vec sparamL, sparamS;
		//ierr = create_sparamLf(&sparamL, Etype, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&sparamL, set_sparamLe_at, gi); CHKERRQ(ierr);
		//ierr = create_sparamSf(&sparamS, Etype, gi); CHKERRQ(ierr);
		ierr = createFieldArray(&sparamS, set_sparamSe_at, gi); CHKERRQ(ierr);
		if (!gi.is_symmetric) {  // Ascpml = diag(1/sparamS) Aupml diag(sparamL)
			ierr = VecPointwiseMult(left_precond, left_precond, sparamS); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(*right_precond, *right_precond, sparamL); CHKERRQ(ierr);
		} else {  // diag(sqrt(sparamL/sparamS)) Aupml diag(sqrt(sparamL/sparamS))
			Vec sqrtLoverS;
			ierr = VecDuplicate(gi.vecTemp, &sqrtLoverS); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(sqrtLoverS, sparamL, sparamS); CHKERRQ(ierr);
			ierr = sqrtVec(sqrtLoverS, gi); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(left_precond, left_precond, sqrtLoverS); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(*right_precond, *right_precond, sqrtLoverS); CHKERRQ(ierr);
			ierr = VecDestroy(&sqrtLoverS); CHKERRQ(ierr);
		}

		ierr = VecDestroy(&sparamL); CHKERRQ(ierr);
		ierr = VecDestroy(&sparamS); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "s-parameter preconditioner", gi); CHKERRQ(ierr);
	} else if (gi.pc_type == PCEps) {
		//ierr = create_eps(&precond, gi); CHKERRQ(ierr);
		//ierr = createFieldArray(&precond, set_eps_at, gi); CHKERRQ(ierr);
		ierr = createVecHDF5(&precond, "/eps", gi); CHKERRQ(ierr);
		if (!gi.is_symmetric) {
			ierr = VecPointwiseMult(left_precond, left_precond, precond); CHKERRQ(ierr);
		} else {
			ierr = sqrtVec(precond, gi); CHKERRQ(ierr);
			ierr = VecPointwiseMult(left_precond, left_precond, precond); CHKERRQ(ierr);
			ierr = VecPointwiseMult(*right_precond, *right_precond, precond); CHKERRQ(ierr);
		}
		ierr = VecDestroy(&precond); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "eps preconditioner", gi); CHKERRQ(ierr);
	} else if (gi.pc_type == PCJacobi) {
		ierr = VecDuplicate(gi.vecTemp, &precond); CHKERRQ(ierr);
		ierr = MatGetDiagonal(*A, precond); CHKERRQ(ierr);
		if (!gi.is_symmetric) {
			ierr = VecPointwiseMult(left_precond, left_precond, precond); CHKERRQ(ierr);
		} else {
			ierr = sqrtVec(precond, gi); CHKERRQ(ierr);
			ierr = VecPointwiseMult(left_precond, left_precond, precond); CHKERRQ(ierr);
			ierr = VecPointwiseMult(*right_precond, *right_precond, precond); CHKERRQ(ierr);
		}
		ierr = VecDestroy(&precond); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "Jacobi preconditioner", gi); CHKERRQ(ierr);
	} else {
		assert(gi.pc_type == PCIdentity);
	}

	Vec inv_left, inv_right; 
	ierr = VecDuplicate(gi.vecTemp, &inv_left); CHKERRQ(ierr);
	ierr = VecSet(inv_left, 1.0); CHKERRQ(ierr);
	ierr = VecPointwiseDivide(inv_left, inv_left, left_precond); CHKERRQ(ierr);

	ierr = VecDuplicate(gi.vecTemp, &inv_right); CHKERRQ(ierr);
	ierr = VecSet(inv_right, 1.0); CHKERRQ(ierr);
	ierr = VecPointwiseDivide(inv_right, inv_right, *right_precond); CHKERRQ(ierr);

	/** The below is true because right_precond = (left_precond)^-1. */
	ierr = MatDiagonalScale(*A, inv_left, inv_right); CHKERRQ(ierr);
	ierr = VecPointwiseMult(*b, inv_left, *b); CHKERRQ(ierr);
	ierr = VecDestroy(&inv_left); CHKERRQ(ierr);
	ierr = VecDestroy(&inv_right); CHKERRQ(ierr);

	/** Recover the original d_dual and d_prim. */
	if (gi.pml_type == SCPML) {
		ierr = unstretch_d(&gi); CHKERRQ(ierr);
	}

	ierr = VecDestroy(&eps); CHKERRQ(ierr);
	ierr = VecDestroy(&mu); CHKERRQ(ierr);
	ierr = VecDestroy(&epsMask); CHKERRQ(ierr);
	ierr = VecDestroy(&inverse); CHKERRQ(ierr);
	ierr = VecDestroy(&left_precond); CHKERRQ(ierr);

	PetscBool flgBloch;
	ierr = hasBloch(&flgBloch, gi); CHKERRQ(ierr);
	if (gi.is_symmetric && !flgBloch) {
		ierr = numSymmetrize(*A); CHKERRQ(ierr);
		ierr = MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE); CHKERRQ(ierr);
		ierr = MatSetOption(*A, MAT_HERMITIAN, PETSC_FALSE); CHKERRQ(ierr);
	} else {
		ierr = MatSetOption(*A, MAT_SYMMETRIC, PETSC_FALSE); CHKERRQ(ierr);
		ierr = MatSetOption(*A, MAT_HERMITIAN, PETSC_FALSE); CHKERRQ(ierr);
	}

	PetscFunctionReturn(0);
}
