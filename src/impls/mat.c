#include <assert.h>
#include "mat.h"

//ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "I'm here!\n"); CHKERRQ(ierr);
//ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//ierr = MatView(mat, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

const char * const FieldTypeName[] = {"E", "H"};
const char * const GridTypeName[] = {"primary", "dual"};
const char * const PMLTypeName[] = {"SC-PML", "UPML"};
const char * const PCTypeName[] = {"identity", "s-factor", "eps", "Jacobi"};

//const char * const KrylovTypeName[] = {"BiCG", "QMR"};

#undef __FUNCT__
#define __FUNCT__ "setDp"
/**
 * setDp
 * ------------
 * For a matrix row indexed by (w, i, j, k), set the forward (s==Pos) or backward (s==Neg)
 * difference of the p-component of the field in the v-direction, with extra scale multiplied.
 */
PetscErrorCode setDp(Mat A, Sign s, Axis w, PetscInt i, PetscInt j, PetscInt k, Axis p, Axis v, PetscScalar scale, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	/** Set dv. */
	PetscInt coord[] = {i, j, k};
	PetscInt ind = coord[v];
	PetscScalar dv = gi.dl[v][s][ind];  // for forward (s==Pos) difference, use dl at dual grid locations

	/** Set the row and column indices of the matrix elements to set. */
	MatStencil indGw;  // grid point indices of Gw (mapped to row index of A)
	MatStencil indFp[2];  // current and next grid point indices of Fp (mapped to column indices of A).  The next grid point can be either in +d or -d direction

	indGw.c = w;
	indGw.i = i;
	indGw.j = j;
	indGw.k = k;

	indFp[0].c = p;
	indFp[0].i = coord[Xx];
	indFp[0].j = coord[Yy];
	indFp[0].k = coord[Zz];

	if (s == Pos) {
		++coord[v];
	} else {
		assert(s == Neg);
		--coord[v];
	}
	indFp[1].c = p;
	indFp[1].i = coord[Xx];
	indFp[1].j = coord[Yy];
	indFp[1].k = coord[Zz];

	/** Set Nv. */
	PetscInt Nv = gi.N[v];

	PetscScalar dFp[2];
	PetscInt num_dFp = 2;

	/** Two matrix elements in a single row to be set at once. */
	if (s == Pos) {
		dFp[0] = -scale/dv; dFp[1] = scale/dv;  // forward difference
	} else {
		dFp[0] = scale/dv; dFp[1] = -scale/dv;  // backward difference
	}

	/** Handle boundary conditions. */
	if (s == Pos) {  // forward difference
		if (ind == Nv-1) {
			if (gi.bc[v] == Bloch) {
				dFp[1] *= gi.exp_neg_ikL[v];
			} else {
				dFp[1] = 0.0;
			}
		}
	} else {  // backward difference
		assert(s == Neg);
		if (ind == 0) {
			if ((gi.ge == Prim && gi.bc[v] == PMC) || (gi.ge == Dual && gi.bc[v] == PEC)) {
				dFp[0] *= 2.0;
			}

			if (gi.bc[v] == Bloch) {
				dFp[1] /= gi.exp_neg_ikL[v];
			} else {
				dFp[1] = 0.0;
			}
		}
	}

	/** Below, ADD_VALUES is used instead of INSERT_VALUES to deal with cases of 
	  gi.bc[Pp]==gi.bc[Pp]==Bloch and Nv==1.  In such a case, v==0 and v==Nv-1 coincide, 
	  so inserting dFp[1] after dFp[0] overwrites dFp[0], which is not what we want.  
	  On the other hand, if we add dFp[1] to dFp[0], it is equivalent to insert 
	  -scale/dv + scale/dv = 0.0, and this is what should be done. */
	ierr = MatSetValuesStencil(A, 1, &indGw, num_dFp, indFp, dFp, ADD_VALUES); CHKERRQ(ierr);  // Gw <-- scale * (d/dv)Fp

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
PetscErrorCode setDivF(Mat DivF, GridType gtype, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Sign s = (Sign) gtype;  // Neg for gtype==Prim, Pos for gtype==Dual

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {
					/** In theory DivF is an N x 3N matrix, but it is much easier to make it a 
					  3N x 3N square matrix with the distributted array (DA) of PETSc.  To that end 
					  we leave every 2nd and 3rd rows empty.  In other words, even though div(F) is 
					  a scalar, we make it a vector quantity such that 
					  [div(F)]x = div(F)
					  [div(F)]y = 0
					  [div(F)]z = 0
					 */
					Axis w = (Axis) axis;
					ierr = setDp(DivF, s, Xx, i, j, k, w, w, 1.0, gi); CHKERRQ(ierr);
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

	Vec maskE;
	if (gi.ge == Prim) {
		ierr = createFieldArray(&maskE, set_mask_prim_at, gi);
	} else {
		ierr = createFieldArray(&maskE, set_mask_dual_at, gi);
	}

	ierr = MatCreate(PETSC_COMM_WORLD, DivE); CHKERRQ(ierr);
	ierr = MatSetSizes(*DivE, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*DivE, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*DivE);
	ierr = MatMPIAIJSetPreallocation(*DivE, 6, PETSC_NULL, 3, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*DivE, 6, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*DivE, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*DivE, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setDivF(*DivE, gi.ge, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*DivE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*DivE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = MatDiagonalScale(*DivE, PETSC_NULL, maskE); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createDivH"
/**
 * createDivH
 * --------
 * Create the matrix DivH, the divergence operator on H fields.
 */
PetscErrorCode createDivH(Mat *DivH, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Vec maskH;
	if (gi.ge == Prim) {
		ierr = createFieldArray(&maskH, set_mask_dual_at, gi);
	} else {
		ierr = createFieldArray(&maskH, set_mask_prim_at, gi);
	}

	ierr = MatCreate(PETSC_COMM_WORLD, DivH); CHKERRQ(ierr);
	ierr = MatSetSizes(*DivH, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*DivH, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*DivH);
	ierr = MatMPIAIJSetPreallocation(*DivH, 6, PETSC_NULL, 3, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*DivH, 6, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*DivH, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*DivH, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setDivF(*DivH, (GridType)((gi.ge+1) % Ngt), gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*DivH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*DivH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = MatDiagonalScale(*DivH, PETSC_NULL, maskH); CHKERRQ(ierr);

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
PetscErrorCode setFGrad(Mat FGrad, GridType gtype, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Sign s = (Sign)((gtype+1) % Ngt);  // Pos for gtype==Prim, Neg for gtype==Dual

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {
					/** In theory FGrad is a 3N x N matrix, but it is much easier to make it a 
					  3N x 3N square matrix with the distributted array (DA) of PETSc.  To that end 
					  we leave every 2nd and 3rd columns of FGrad empty.  This is also consistent 
					  with the operator composition such as EGrad * DivE for grad(div(E)), because 
					  DivE is constructed so that every 2nd and 3rd rows are empty.  In other words, 
					  even though phi in F = grad(phi) is a scalar, we take a vector G such that 
					  Gx = phi
					  Gy = 0
					  Gz = 0
					 */
					Axis w = (Axis) axis;
					ierr = setDp(FGrad, s, w, i, j, k, Xx, w, 1.0, gi); CHKERRQ(ierr);
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

	Vec maskE;
	if (gi.ge == Prim) {
		ierr = createFieldArray(&maskE, set_mask_prim_at, gi);
	} else {
		ierr = createFieldArray(&maskE, set_mask_dual_at, gi);
	}

	ierr = MatCreate(PETSC_COMM_WORLD, EGrad); CHKERRQ(ierr);
	ierr = MatSetSizes(*EGrad, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*EGrad, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*EGrad);
	ierr = MatMPIAIJSetPreallocation(*EGrad, 2, PETSC_NULL, 1, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*EGrad, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*EGrad, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*EGrad, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setFGrad(*EGrad, gi.ge, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*EGrad, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*EGrad, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatDiagonalScale(*EGrad, maskE, PETSC_NULL); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createHGrad"
/**
 * createHGrad
 * --------
 * Create the matrix HGrad, the gradient operator generating H fields.
 */
PetscErrorCode createHGrad(Mat *HGrad, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Vec maskH;
	if (gi.ge == Prim) {
		ierr = createFieldArray(&maskH, set_mask_dual_at, gi);
	} else {
		ierr = createFieldArray(&maskH, set_mask_prim_at, gi);
	}

	ierr = MatCreate(PETSC_COMM_WORLD, HGrad); CHKERRQ(ierr);
	ierr = MatSetSizes(*HGrad, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*HGrad, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*HGrad);
	ierr = MatMPIAIJSetPreallocation(*HGrad, 2, PETSC_NULL, 1, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*HGrad, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*HGrad, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*HGrad, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setFGrad(*HGrad, gi.ge, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*HGrad, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*HGrad, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatDiagonalScale(*HGrad, maskH, PETSC_NULL); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "setCF"
/**
 * setCF
 * -----
 * Set up the curl(F) operator matrix CF for given F == E or H.
 */
PetscErrorCode setCF(Mat CF, GridType gtype, GridInfo gi)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;
	
	Sign s = (Sign)((gtype+1) % Ngt);  // Pos for gtype==Prim, Neg for gtype==Dual

	/** Get corners and widths of Yee's grid included in this proces. */
	PetscInt ox, oy, oz;  // coordinates of beginning corner of Yee's grid in this process
	PetscInt nx, ny, nz;  // local widths of Yee's grid in this process
	ierr = DMDAGetCorners(gi.da, &ox, &oy, &oz, &nx, &ny, &nz); CHKERRQ(ierr);

	PetscInt i, j, k, axis;  // x, y, z indices of grid point
	for (k = oz; k < oz+nz; ++k) {
		for (j = oy; j < oy+ny; ++j) {
			for (i = ox; i < ox+nx; ++i) {
				for (axis = 0; axis < Naxis; ++axis) {  // direction of curl
					Axis n = (Axis) axis;
					Axis h = (Axis)((axis+1) % Naxis);  // horizontal axis
					Axis v = (Axis)((axis+2) % Naxis);  // vertical axis

					ierr = setDp(CF, s, n, i, j, k, v, h, 1.0, gi); CHKERRQ(ierr);
					ierr = setDp(CF, s, n, i, j, k, h, v, -1.0, gi); CHKERRQ(ierr);
				}
			}
		}
	}

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

	Vec maskE, maskH;
	if (gi.ge == Prim) {
		ierr = createFieldArray(&maskE, set_mask_prim_at, gi);
		ierr = createFieldArray(&maskH, set_mask_dual_at, gi);
	} else {
		ierr = createFieldArray(&maskE, set_mask_dual_at, gi);
		ierr = createFieldArray(&maskH, set_mask_prim_at, gi);
	}

	ierr = MatCreate(PETSC_COMM_WORLD, CE); CHKERRQ(ierr);
	ierr = MatSetSizes(*CE, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*CE, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*CE);
	ierr = MatMPIAIJSetPreallocation(*CE, 4, PETSC_NULL, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*CE, 4, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*CE, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*CE, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setCF(*CE, gi.ge, gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*CE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*CE, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = MatDiagonalScale(*CE, maskH, maskE); CHKERRQ(ierr);

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

	Vec maskE, maskH;
	if (gi.ge == Prim) {
		ierr = createFieldArray(&maskE, set_mask_prim_at, gi);
		ierr = createFieldArray(&maskH, set_mask_dual_at, gi);
	} else {
		ierr = createFieldArray(&maskE, set_mask_dual_at, gi);
		ierr = createFieldArray(&maskH, set_mask_prim_at, gi);
	}

	ierr = MatCreate(PETSC_COMM_WORLD, CH); CHKERRQ(ierr);
	ierr = MatSetSizes(*CH, gi.Nlocal_tot, gi.Nlocal_tot, PETSC_DETERMINE, PETSC_DETERMINE); CHKERRQ(ierr);
	ierr = MatSetType(*CH, MATRIX_TYPE); CHKERRQ(ierr);
	ierr = MatSetFromOptions(*CH);
	ierr = MatMPIAIJSetPreallocation(*CH, 4, PETSC_NULL, 2, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSeqAIJSetPreallocation(*CH, 4, PETSC_NULL); CHKERRQ(ierr);
	ierr = MatSetLocalToGlobalMapping(*CH, gi.map, gi.map); CHKERRQ(ierr);
	ierr = MatSetStencil(*CH, Naxis, gi.Nlocal_g, gi.start_g, Naxis); CHKERRQ(ierr);
	ierr = setCF(*CH, (GridType)((gi.ge+1) % Ngt), gi); CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*CH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*CH, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

	ierr = MatDiagonalScale(*CH, maskE, maskH); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "createCGF"
/**
 * createCGF
 * -------
 * Create the matrix CGF, the curl((mu or eps)^-1 curl) operator on E- or H-fields.
 */

PetscErrorCode createCGF(Mat *CGF, Mat CG, Mat GF, GridInfo gi)
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
	  a local submatrix. e.g., if G=H and F=E, CHE is 9-by-9 and distributed among 3 processors, 
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
	ierr = MatMatMult(CG, GF, MAT_INITIAL_MATRIX, 13.0/(4.0+4.0), CGF); CHKERRQ(ierr); // CGF = CG*(invMu or invEps)*CF
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
PetscErrorCode setAtemplate_at(Mat Atemplate, Axis Pp, PetscInt i, PetscInt j, PetscInt k, GridInfo gi)
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

	if ((gi.x_type==Etype && gi.ge==Prim) || (gi.x_type==Htype && gi.ge==Dual)) {
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
PetscErrorCode setAtemplate(Mat Atemplate, GridInfo gi)
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
					ierr = setAtemplate_at(Atemplate, (Axis)axis, i, j, k, gi);
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
	ierr = setAtemplate(*Atemplate, gi); CHKERRQ(ierr);
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
#define __FUNCT__ "createGDsym"
/**
 * createGDsym
 * -------
 * Create the symmetric matrix GD, the grad(mu^-1 eps^-2 div) operator on the E-field, or the 
 * grad(eps^-1 mu^-2 div) operator on the H-field.
 */
PetscErrorCode createGDsym(Mat *GD, GridInfo gi)
{
/** Need to divide by (eps^2 mu) rather than eps^2. */
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Mat DivF, FGrad;
	Vec epsNode, muNode, invNode;

	/** Set the inverse of the elementwise product of eps and mu vectors at nodes. */
	ierr = createVecPETSc(&epsNode, "eps_node", gi); CHKERRQ(ierr);
	if (gi.has_mu) {
		ierr = createVecPETSc(&muNode, "mu_node", gi); CHKERRQ(ierr);
	} else {
		ierr = VecDuplicate(gi.vecTemp, &muNode); CHKERRQ(ierr);
		ierr = VecSet(muNode, 1.0); CHKERRQ(ierr);
	}

	/** Set up the matrix DivF and FGrad, the divergence on F and gradient operator generating F. */
	ierr = VecDuplicate(gi.vecTemp, &invNode); CHKERRQ(ierr);
	ierr = VecPointwiseMult(invNode, epsNode, muNode); CHKERRQ(ierr);
	if (gi.x_type == Etype) {
		ierr = VecPointwiseMult(invNode, epsNode, invNode); CHKERRQ(ierr);
		ierr = createDivE(&DivF, gi); CHKERRQ(ierr);
		ierr = createEGrad(&FGrad, gi); CHKERRQ(ierr);
	} else {
		ierr = VecPointwiseMult(invNode, muNode, invNode); CHKERRQ(ierr);
		ierr = createDivH(&DivF, gi); CHKERRQ(ierr);
		ierr = createHGrad(&FGrad, gi); CHKERRQ(ierr);
	}

	ierr = VecReciprocal(invNode); CHKERRQ(ierr);
	ierr = MatDiagonalScale(DivF, invNode, PETSC_NULL); CHKERRQ(ierr);
	ierr = VecDestroy(&epsNode); CHKERRQ(ierr);
	ierr = VecDestroy(&muNode); CHKERRQ(ierr);
	ierr = VecDestroy(&invNode); CHKERRQ(ierr);

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
	ierr = MatMatMult(FGrad, DivF, MAT_INITIAL_MATRIX, 11.0/(2.0+6.0), GD); CHKERRQ(ierr); // GD = EGrad*invEpsNode*DivE
	//ierr = MatMatMult(CH, HE, MAT_INITIAL_MATRIX, PETSC_DEFAULT, GD); CHKERRQ(ierr); // GD = CH*invMu*CE

	/** Destroy matrices and vectors. */
	ierr = MatDestroy(&DivF); CHKERRQ(ierr);
	ierr = MatDestroy(&FGrad); CHKERRQ(ierr);

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

	if (gi.bc[Xx]==PEC || gi.bc[Yy]==PEC || gi.bc[Zz]==PEC) *flgPEC = PETSC_TRUE;
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

	if ((gi.bc[Xx]==Bloch && gi.exp_neg_ikL[Xx]!=1.0) 
			|| (gi.bc[Yy]==Bloch && gi.exp_neg_ikL[Yy]!=1.0)
			|| (gi.bc[Zz]==Bloch && gi.exp_neg_ikL[Zz]!=1.0)) *flgBloch = PETSC_TRUE;
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
 * Stretches dx, dy, dz with s-factors.
 * Note that this function can be written to take GridInfo instead of GridInfo* because dl is 
 * a pointer variable; even if GridInfo were used and the argument gi is delivered as a 
 * copy, the pointer value dl is the same as the original, so modifying dl[axis][gt][n] and 
 * dl[axis][gt][n] modifies the original dl elements.
 * However, to make sure that users understand that the contents of gi change in this function, this
 * function is written to take GridInfo*.
 */
#undef __FUNCT__
#define __FUNCT__ "stretch_d"
PetscErrorCode stretch_d(GridInfo *gi)
{
	PetscFunctionBegin;

	/** Stretch gi.dl by gi.s_factor. */
	PetscInt axis, gt, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (gt = 0; gt < Ngt; ++gt) {
			for (n = 0; n < gi->N[axis]; ++n) {
				gi->dl[axis][gt][n] *= gi->s_factor[axis][gt][n];
			}
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "unstretch_d"
PetscErrorCode unstretch_d(GridInfo *gi)
{
	PetscFunctionBegin;

	/** Recover the original gi.dl. */
	PetscInt axis, gt, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (gt = 0; gt < Ngt; ++gt) {
			for (n = 0; n < gi->N[axis]; ++n) {
				gi->dl[axis][gt][n] = gi->dl_orig[axis][gt][n];
			}
		}
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "make_d_one"
PetscErrorCode make_d_one(GridInfo *gi)
{
	PetscFunctionBegin;

	PetscInt axis, gt, n;
	for (axis = 0; axis < Naxis; ++axis) {
		for (gt = 0; gt < Ngt; ++gt) {
			for (n = 0; n < gi->N[axis]; ++n) {
				gi->dl[axis][gt][n] = 1.0;
			}
		}
	}

	PetscFunctionReturn(0);
}

/**
 * Modify create_A_and_b() so that the added continuity equation is symmetric.
 */
#undef __FUNCT__
#define __FUNCT__ "create_A_and_b4"
PetscErrorCode create_A_and_b4(Mat *A, Vec *b, Vec *right_precond, Mat *CF, Vec *conjParam, Vec *conjSrc, GridInfo gi, TimeStamp *ts)
{
	PetscFunctionBegin;
	PetscErrorCode ierr;

	Vec eps, mu, param, paramMask; 
	Vec srcJ, srcM;
	Vec inverse;  // store various inverse vectors
	Vec left_precond, precond;
	Mat CE, CH;  // curl operators on E and H
	Mat CG, CGF; 

	if (gi.verbose_level >= VBMedium) {
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "Create matrix for %s on %s grid with %s, preconditioned by %s.\n", FieldTypeName[gi.x_type], GridTypeName[gi.x_type==Etype ? gi.ge:((gi.ge+1)%Ngt)], PMLTypeName[gi.pml_type], PCTypeName[gi.pc_type]); CHKERRQ(ierr);
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, "The matrix is %s, continuity eq %s", (gi.is_symmetric ? "symmetric":"non-symmetric"), (gi.add_conteq ? "added":"not added")); CHKERRQ(ierr);
		if (gi.add_conteq) {
			ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, " with factor %f", gi.factor_conteq); CHKERRQ(ierr);
		}
		ierr = PetscFPrintf(PETSC_COMM_WORLD, stdout, ".\n"); CHKERRQ(ierr);
	}

	ierr = VecDuplicate(gi.vecTemp, &inverse); CHKERRQ(ierr);

	/** Create input vectors. */
	ierr = createVecPETSc(&eps, "eps", gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "eps vector", gi); CHKERRQ(ierr);

	if (gi.has_mu) {
		ierr = createVecPETSc(&mu, "mu", gi); CHKERRQ(ierr);
	} else {
		ierr = VecDuplicate(gi.vecTemp, &mu); CHKERRQ(ierr);
		ierr = VecSet(mu, 1.0); CHKERRQ(ierr);
	}
	ierr = updateTimeStamp(VBDetail, ts, "mu vector", gi); CHKERRQ(ierr);

	ierr = createVecPETSc(&srcJ, "srcJ", gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "J vector", gi); CHKERRQ(ierr);

	ierr = createVecPETSc(&srcM, "srcM", gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "M vector", gi); CHKERRQ(ierr);

	/** Stretch parameters by  gi.s_factor. */
	if (gi.pml_type == SCPML) {
		ierr = stretch_d(&gi); CHKERRQ(ierr);
	} else {  // UPML
		assert(gi.pml_type == UPML); 
		Vec sfactor;

		ierr = createFieldArray(&sfactor, set_sfactor_eps_at, gi); CHKERRQ(ierr);
		ierr = VecPointwiseMult(eps, eps, sfactor); CHKERRQ(ierr);

		ierr = createFieldArray(&sfactor, set_sfactor_mu_at, gi); CHKERRQ(ierr);
		ierr = VecPointwiseMult(mu, mu, sfactor); CHKERRQ(ierr);

		ierr = VecDestroy(&sfactor); CHKERRQ(ierr);
	}

	/** Set up the matrix CE, the curl operator on E fields. */
	ierr = createCE(&CE, gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "CE matrix", gi); CHKERRQ(ierr);

	/** Set up the matrix CH, the curl operator on H fields. */
	ierr = createCH(&CH, gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "CH matrix", gi); CHKERRQ(ierr);

	/** Set up the matrix CF, the operator giving G fields from F fields. */
	ierr = VecDuplicate(gi.vecTemp, &param); CHKERRQ(ierr);
	ierr = VecDuplicate(gi.vecTemp, &paramMask); CHKERRQ(ierr);
	ierr = VecDuplicate(gi.vecTemp, conjParam); CHKERRQ(ierr);
	ierr = VecDuplicate(gi.vecTemp, conjSrc); CHKERRQ(ierr);
	if (gi.x_type == Etype) {
		ierr = VecCopy(eps, param); CHKERRQ(ierr);
		ierr = VecCopy(mu, *conjParam); CHKERRQ(ierr);
		ierr = VecCopy(srcM, *conjSrc); CHKERRQ(ierr);

		CG = CH;
		*CF = CE;
		ierr = VecSet(inverse, 1.0); CHKERRQ(ierr);
		ierr = VecPointwiseDivide(inverse, inverse, mu); CHKERRQ(ierr);
	} else {  // Htype
		assert(gi.x_type == Htype);
		ierr = VecCopy(mu, param); CHKERRQ(ierr);
		ierr = VecCopy(eps, *conjParam); CHKERRQ(ierr);
		ierr = VecCopy(srcJ, *conjSrc); CHKERRQ(ierr);

		CG = CE;
		*CF = CH;
		ierr = VecSet(inverse, 1.0); CHKERRQ(ierr);
		ierr = VecPointwiseDivide(inverse, inverse, eps); CHKERRQ(ierr);
	}
	ierr = MatDiagonalScale(CG, PETSC_NULL, inverse); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "CG matrix", gi); CHKERRQ(ierr);

	ierr = VecCopy(param, paramMask); CHKERRQ(ierr);
	ierr = infMaskVec(paramMask, gi); CHKERRQ(ierr);  // to handle TruePEC objects
	ierr = maskInf2One(param, gi); CHKERRQ(ierr);  // to handle TruePEC objects

	/** Create the matrix CGF, the curl(mu^-1 curl) operator or curl(eps^-1 curl). */
	ierr = createCGF(&CGF, CG, *CF, gi); CHKERRQ(ierr);
	ierr = updateTimeStamp(VBDetail, ts, "CGF matrix", gi); CHKERRQ(ierr);

	/** Create b. */
	ierr = VecDuplicate(gi.vecTemp, b); CHKERRQ(ierr);
	if (gi.x_type == Etype) {
		ierr = VecCopy(srcJ, *b); CHKERRQ(ierr);
		ierr = VecScale(*b, PETSC_i*gi.omega); CHKERRQ(ierr);
		ierr = MatMultAdd(CG, srcM, *b, *b); CHKERRQ(ierr);
		ierr = VecScale(*b, -1.0); CHKERRQ(ierr);
	} else {
		ierr = VecCopy(srcM, *b); CHKERRQ(ierr);
		ierr = VecScale(*b, -PETSC_i*gi.omega); CHKERRQ(ierr);
		ierr = MatMultAdd(CG, srcJ, *b, *b); CHKERRQ(ierr);
	}
	ierr = updateTimeStamp(VBDetail, ts, "b vector", gi); CHKERRQ(ierr);

	if (!gi.add_conteq) {
		/** Below, isn't *A = CGF the same as A = &CGF?  No.  Remember that A is a return value.  
		  When this function is called, we do:
		  Mat B;
		  ...
		  ierr = create_XXX_A_YYY(&B, ...); CHKERRQ(ierr);
		  The intension of this function call is to fill the memory pointed by &B. *A = CGF fulfills 
		  this intension.
		  On the other hand, if the below line is A = &CGF, it is nothing but changing the value of 
		  the pointer variable A from &B to &CGF.  Therefore nothing is returned to B. */
		*A = CGF;
	} else {  // currently, add_conteq only works for x_type == Etype
		ierr = createAtemplate(A, gi); CHKERRQ(ierr);
		ierr = MatAXPY(*A, 1.0, CGF, SUBSET_NONZERO_PATTERN); CHKERRQ(ierr);
		ierr = MatDestroy(&CGF); CHKERRQ(ierr);

		/** Create the gradient-divergence operator. */
		Mat GD;
		ierr = createGDsym(&GD, gi); CHKERRQ(ierr);
		ierr = MatDiagonalScale(GD, param, PETSC_NULL); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "GD matrix", gi); CHKERRQ(ierr);

		/** Create b. */
		Vec b_aug;
		ierr = VecDuplicate(gi.vecTemp, &b_aug); CHKERRQ(ierr);
		if (gi.x_type == Etype) {
			ierr = VecCopy(srcJ, b_aug); CHKERRQ(ierr);  // b_aug = J
		} else {
			ierr = VecCopy(srcM, b_aug); CHKERRQ(ierr);  // b_aug = M
		}

		ierr = VecScale(b_aug, gi.factor_conteq*PETSC_i/gi.omega); CHKERRQ(ierr);  // b_aug = s*(i/omega)*J
		ierr = MatMultAdd(GD, b_aug, *b, *b); CHKERRQ(ierr);  // b = -i*omega*J + GD * s*(i/omega)*J
		ierr = VecDestroy(&b_aug); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "b_aug vector", gi); CHKERRQ(ierr);

		ierr = MatDiagonalScale(GD, PETSC_NULL, param); CHKERRQ(ierr);
		ierr = MatAXPY(*A, gi.factor_conteq, GD, SUBSET_NONZERO_PATTERN); CHKERRQ(ierr);
		ierr = MatDestroy(&GD); CHKERRQ(ierr);
	}

	ierr = MatDiagonalScale(*A, paramMask, paramMask); CHKERRQ(ierr);  // omega^2*mu*eps is not subtracted yet, so the diagonal entries will be nonzero
	ierr = VecPointwiseMult(*b, paramMask, *b); CHKERRQ(ierr);  // force E = 0 on TruePEC.  comment this line to allow source on TruePEC

	if (!gi.solve_eigen) {
		Vec negW2Param = param;
		ierr = VecScale(negW2Param, -gi.omega*gi.omega); CHKERRQ(ierr);
		ierr = MatDiagonalSet(*A, negW2Param, ADD_VALUES); CHKERRQ(ierr);
	}
	ierr = updateTimeStamp(VBDetail, ts, "A matrix", gi); CHKERRQ(ierr);

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
		if (gi.x_type == Etype) {
			ierr = createFieldArray(&sqrtLS, set_dLe_at, gi); CHKERRQ(ierr);
			ierr = createFieldArray(&dS, set_dSe_at, gi); CHKERRQ(ierr);
		} else {
			assert(gi.x_type == Htype);
			ierr = createFieldArray(&sqrtLS, set_dLh_at, gi); CHKERRQ(ierr);
			ierr = createFieldArray(&dS, set_dSh_at, gi); CHKERRQ(ierr);
		}
		ierr = VecPointwiseMult(sqrtLS, sqrtLS, dS); CHKERRQ(ierr);
		ierr = VecDestroy(&dS); CHKERRQ(ierr);
		ierr = sqrtVec(sqrtLS, gi); CHKERRQ(ierr);

		ierr = VecPointwiseMult(*right_precond, *right_precond, sqrtLS); CHKERRQ(ierr);
		ierr = VecDestroy(&sqrtLS); CHKERRQ(ierr);

		Vec sqrtDoubleFbc;
		ierr = createFieldArray(&sqrtDoubleFbc, set_double_Fbc_at, gi); CHKERRQ(ierr);
		ierr = sqrtVec(sqrtDoubleFbc, gi); CHKERRQ(ierr);

		ierr = VecPointwiseDivide(*right_precond, *right_precond, sqrtDoubleFbc); CHKERRQ(ierr);
		ierr = VecDestroy(&sqrtDoubleFbc); CHKERRQ(ierr);

		ierr = VecPointwiseDivide(left_precond, left_precond, *right_precond); CHKERRQ(ierr);
	}

	/** Apply the preconditioner. Only one type of preconditioners is applied. */
	if (gi.pc_type == PCSfactor) {  
		Vec sfactorL, sfactorS;
		if (gi.x_type == Etype) {
			ierr = createFieldArray(&sfactorL, set_sfactorLe_at, gi); CHKERRQ(ierr);
			ierr = createFieldArray(&sfactorS, set_sfactorSe_at, gi); CHKERRQ(ierr);
		} else {
			assert(gi.x_type == Htype);
			ierr = createFieldArray(&sfactorL, set_sfactorLh_at, gi); CHKERRQ(ierr);
			ierr = createFieldArray(&sfactorS, set_sfactorSh_at, gi); CHKERRQ(ierr);
		}
		if (!gi.is_symmetric) {  // Ascpml = diag(1/sfactorS) Aupml diag(sfactorL)
			ierr = VecPointwiseMult(left_precond, left_precond, sfactorS); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(*right_precond, *right_precond, sfactorL); CHKERRQ(ierr);
		} else {  // diag(sqrt(sfactorL/sfactorS)) Aupml diag(sqrt(sfactorL/sfactorS))
			Vec sqrtLoverS;
			ierr = VecDuplicate(gi.vecTemp, &sqrtLoverS); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(sqrtLoverS, sfactorL, sfactorS); CHKERRQ(ierr);
			ierr = sqrtVec(sqrtLoverS, gi); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(left_precond, left_precond, sqrtLoverS); CHKERRQ(ierr);
			ierr = VecPointwiseDivide(*right_precond, *right_precond, sqrtLoverS); CHKERRQ(ierr);
			ierr = VecDestroy(&sqrtLoverS); CHKERRQ(ierr);
		}

		ierr = VecDestroy(&sfactorL); CHKERRQ(ierr);
		ierr = VecDestroy(&sfactorS); CHKERRQ(ierr);
		ierr = updateTimeStamp(VBDetail, ts, "s-factor preconditioner", gi); CHKERRQ(ierr);
	} else if (gi.pc_type == PCParam) {
		if (gi.x_type == Etype) {
			ierr = createVecHDF5(&precond, "/eps", gi); CHKERRQ(ierr);
		} else {
			assert(gi.x_type == Htype);
			if (gi.has_mu) {
				ierr = createVecHDF5(&precond, "/mu", gi); CHKERRQ(ierr);
			} else {
				ierr = VecDuplicate(gi.vecTemp, &precond); CHKERRQ(ierr);
				ierr = VecSet(precond, 1.0); CHKERRQ(ierr);
			}
		}
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

	/** Recover the original dl. */
	if (gi.pml_type == SCPML) {
		ierr = unstretch_d(&gi); CHKERRQ(ierr);
	}

	ierr = MatDestroy(&CG); CHKERRQ(ierr);
	ierr = VecDestroy(&mu); CHKERRQ(ierr);
	ierr = VecDestroy(&eps); CHKERRQ(ierr);
	ierr = VecDestroy(&param); CHKERRQ(ierr);
	ierr = VecDestroy(&paramMask); CHKERRQ(ierr);
	ierr = VecDestroy(&srcJ); CHKERRQ(ierr);
	ierr = VecDestroy(&srcM); CHKERRQ(ierr);
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
