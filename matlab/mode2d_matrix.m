function A = mode2d_matrix(eps_grid, mu_grid, gi)
% This operator does not support Bloch boundary condition with nonzero k_Bloch; 
% need to update get_DpEq(), etc, for Bloch BC.
% The operator A acts on H fields, not E fields.  Therefore, A x = lambda x 
% solves for H fields of an eigenmode of the system.

%% Import constants.
const;

%% Derive various parameters (for 2D problems) from 3D grid info.
BC = [gi.BC(Xx,:); gi.BC(Yy,:)];
Nx = gi.N(Xx); Ny = gi.N(Yy);
N = Nx*Ny;
Npml = [gi.Npml(Xx,:); gi.Npml(Yy,:)];
dx_prim = gi.dL{Prim,Xx};
dy_prim = gi.dL{Prim,Yy};
dx_dual = gi.dL{Dual,Xx};
dy_dual = gi.dL{Dual,Yy};
omega = gi.angular_freq;

%% Set up the diagonal matrices from edge lengths.
mat_dx_prim = repmat(dx_prim.', [1 Ny]);
mat_dx_dual = repmat(dx_dual.', [1 Ny]);
mat_dy_prim = repmat(dy_prim, [Nx 1]);
mat_dy_dual = repmat(dy_dual, [Nx 1]);

mat_dx_prim = spdiags(mat_dx_prim(:), 0, sparse(N,N));
mat_dx_dual = spdiags(mat_dx_dual(:), 0, sparse(N,N));
mat_dy_prim = spdiags(mat_dy_prim(:), 0, sparse(N,N));
mat_dy_dual = spdiags(mat_dy_dual(:), 0, sparse(N,N));

%% Create differential operators.
DxHy = get_DpHq(N, Nx, Ny, Xx, BC, mat_dx_dual);
DyHx = get_DpHq(N, Nx, Ny, Yy, BC, mat_dy_dual);

DxHx = get_DpHp(N, Nx, Ny, Xx, BC, mat_dx_prim);
DyHy = get_DpHp(N, Nx, Ny, Yy, BC, mat_dy_prim);

DxEz = get_DpEz(N, Nx, Ny, Xx, BC, mat_dx_prim);
DyEz = get_DpEz(N, Nx, Ny, Yy, BC, mat_dy_prim);

DxHz = get_DpHz(N, Nx, Ny, Xx, BC, mat_dx_dual);
DyHz = get_DpHz(N, Nx, Ny, Yy, BC, mat_dy_dual);

%% Set up s-parameters of UPML.
sx_prim = s_struct([Nx Ny], Prim, Xx, Npml(Xx,Neg), Npml(Xx,Pos), dx_prim(1), dx_prim(end), omega);  % sx at primary grid points (integral indices in x; Ex points)
sx_dual = s_struct([Nx Ny], Dual, Xx, Npml(Xx,Neg), Npml(Xx,Pos), dx_prim(1), dx_prim(end), omega);  % sx at dual grid points (half-integral indices in x; between Ex points)
sy_prim = s_struct([Nx Ny], Prim, Yy, Npml(Yy,Neg), Npml(Yy,Pos), dy_prim(1), dy_prim(end), omega);  % sy at primary grid points (integral indices in y; Ey points)
sy_dual = s_struct([Nx Ny], Dual, Yy, Npml(Yy,Neg), Npml(Yy,Pos), dy_prim(1), dy_prim(end), omega);  % sy at dual grid points (half-integral indices in y; between Ey points)

%% Create eps_ww and mu_ww.
[eps_xx, eps_yy, eps_zz] = eps_ww_from_eps_grid(eps_grid, gi);

mu_xx = mu_grid;
mu_yy = mu_grid;
mu_zz = mu_grid;

%% Create eps and mu matrices.  Note that each of eps_xx, eps_yy, mu_zz can be either a scalar or matrix.
eps_xx_mat = eps_xx .* sy_prim ./ sx_dual;  % eps_xx is multiplied to Ex.  Ex's are at dual grid positions in x, and at grid positions in y.
eps_yy_mat = eps_yy .* sx_prim ./ sy_dual;  % eps_yy is multiplied to Ey.  Ey's are at grid positions in x, and at dual grip positions in x.
eps_zz_mat = eps_zz .* sx_prim .* sy_prim;  % eps_zz is multiplied to Ez.  Ez's are at grid positions in both x and y.
mu_xx_mat = mu_xx .* sy_dual ./ sx_prim;  % mu_xx is multiplied to Hx.  Hx's are at grid positions in x, and at dual grid positions in y.
mu_yy_mat = mu_yy .* sx_dual ./ sy_prim;  % mu_xx is multiplied to Hx.  Hy's are at dual grid positions in x, and at grid positions in y.
mu_zz_mat = mu_zz .* sx_dual .* sy_dual;  % mu_zz is multiplied to Hz.  Hz's are at dual grid positions in both x and y.

clear sx_prim sx_dual sy_prim sy_dual;

eps_xx_mat = diag(sparse(eps_xx_mat(:)));
eps_yy_mat = diag(sparse(eps_yy_mat(:)));
eps_zz_mat = diag(sparse(eps_zz_mat(:)));
mu_xx_mat = diag(sparse(mu_xx_mat(:)));
mu_yy_mat = diag(sparse(mu_yy_mat(:)));
mu_zz_mat = diag(sparse(mu_zz_mat(:)));


%% Create the operator.
filler = sparse(N,N);  % just zero matrix to put between submatrix blocks
A = -omega^2 * [eps_yy_mat filler; filler eps_xx_mat] * [mu_xx_mat filler; filler mu_yy_mat] ...
    + [eps_yy_mat filler; filler -eps_xx_mat] * [DyEz; DxEz] * (eps_zz_mat \ [-DyHx DxHy]) ...
    - [DxHz; DyHz] * (mu_zz_mat \ ([DxHx DyHy] * [mu_xx_mat filler; filler mu_yy_mat]));


% To force the H field components normal to PEC to be 0, mask the matrix appropriately.
mask_hx = ones(Nx,Ny);
mask_hy = ones(Nx,Ny);

if BC(Xx,Neg) == PEC
    mask_hx(1,:) = 0;
end

if BC(Yy,Neg) == PEC
    mask_hy(:,1) = 0;
end

mask_hx = diag(sparse(mask_hx(:)));
mask_hy = diag(sparse(mask_hy(:)));

mask = [mask_hx filler; filler mask_hy];
A = mask*A;