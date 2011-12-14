function [rho epse] = rho_from_e_eps(ex, ey, ez, eps_grid, gamma)
% Get rho and the D-field from the E-field, eps_grid, and gamma.

const;

%% Derive 2D quantities from 3D grid info.
gi = ez.gi;
BC = [gi.BC(Xx,:); gi.BC(Yy,:)];
Nx = gi.N(Xx);
Ny = gi.N(Yy);
N = Nx*Ny;
dx_dual = gi.dL{Dual,Xx};
dy_dual = gi.dL{Dual,Yy};

%omega = gi.angular_freq;

%% Set up a diagonal matrices from edge lengths.
mat_dx_dual = repmat(dx_dual.', [1 Ny]);
mat_dy_dual = repmat(dy_dual, [Nx 1]);

mat_dx_dual = spdiags(mat_dx_dual(:), 0, sparse(N,N));
mat_dy_dual = spdiags(mat_dy_dual(:), 0, sparse(N,N));


%% Create differential operators.
DxEx = get_DpEp(N, Nx, Ny, Xx, BC, mat_dx_dual);
DyEy = get_DpEp(N, Nx, Ny, Yy, BC, mat_dy_dual);

%% Create eps_xx, eps_yy, eps_zz.
[eps_xx, eps_yy, eps_zz] = eps_ww_from_eps_grid(eps_grid, gi);
epse_x_array = eps_xx(:).*ex.array(:);
epse_y_array = eps_yy(:).*ey.array(:);
epse_z_array = eps_zz(:).*ez.array(:);

%% Calculate Ex fields.
% The minus sign in front of gamma is because the z dependence is exp(-gamma z) rather than exp(gamma z).
% This is because the time dependence is exp(i omega t).
% See Veronis 2007, Journal of Lightwave Technology, 25 (9) pp. 2511-2521, 2007.
rho_array = DxEx*(epse_x_array) + DyEy*(epse_y_array) - gamma*epse_z_array;

rho = scalar2d('rho', rho_array, Zz, 1, [Prim, Prim, Prim], gi);
epse_x = scalar2d('Dx', epse_x_array, Zz, 1, [Dual, Prim, Prim], gi);
epse_y = scalar2d('Dy', epse_y_array, Zz, 1, [Prim, Dual, Prim], gi);
epse_z = scalar2d('Dz', epse_z_array, Zz, 1, [Prim, Prim, Dual], gi);

epse = {epse_x, epse_y, epse_z};