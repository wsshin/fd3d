function ez = ez_from_hxy(hx, hy, eps_zz)

const;

%% Derive 2D quantities from 3D grid info.
gi = hx.gi;
BC = [gi.BC(Xx,:); gi.BC(Yy,:)];
Nx = gi.N(Xx);
Ny = gi.N(Yy);
N = Nx*Ny;
dx_dual = gi.dL{Dual,Xx};
dy_dual = gi.dL{Dual,Yy};
omega = gi.angular_freq;

%% Set up the diagonal matrices from edge lengths.
mat_dx_dual = repmat(dx_dual.', [1 Ny]);
mat_dy_dual = repmat(dy_dual, [Nx 1]);

mat_dx_dual = spdiags(mat_dx_dual(:), 0, sparse(N,N));
mat_dy_dual = spdiags(mat_dy_dual(:), 0, sparse(N,N));

%% Create differential operators.
DxHy = get_DpHq(N, Nx, Ny, Xx, BC, mat_dx_dual);
DyHx = get_DpHq(N, Nx, Ny, Yy, BC, mat_dy_dual);

%% Calculate Ez fields.
ez_array = (DxHy*hy.array(:) - DyHx*hx.array(:))./eps_zz(:)/omega/sqrt(-1);
ez = scalar2d('Ez', ez_array, Zz, 1, [Prim, Prim, Dual], gi);
