function ey = ey_from_hxez(hx, ez, mu_xx, gamma)

const;

%% Derive 2D quantities from 3D grid info.
gi = ez.gi;
BC = [gi.BC(Xx,:); gi.BC(Yy,:)];
Nx = gi.N(Xx);
Ny = gi.N(Yy);
N = Nx*Ny;
dy_prim = gi.dL{Prim,Yy};
omega = gi.angular_freq;

%% Set up a diagonal matrice from edge lengths.
mat_dy_prim = repmat(dy_prim, [Nx 1]);
mat_dy_prim = spdiags(mat_dy_prim(:), 0, sparse(N,N));

%% Create differential operators.
DyEz = get_DpEz(N, Nx, Ny, Yy, BC, mat_dy_prim);

%% Calculate Ey fields.
ey_array = (-sqrt(-1)*omega*(mu_xx(:).*hx.array(:)) - DyEz*ez.array(:))./gamma;
ey = scalar2d('Ey', ey_array, Zz, 1, [Prim, Dual, Prim], gi);