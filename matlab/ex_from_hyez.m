function ex = ex_from_hyez(hy, ez, mu_yy, gamma)

const;

%% Derive 2D quantities from 3D grid info.
gi = ez.gi;
BC = [gi.BC(Xx,:); gi.BC(Yy,:)];
Nx = gi.N(Xx);
Ny = gi.N(Yy);
N = Nx*Ny;
dx_prim = gi.dL{Prim,Xx};
omega = gi.angular_freq;

%% Set up a diagonal matrices from edge lengths.
mat_dx_prim = repmat(dx_prim.', [1 Ny]);
mat_dx_prim = spdiags(mat_dx_prim(:), 0, sparse(N,N));

%% Create differential operators.
DxEz = get_DpEz(N, Nx, Ny, Xx, BC, mat_dx_prim);

%% Calculate Ex fields.
ex_array = (sqrt(-1)*omega*(mu_yy(:).*hy.array(:)) - DxEz*ez.array(:))./gamma;
ex = scalar2d('Ex', ex_array, Zz, 1, [Dual, Prim, Prim], gi);