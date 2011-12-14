[~, FD3D_ROOT] = system('source ~/.fd3d;echo $FD3D_ROOT');  % deleted when "clear all" is called
%[~, PETSC_DIR] = system('source ~/.fd3d;echo $PETSC_DIR');  % deleted when "clear all" is called

c0 = 299792458;
mu0 = 4*pi*1E-7;
eps0 = 1 / (c0^2 * mu0);
eta0 = sqrt(mu0/eps0);

heV = 4.13566733e-15;
hbar = heV / (2*pi);

%% Boundary conditions
% PMC: it is used only at negative ends of axes, not at positive ends.
% Bloch: when it is used at either of a positive and negative end of an axis, 
% it should be used at the other end, too.
PEC = 1;
PMC = 2;
Bloch = 3;

Xx = 1;
Yy = 2;
Zz = 3;
Naxis = 3;

AxisName = ['x', 'y', 'z'];

Neg = 1;
Pos = 2;
Nsign = 2;

Prim = 1;
Dual = 2;

Efield = 1;
Hfield = 2;