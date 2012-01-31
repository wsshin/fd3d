function uavg = uavg_from_eh(e, h, eps, eps_n, wvlen_n)
% The time-varying part of the energy density is calculated at the center of Yee's cell.  
%
% e, h, eps: E field, H field, and permittivity at the operation wavelength
% wvlen_n, eps_n: wavelength slightly different from the operation wavelength
% and permittivity at the frequency

const;

gi = e{Xx}.gi;
Nx = gi.N(Xx); Ny = gi.N(Yy); Nz = gi.N(Zz);

hx = h{Xx}.array;
hy = h{Yy}.array;
hz = h{Zz}.array;

% time-averaged magnetic energy density.
umx = 0.25.*(abs(hx).^2);
umy = 0.25.*(abs(hy).^2);
umz = 0.25.*(abs(hz).^2);

clear hx hy hz;


% Interpolate um at the center of Yee's cell.  When interpolating, take 
% umx, umy, umz as if they are the x,y,z components of the energy density
% "vector", and take the average of each component.  This way we can keep 
% the total energy constant before and after the interpolation, except the 
% contribution of the boundary values.

% Interpolate umx in the x direction.
switch gi.BC(Xx,Pos)
    case PEC
        umx(end+1,:,:) = zeros(Ny,Nz);
    case PMC
        umx(end+1,:,:) = umx(end,Ny,Nz);
    case Bloch
        umx(end+1,:,:) = umx(1,:,:);
    otherwise
        error('Not a supported boundary condition');
end
umx = (umx(1:end-1,:,:) + umx(2:end,:,:))/2;

% Interpolate umy in the y direction.
switch gi.BC(Yy,Pos)
    case PEC
        umy(:,end+1,:) = zeros(Nx,Nz);
    case PMC
        umy(:,end+1,:) = umy(Nx,end,Nz);
    case Bloch
        umy(:,end+1,:) = umy(:,1,:);
    otherwise
        error('Not a supported boundary condition');
end
umy = (umy(:,1:end-1,:) + umy(:,2:end,:))/2;

% Interpolate umz in the z direction.
switch gi.BC(Zz,Pos)
    case PEC
        umz(:,:,end+1) = zeros(Nx,Ny);
    case PMC
        umz(:,:,end+1) = umz(Nx,Ny,end);
    case Bloch
        umz(:,:,end+1) = umz(:,:,1);
    otherwise
        error('Not a supported boundary condition');
end
umz = (umz(:,:,1:end-1) + umz(:,:,2:end))/2;

um = umx + umy + umz;
clear umx umy umz;


% AC part of the electric energy density.
omega = gi.angular_freq();
omega_n = 2*pi/wvlen_n;

ex = e{Xx}.array;
ey = e{Yy}.array;
ez = e{Zz}.array;

weps_xx = omega .* real(eps{Xx}.array);
weps_yy = omega .* real(eps{Yy}.array);
weps_zz = omega .* real(eps{Zz}.array);

weps_nxx = omega_n .* real(eps_n{Xx}.array);
weps_nyy = omega_n .* real(eps_n{Yy}.array);
weps_nzz = omega_n .* real(eps_n{Zz}.array);

coeff_xx = (weps_xx - weps_nxx) ./ (omega - omega_n);
coeff_yy = (weps_yy - weps_nyy) ./ (omega - omega_n);
coeff_zz = (weps_zz - weps_nzz) ./ (omega - omega_n);

uex = 0.25.*coeff_xx.*(abs(ex).^2);
uey = 0.25.*coeff_yy.*(abs(ey).^2);
uez = 0.25.*coeff_zz.*(abs(ez).^2);

clear ex ey ez weps_xx weps_yy weps_zz weps_nxx weps_nyy weps_nzz;

% Interpolate ue at the center of Yee's cell.  When interpolating, take 
% uex, uey, uez as if they are the x,y,z components of the energy density
% "vector", and take the average of each component.  This way we can keep 
% the total energy constant before and after the interpolation, except the 
% contribution of the boundary values.
% Note that we have to interpolate ue twice to get the values at the center 
% of Yee's cell.

% Interpolate uex in the y direction.
switch gi.BC(Yy,Pos)
    case PEC
        uex(:,end+1,:) = zeros(Nx,Nz);
    case PMC
        uex(:,end+1,:) = uex(Nx,end,Nz);
    case Bloch
        uex(:,end+1,:) = uex(:,1,:);
    otherwise
        error('Not a supported boundary condition');
end
uex = (uex(:,1:end-1,:) + uex(:,2:end,:))/2;

% Interpolate uex in the z direction.
switch gi.BC(Zz,Pos)
    case PEC
        uex(:,:,end+1) = zeros(Nx,Ny);
    case PMC
        uex(:,:,end+1) = uex(Nx,Ny,end);
    case Bloch
        uex(:,:,end+1) = uex(:,:,1);
    otherwise
        error('Not a supported boundary condition');
end
uex = (uex(:,:,1:end-1) + uex(:,:,2:end))/2;

% Interpolate uey in the z direction.
switch gi.BC(Zz,Pos)
    case PEC
        uey(:,:,end+1) = zeros(Nx,Ny);
    case PMC
        uey(:,:,end+1) = uey(Nx,Ny,end);
    case Bloch
        uey(:,:,end+1) = uey(:,:,1);
    otherwise
        error('Not a supported boundary condition');
end
uey = (uey(:,:,1:end-1) + uey(:,:,2:end))/2;

% Interpolate uey in the x direction.
switch gi.BC(Xx,Pos)
    case PEC
        uey(end+1,:,:) = zeros(Ny,Nz);
    case PMC
        uey(end+1,:,:) = uey(end,Ny,Nz);
    case Bloch
        uey(end+1,:,:) = uey(1,:,:);
    otherwise
        error('Not a supported boundary condition');
end
uey = (uey(1:end-1,:,:) + uey(2:end,:,:))/2;

% Interpolate uez in the x direction.
switch gi.BC(Xx,Pos)
    case PEC
        uez(end+1,:,:) = zeros(Ny,Nz);
    case PMC
        uez(end+1,:,:) = uez(end,Ny,Nz);
    case Bloch
        uez(end+1,:,:) = uez(1,:,:);
    otherwise
        error('Not a supported boundary condition');
end
uez = (uez(1:end-1,:,:) + uez(2:end,:,:))/2;

% Interpolate uez in the y direction.
switch gi.BC(Yy,Pos)
    case PEC
        uez(:,end+1,:) = zeros(Nx,Nz);
    case PMC
        uez(:,end+1,:) = uez(Nx,end,Nz);
    case Bloch
        uez(:,end+1,:) = uez(:,1,:);
    otherwise
        error('Not a supported boundary condition');
end
uez = (uez(:,1:end-1,:) + uez(:,2:end,:))/2;

ue = uex + uey + uez;
clear uex uey uez;

uavg = scalar3d('u_avg', ue+um, [Dual Dual Dual], gi);