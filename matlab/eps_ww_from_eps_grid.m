function [eps_xx, eps_yy, eps_zz] = eps_ww_from_eps_grid(eps_grid, gi)

const;

BC = [gi.BC(Xx,:); gi.BC(Yy,:)];

% Pad appropriate material parameters to the positive end of x, and take harmonic averages of neighboring points along x.
switch BC(Xx,Pos)
    case PEC
        % eps_xx = [eps_grid; ones(1,Ny)*Inf];
        % The above assignment seems correct in that PEC has infinite conductivity.  But 
        % it actually results in large field values at the boundary.  If we remember
        % that PEC symmetry plane simulates the same material on the other side of the 
        % plane, assigning the same permittivity on the PEC plane as in the problem domain
        % seems reasonable.
        eps_xx = [eps_grid; eps_grid(end,:)];
    case PMC
        eps_xx = [eps_grid; eps_grid(end,:)];
    case Bloch
        eps_xx = [eps_grid; eps_grid(1,:)];
    otherwise
        error('Not a supported boundary condition');
end
%eps_xx = [eps_grid; eps_grid(end,:)];
eps_xx = 0.5*(1./eps_xx(1:end-1,:) + 1./eps_xx(2:end,:));  % 1/0 = Inf
eps_xx = 1./eps_xx;

% Pad appropriate material parameters to the positive end of y, and take harmonic averages of neighboring points along y.
switch BC(Yy,Pos)
    case PEC
        % eps_yy = [eps_grid, ones(Nx,1)*Inf];
        % The above assignment seems correct in that PEC has infinite conductivity.  But 
        % it actually results in large field values at the boundary.  If we remember
        % that PEC symmetry plane simulates the same material on the other side of the 
        % plane, assigning the same permittivity on the PEC plane as in the problem domain
        % seems reasonable.
        eps_yy = [eps_grid, eps_grid(:,end)];
    case PMC
        eps_yy = [eps_grid, eps_grid(:,end)];
    case Bloch
        eps_yy = [eps_grid, eps_grid(:,1)];
    otherwise
        error('Not a supported boundary condition');
end
%eps_yy = [eps_grid, eps_grid(:,end)];
eps_yy = 0.5*(1./eps_yy(:,1:end-1) + 1./eps_yy(:,2:end));
eps_yy = 1./eps_yy;

eps_zz = eps_grid;
