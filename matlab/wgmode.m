classdef wgmode < handle
    %WGMODE represents the waveguide mode whose cross section is on the xy plane.
    
    properties (Access = protected)
        eps
        mu
        iscalculated = false;
    end
    
    properties (Dependent)
        wvlen
        angular_freq
    end
    
    properties (SetAccess = private)
        gamma
        e
        epse  % eps * E = D field
        h
        sz
        rho
    end
    
    properties (SetAccess = protected)
        gi
        num_gamma
    end
    
    methods (Abstract)
        update_eps_mu(this)
    end
    
    methods
        function wvlen = get.wvlen(this)
            wvlen = this.gi.wvlen;
        end
        
        function set.wvlen(this, wvlen)
            this.gi.wvlen = wvlen;
            this.iscalculated = false;
        end
        
        function omega = get.angular_freq(this)
            omega = this.gi.angular_freq;
        end
        
        function set.angular_freq(this, omega)
            this.gi.angular_freq = omega;
            this.iscalculated = false;
        end
        
        function calc(this, gamma_expected)
            const;

            disp('-------------------------------------------------------');
            disp(['wvlen: ', num2str(this.wvlen)]);
            disp(['angular freq: ', num2str(this.angular_freq)]);
            
            Nx = this.gi.N(Xx);
            Ny = this.gi.N(Yy);
            N = Nx * Ny;
            
            %% Get the FDFD matrix.
            this.update_eps_mu();
            A = mode2d_matrix(this.eps, this.mu, this.gi);

            %% Solve the equation.
            % Note that eigs() gives approximate eigenvalues and eigenvectors, and
            % the results differ from one execution of eigs() to another, even if 
            % the arguments given to eigs() are exactly the same.
%             options.issym = 1;
%             options.isreal = 1;
%             [x eigval] = eigs(A,this.num_gamma,gamma_expected^2,options);
            [x eigval] = eigs(A,this.num_gamma,gamma_expected^2);
            this.iscalculated = true;

            this.gamma = sqrt(eigval(this.num_gamma));

            hx = scalar2d('Hx', reshape(x(1:N,this.num_gamma), Nx, Ny), Zz, 1, [Prim, Dual, Dual], this.gi);
            hy = scalar2d('Hy', reshape(x(N+1:2*N,this.num_gamma), Nx, Ny), Zz, 1, [Dual, Prim, Dual], this.gi);
            hx.margin = this.gi.Npml;
            hy.margin = this.gi.Npml;
            this.h = {hx, hy};

            eps_zz = this.eps;
            mu_xx = this.mu;
            mu_yy = this.mu;

            ez = ez_from_hxy(hx, hy, eps_zz);
            ex = ex_from_hyez(hy, ez, mu_yy, this.gamma);
            ey = ey_from_hxez(hx, ez, mu_xx, this.gamma);
            ex.margin = this.gi.Npml;
            ey.margin = this.gi.Npml;
            ez.margin = this.gi.Npml;
            this.e = {ex, ey, ez};

            this.sz = poynting_from_eh(Zz, 1, this.e, this.h);
            this.sz.margin = this.gi.Npml;
            
            [this.rho this.epse] = rho_from_e_eps(ex, ey, ez, this.eps, this.gamma);


            disp(['gamma expected: ', num2str(gamma_expected)]);
            disp(['gamma calculated: ', num2str(this.gamma)]);
            disp(['power flux: ', num2str(this.sz.flux())]);
            fprintf('\n');
        end
        
        function gamma = get.gamma(this)
            if ~this.iscalculated
                error('Mode is not calculated yet');
            end
            gamma = this.gamma;
        end
        
        function e = get.e(this)
            if ~this.iscalculated
                error('Mode is not calculated yet');
            end
            e = this.e;
        end
        
        function h = get.h(this)
            if ~this.iscalculated
                error('Mode is not calculated yet');
            end
            h = this.h;
        end
        
        function sz = get.sz(this)
            if ~this.iscalculated
                error('Mode is not calculated yet');
            end
            sz = this.sz;
        end
    end
end

