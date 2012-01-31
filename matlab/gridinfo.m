classdef gridinfo < handle
    %GRIDINFO3D has all the information related to the grid used in FD3D.

    properties (SetAccess = private)
        L0  % length unit in meter
        display_length_unit = 'nm'  % length unit to display
        dlu_scale = 1e9;  % conversion factor to display length unit
        BC
        exp_neg_ikL
        N
        Npml
        dL  % the array dL{Grid,Axis} includes PML regions.
        L
    end
    
    properties
        wvlen
    end
    
    properties (Dependent)
        angular_freq
        true_wvlen
        eV
    end
    
    methods
        function this = gridinfo(L0, wvlen, BC, Npml, dx_prim, dy_prim, dz_prim, k_Bloch)
            const;
            this.L0 = L0;
            this.wvlen = wvlen;
            this.BC = BC;
            this.Npml = Npml;
            
            d_prim = {dx_prim; dy_prim; dz_prim};
            this.N = [length(dx_prim); length(dy_prim); length(dz_prim)];

            for axis = 1:Naxis
                this.dL{Prim,axis} = d_prim{axis};
                dw_prim = this.dL{Prim,axis};
                dw_dual = (dw_prim(1:end-1) + dw_prim(2:end)) / 2;  % length: Nw-1
                if BC(axis,Neg) == PMC  % length(dw_dual) becomes Nw
                    dw_dual = [dw_prim(1), dw_dual];  % dw_prim(1) == 2 * (w_dual(1) - w_prim(1))
                else  % for periodic BC or PEC
                    dw_dual = [(dw_prim(1)+dw_prim(end))/2, dw_dual];  % sum(dw_prim) == sum(dw_dual)
                end
                this.dL{Dual,axis} = dw_dual;

                lw_prim = cumsum([0 this.dL{Prim,axis}]);  % length: Nw+1
                lw_dual = cumsum([0 this.dL{Dual,axis}]);
                lw_dual = lw_dual - ( lw_dual(2) - (lw_prim(1)+lw_prim(2))/2 );

                lw_orig = lw_prim(Npml(axis,Neg)+1);
                lw_prim = lw_prim - lw_orig;
                lw_dual = lw_dual - lw_orig;

                this.L{Prim,axis} = lw_prim;
                this.L{Dual,axis} = lw_dual;
            end
            
            if nargin >= 8
                assert(isvector(k_Bloch) && length(k_Bloch)==3);
                Lx = this.L{Prim,Xx}(end);
                Ly = this.L{Prim,Yy}(end);
                Lz = this.L{Prim,Zz}(end);
                this.exp_neg_ikL = [exp(-sqrt(-1)*k_Bloch(Xx)*Lx); exp(-sqrt(-1)*k_Bloch(Yy)*Ly); exp(-sqrt(-1)*k_Bloch(Zz)*Lz)]; 
            else
                this.exp_neg_ikL = [1.0; 1.0; 1.0];
            end
        end
        
        function lambda = get.true_wvlen(this)
            lambda = this.wvlen * this.L0;
        end
        
        function omega = get.angular_freq(this)
            omega = 2*pi / this.wvlen;
        end
        
        function set.angular_freq(this, omega)
            this.wvlen = 2*pi/omega;
        end
        
        function energy = get.eV(this)
            const;
            energy = heV*c0 / this.true_wvlen;
        end
    end
end