classdef scalar2d < handle
    %SCALAR2D represents a 2D arary of scalar values.  It can represents a single
    % directional component of a 3D vector field (e.g. Ex of E) on a 2D plane.
    
    properties
        margin = [0 0; 0 0; 0 0];
        phase_angle = 0.0;
        draw_colorbar = true;
        draw_function = @(x) x;
        draw_abs = false;
        draw_abs_cscale = false;  % set true to prevent the colorbar change in the movie
        draw_grid = false;
    end
    
    properties (SetAccess = private)
        array
        gi
        grid_kind
        name
        normal
        intercept
        calc_flux = false;
        Pp
        Qq
    end
    
    properties (Access = private)
        real_intercept
    end
    
    methods
        function this = scalar2d(name, array, normal, intercept, gk, gi, calc_flux)
            const;
            this.name = name;
            this.normal = normal;
            this.intercept = intercept;
            this.grid_kind = gk;
            this.gi = gi;
            if nargin >= 7
                this.calc_flux = calc_flux;
            end
            [this.Pp this.Qq] = cycle_axis(normal);
            this.array = reshape(array, gi.N(this.Pp), gi.N(this.Qq));

            switch this.grid_kind(this.normal)
                case Prim
                    this.real_intercept = this.gi.L{Prim,this.normal}(this.intercept);
                case Dual
                    this.real_intercept = this.gi.L{Dual,this.normal}(this.intercept+1);
                otherwise
                    error('Not a supported kind of grid');
            end
        end
        
        function [field, Lp, Lq, dp, dq] = data_plotted(this)
            const;
            
            field = this.array;
            field = exp(sqrt(-1) * this.phase_angle) .* field;
            field = this.draw_function(field);
            
            if this.draw_abs
                field = abs(field);
            else 
                field = real(field);
            end
            
            mp_neg = this.margin(this.Pp,Neg);
            mp_pos = this.margin(this.Pp,Pos);
            mq_neg = this.margin(this.Qq,Neg);
            mq_pos = this.margin(this.Qq,Pos);
            
            field = field(1+mp_neg:end-mp_pos, 1+mq_neg:end-mq_pos);

            % If the field values are defined on dual grid points, Lp and Lq are locations of primary grid lines.
            Lp = this.gi.L{conjugate_grid(this.grid_kind(this.Pp)), this.Pp};
            Lq = this.gi.L{conjugate_grid(this.grid_kind(this.Qq)), this.Qq};
            Lp = Lp(1+mp_neg:end-mp_pos);
            Lq = Lq(1+mq_neg:end-mq_pos);
            
            % If the field values are defined on dual grid points, Lp and Lq are distances between primary grid lines.
            dp = this.gi.dL{conjugate_grid(this.grid_kind(this.Pp)), this.Pp};
            dq = this.gi.dL{conjugate_grid(this.grid_kind(this.Qq)), this.Qq};
            dp = dp(1+mp_neg:end-mp_pos);
            dq = dq(1+mq_neg:end-mq_pos);
        end
        
        function h = plot(this, cscale)
            const;

            [field, Lp, Lq] = this.data_plotted();

            if this.draw_abs_cscale  % calculate [cmin cmax] from the phasors, before real(field) or abs(field) is taken
                cmax = max(abs(field(:)));
                if this.draw_abs
                    cmin = 0.0;
                else
                    cmin = -cmax;
                end
            end
            
            if ~this.draw_abs_cscale  % calculate [cmin cmax] after real(field) or abs(field) is taken above
                cmax = max(real(field(:)));
                cmin = min(real(field(:)));
            end

            crange = [cmin cmax];
            if nargin >= 2
                if length(cscale)==1
                    crange = cscale * crange;
                elseif length(cscale)==2
                    crange = cscale;
                end
            end
            
            % Pad the end boundaries.
            field = [field; field(end,:)];
            field = [field, field(:,end)];
            field = double(field);  % CData should be in double precision
            
            h = pcolor(Lp, Lq, field.');  % transpose field
            axis image, axis xy;
            
            caxis(crange);
            if this.draw_colorbar
                colorbar;
            end
            
            if this.draw_grid
                set(h,'EdgeAlpha',1)
            else
                set(h,'EdgeAlpha',0)
            end
            
            L0 = this.gi.L0;
            dlu = this.gi.display_length_unit;
            dlu_scale = this.gi.dlu_scale;

            if this.draw_abs
                plot_title = strcat('|', this.name, '|', ', \lambda =', num2str(this.gi.wvlen*L0*dlu_scale), dlu, ' (', AxisName(this.normal), '=', num2str(this.real_intercept*L0*dlu_scale), dlu, ')');
                colormap('hot');
            else 
                plot_title = strcat(this.name, ', \lambda =', num2str(this.gi.wvlen*L0*dlu_scale), dlu, ' (', AxisName(this.normal), '=', num2str(this.real_intercept*L0*dlu_scale), dlu, ')');
                colormap('jet');
            end

            if this.calc_flux
                title({plot_title; strcat('Flux=',num2str(this.flux()))});
            else
                title(plot_title);
            end
            
            xlabel(strcat(AxisName(this.Pp), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));
            ylabel(strcat(AxisName(this.Qq), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));

            drawnow;
        end
        
        function set.phase_angle(this, phase_angle)
            this.phase_angle = phase_angle;
        end
        
        function set.margin(this, margin)
            this.margin = margin;
        end
        
        % The flux of the quantity that is plotted
        function flux_value = flux(this)
            const;
            [field, Lp, Lq, dp, dq] = this.data_plotted();
            dS = dp.' * dq;

            flux_value = field .* dS;
            flux_value = sum(flux_value(:));
        end
        
        function create_movie(this, filename, num_periods, framerate, cscale)
            fig1 = figure(1);
            winsize = get(fig1, 'Position');
            winsize(1:2) = [0 0];
            mat_movie = moviein(1, fig1, winsize);
            set(fig1, 'NextPlot', 'replacechildren');
        
            for ind_frame = 1:framerate*num_periods
                this.phase_angle = 2*pi*(ind_frame-1)/framerate;

                if nargin >= 5
                    this.plot(cscale);
                else
                    this.plot();
                end                    
                
                mat_movie(ind_frame) = getframe(fig1, winsize);
            end
            
            movie2avi(mat_movie, filename);
        end
    end
end
