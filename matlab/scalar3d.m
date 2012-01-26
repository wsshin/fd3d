classdef scalar3d < handle
    %SCALAR3D represents a 3D arary of scalar values.  It can represents a single
    % directional component of a 3D vector field (e.g. Ex of E).
    
    properties
        name
        freq
        grid_kind
        margin = [0 0; 0 0; 0 0];
        phase_angle = 0.0;
        view_angle = [-135, -45];
        opaque = true;
        opacity = 0.1;
        draw_colorbar = true;
        draw_function = @(x) x;
        draw_abs = false;
        draw_abs_cscale = false;  % set true to prevent the colorbar change in the movie
        draw_grid = false;
        geometry_color = 'black';
    end

    properties (SetAccess = private)
        array
        gi
    end
    
    properties (GetAccess = private)
        geometry = [];
    end
        
    methods
        function this = scalar3d(name, array, gk, gi)
            const;
            this.name = name;
            assert(isvector(gk) && length(gk)==3);
            this.grid_kind = gk;
            this.gi = gi;
            this.array = reshape(array, gi.N(Xx), gi.N(Yy), gi.N(Zz));
        end
        
        function s2d = get_slice(this, normal, intercept)
            const;
            switch normal
                case Xx
                    a2d = this.array(intercept,:,:);
                    a2d = reshape(a2d, this.gi.N(Yy), this.gi.N(Zz));
                case Yy
                    a2d = this.array(:,intercept,:);
                    a2d = reshape(a2d, this.gi.N(Xx), this.gi.N(Zz));
                    a2d = a2d.';  % ensure the right-hand rule of the 3D Cartesian coordinates
                case Zz
                    a2d = this.array(:,:,intercept);
                    a2d = reshape(a2d, this.gi.N(Xx), this.gi.N(Yy));
                otherwise
                    error('normal is not a proper Cartesian direction');
            end
            s2d = scalar2d(this.name, a2d, normal, intercept, this.grid_kind, this.gi);
            s2d.phase_angle = this.phase_angle;
            s2d.margin = this.margin;
            s2d.draw_colorbar = this.draw_colorbar;
        end
        
        function set.phase_angle(this, phase_angle)
            this.phase_angle = phase_angle;
        end
        
        function data = exportable(this, with_position)
            const;

            field = this.array;
            
            mx_neg = this.margin(Xx,Neg);
            mx_pos = this.margin(Xx,Pos);
            my_neg = this.margin(Yy,Neg);
            my_pos = this.margin(Yy,Pos);
            mz_neg = this.margin(Zz,Neg);
            mz_pos = this.margin(Zz,Pos);

            field = field(1+mx_neg:end-mx_pos, 1+my_neg:end-my_pos, 1+mz_neg:end-mz_pos);

            if with_position
                Lx = this.gi.L{this.grid_kind(Xx), Xx};
                Ly = this.gi.L{this.grid_kind(Yy), Yy};
                Lz = this.gi.L{this.grid_kind(Zz), Zz};

                Lx = Lx(1+mx_neg:end-mx_pos);
                Ly = Ly(1+my_neg:end-my_pos);
                Lz = Lz(1+mz_neg:end-mz_pos);

                [x, y, z] = ndgrid(Lx, Ly, Lz);


                % Pad the end boundaries.
                field(end+1,:,:) = field(end,:,:);
                field(:,end+1,:) = field(:,end,:);
                field(:,:,end+1) = field(:,:,end);

                data = [x(:),y(:),z(:),field(:)];
            else
                data = permute(field, [2 1 3]);  % make field compatible to meshgrid()                
            end
            
            data = single(data);
        end

        function h = plot_iso(this, isovalue)
            const;

            L0 = this.gi.L0;
            dlu = this.gi.display_length_unit;
            dlu_scale = this.gi.dlu_scale;

            field = this.array;
            field = exp(sqrt(-1) * this.phase_angle) .* field;
            
            mx_neg = this.margin(Xx,Neg);
            mx_pos = this.margin(Xx,Pos);
            my_neg = this.margin(Yy,Neg);
            my_pos = this.margin(Yy,Pos);
            mz_neg = this.margin(Zz,Neg);
            mz_pos = this.margin(Zz,Pos);

            Lx = this.gi.L{this.grid_kind(Xx), Xx};
            Ly = this.gi.L{this.grid_kind(Yy), Yy};
            Lz = this.gi.L{this.grid_kind(Zz), Zz};

            Lx = Lx(1+mx_neg:end-mx_pos);
            Ly = Ly(1+my_neg:end-my_pos);
            Lz = Lz(1+mz_neg:end-mz_pos);

            [x, y, z] = meshgrid(Lx, Ly, Lz);
            
            field = this.draw_function(field(1+mx_neg:end-mx_pos, 1+my_neg:end-my_pos, 1+mz_neg:end-mz_pos));
                                   
            if this.draw_abs
                field = abs(field);
                plot_title = strcat('|', this.name, '|', ', \lambda =', num2str(this.gi.wvlen*L0*dlu_scale), dlu);
            else 
                field = real(field);
                plot_title = strcat(this.name, ', \lambda =', num2str(this.gi.wvlen*L0*dlu_scale), dlu);
            end

                        
            % Pad the end boundaries.
            field(end+1,:,:) = field(end,:,:);
            field(:,end+1,:) = field(:,end,:);
            field(:,:,end+1) = field(:,:,end);

            field = permute(field, [2 1 3]);  % make field compatible to meshgrid()
                        
            field = double(field);  % CData should be in double precision

            h = patch(isosurface(x, y, z, field, isovalue));

            isonormals(x, y, z, field, h)
            set(h,'FaceColor',[0.5 0.5 0.5],'EdgeColor','none');
            
            if ~this.opaque
                alpha(this.opacity);
            end
            
            axis image;
            
            view(this.view_angle(1), this.view_angle(2));

            xlabel(strcat(AxisName(Xx), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));
            ylabel(strcat(AxisName(Yy), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));
            zlabel(strcat(AxisName(Zz), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));

            title(plot_title);
            
            drawnow;
        end        
        
        function h = plot(this, sx, sy, sz, cscale)
            const;

            L0 = this.gi.L0;
            dlu = this.gi.display_length_unit;
            dlu_scale = this.gi.dlu_scale;

            field = this.array;
            field = exp(sqrt(-1) * this.phase_angle) .* field;
            
            mx_neg = this.margin(Xx,Neg);
            mx_pos = this.margin(Xx,Pos);
            my_neg = this.margin(Yy,Neg);
            my_pos = this.margin(Yy,Pos);
            mz_neg = this.margin(Zz,Neg);
            mz_pos = this.margin(Zz,Pos);

            Lx = this.gi.L{this.grid_kind(Xx), Xx};
            Ly = this.gi.L{this.grid_kind(Yy), Yy};
            Lz = this.gi.L{this.grid_kind(Zz), Zz};

            Lx = Lx(1+mx_neg:end-mx_pos);
            Ly = Ly(1+my_neg:end-my_pos);
            Lz = Lz(1+mz_neg:end-mz_pos);

            [x, y, z] = meshgrid(Lx, Ly, Lz);
            
            field = this.draw_function(field(1+mx_neg:end-mx_pos, 1+my_neg:end-my_pos, 1+mz_neg:end-mz_pos));

            if this.draw_abs_cscale  % calculate [cmin cmax] from the phasors, before real(field) or abs(field) is taken
                cmax = max(abs(field(:)));
                if this.draw_abs
                    cmin = 0.0;
                else
                    cmin = -cmax;
                end
            end
                                    
            if this.draw_abs
                field = abs(field);
                plot_title = strcat('|', this.name, '|', ', \lambda =', num2str(this.gi.wvlen*L0*dlu_scale), dlu);
                colormap('hot');
            else 
                field = real(field);
                plot_title = strcat(this.name, ', \lambda =', num2str(this.gi.wvlen*L0*dlu_scale), dlu);
                colormap('jet');
            end

            if ~this.draw_abs_cscale  % calculate [cmin cmax] after real(field) or abs(field) is taken above
                cmax = max(real(field(:)));
                cmin = min(real(field(:)));
            end
            
            crange = [cmin cmax];
            if nargin >= 5
                if length(cscale)==1
                    crange = cscale * crange;
                elseif length(cscale)==2
                    crange = cscale;
                end
            end
            
            % Pad the end boundaries.
            field(end+1,:,:) = field(end,:,:);
            field(:,end+1,:) = field(:,end,:);
            field(:,:,end+1) = field(:,:,end);

            field = permute(field, [2 1 3]);  % make field compatible to meshgrid()
                        
            field = double(field);  % CData should be in double precision

            h = slice(x,y,z, field, sx, sy, sz);
            %colormap hsv;
            
            if this.draw_grid
                set(h,'EdgeAlpha',1)
            else
                set(h,'EdgeAlpha',0)
            end

            if this.opaque
                set(h,'FaceLighting','phong','FaceColor','interp', 'AmbientStrength',0.5)
%                light('Position',[-0.6 1 -0.2],'Style','infinite');            
            else
                alpha('color');
                set(h,'EdgeColor','none','FaceColor','interp', 'FaceAlpha','interp');
                alphamap('rampdown');
                alphamap('increase', this.opacity);
            end
            
            axis image;

            caxis(crange);
            if this.draw_colorbar
                colorbar;
            end
            
            for i = 1:length(this.geometry)
                polygon = this.geometry{i}.';
                line(polygon(1,:), polygon(2,:), polygon(3,:), 'Color', this.geometry_color, 'LineWidth', 0.5);
            end
            
            view(this.view_angle(1), this.view_angle(2));

            xlabel(strcat(AxisName(Xx), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));
            ylabel(strcat(AxisName(Yy), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));
            zlabel(strcat(AxisName(Zz), ' (', char(hex2dec('00D7')), ...  % 00D7 is a unicode character for the multiplication sign.
                num2str(L0*dlu_scale), dlu, ')'));

            title(plot_title);
            
            %drawnow;
        end
        
        function create_movie(this, filename, num_periods, framerate, sx, sy, sz, cscale, command_str)
            fig = figure;
            winsize = get(fig, 'Position');
            winsize(1:2) = [1 1];  % to prevent unwanted borders
            winsize(3:4) = [winsize(3)-1, winsize(4)-1];
            mat_movie = moviein(1, fig, winsize);
            set(fig, 'NextPlot', 'replacechildren');
        
            for ind_frame = 1:framerate*num_periods
                this.phase_angle = 2*pi*(ind_frame-1)/framerate;

                if nargin >= 8
                    this.plot(sx, sy, sz, cscale);
                else
                    this.plot(sx, sy, sz);
                end
                
                % Custom graphics commands passed as a string.
                eval(command_str);
                
                mat_movie(ind_frame) = getframe(fig, winsize);
            end
            
            movie2avi(mat_movie, filename);
        end
    end
end
