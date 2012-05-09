clear all; close all; clc
const;

%% Read GridInfo.
inputname = 'distsrc';
gi = read_gi(inputname);


%% Read the solution.
[e h] = fetch_eh(inputname, gi);
s3d = h{Yy};

%% Read the eps file.
eps = fetch_eps(inputname, gi);

%% Set structure plot parameters.
eps.margin = gi.Npml;
eps.opaque = false;
eps.opacity = 0.4;

%% Set field plot parameters.
% relative_phase = 0.25;
% s3d.phase_angle = 2*pi*relative_phase;  % multiply a phase factor
s3d.margin = gi.Npml;  % set margins to exclude the PML regions
s3d.draw_abs_cscale = true;  % set true to use the color bar ranging from -max(amplitude) to +max(amplitude)
% s3d.draw_abs = true;  % set true to draw the amplitudes
% s3d.opaque = false;
% s3d.opacity = 0.2;  % set the opacity
% s3d.draw_colorbar = true;
s3d.view_angle = [135, -45];

%% Plot the structure and field.
eps.plot_iso(0);

hold on
s3d.plot(0, 0, []);  % e.g., s3d.plot([x1 ... xp], [y1 ... yq], [z1 ... zr])
hold off


%% Create a movie.
% s3d.draw_abs_cscale = true;  % this forces the color bar to be constant at all instance
% s3d.create_movie(strcat(inputname, '.avi'), 3, 30, 0, 0, 700);  %  e.g., s3d.create_movie(filename, num_periods, framerate, [x1 ... xp], [y1 ... yq], [z1 ... zr], cscale);
