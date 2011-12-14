clear all; close all; clc;
const;

%% Read GridInfo.
inputname = 'plane_normal';
%inputname = 'plane_oblique';
gi = retrieve_gi(inputname);

%% Read the solution.
normal = Yy;
intercept = 1;
[e h] = read_eh2d(inputname, normal, intercept, gi);
s2d = e{Yy};

%% Set plot parameters.
% relative_phase = 0.25;
% s2d.phase_angle = 2*pi*relative_phase;  % multiply a phase factor
s2d.margin = gi.Npml;  % set margins to exclude the PML regions
s2d.draw_abs_cscale = true;  % set true to use the color bar ranging from -max(amplitude) to +max(amplitude)
% s2d.draw_abs = true;  % set true to draw the amplitudes

%% Plot the solution.
s2d.plot();

%% Create a movie.
% s2d.draw_abs_cscale = true;  % this forces the color bar to be constant at all instance
% s2d.create_movie(strcat(inputname, '.avi'), 3, 30, 1);  % e.g., s2d.create_movie(filename, num_periods, framerate, cscale)
