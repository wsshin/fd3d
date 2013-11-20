clear all; close all; clear classes; clc;

%% Set flags.
inspect_only = false;
has_sol = false;  % true if solution files exist

%% Solver Options
solveropts.method = 'inputfile';
filenamebase = 'pointsrc_2d';
solveropts.filenamebase = filenamebase;

if ~has_sol  % solution files do not exist
	%% Input Files
	[~, ~, obj_array, src_array] = maxwell_run(...
		'OSC', 1e-9, 1550, ...
		'DOM', {'vacuum', 'none', 1.0}, [-60, 60; -60, 60; 0, 1], 2, BC.p, [0 0 0], ...
		'SRCJ', PointSrc(Axis.z, [0, 0, 0.5]), ...
		solveropts, inspect_only);

	if ~inspect_only
		save(filenamebase, 'obj_array', 'src_array');
	end
else  % solution files exist
	[E, H] = read_output(filenamebase);
	load(filenamebase);  % read obj_array and src_array
	vis2d(E{Axis.z}, obj_array, src_array);
end
