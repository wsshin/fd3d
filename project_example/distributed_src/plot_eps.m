clear all; close all; clc
const;

%% Read GridInfo.
inputname = 'distsrc';
gi = read_gi(inputname);

%% Read the eps file.
s3d = fetch_eps(inputname, gi);
%s3d.margin = gi.Npml;

%% Get a slice.
s2d = s3d.get_slice(Yy, 1);

%% Plot the structure on the slice.
s2d.plot();
