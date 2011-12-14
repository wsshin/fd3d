clear all; close all; clc
const;

%% Read GridInfo.
inputname = 'distsrc';
gi = retrieve_gi(inputname);

%% Read the eps file.
eps = read_eps(inputname, gi);

%% Set structure plot parameters.
eps.margin = gi.Npml;
%eps.opaque = false;
%eps.opacity = 0.4;

%% Plot the structure and field.
eps.plot_iso(0);
