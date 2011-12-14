clear all; close all; clc;
const;

%% Read GridInfo.
inputname = 'diel_2disk';
gi = retrieve_gi(inputname);

%% Import relevant matrices, and reorder matrix indices.
ind = PetscBinaryRead(strcat(inputname, '_ind'), 'complex');
[s ind] = sort(ind);

A = PetscBinaryRead(strcat(inputname, '_A'), 'complex');
HE = PetscBinaryRead(strcat(inputname, '_HE'), 'complex');

A = A(ind,ind);
HE = HE(ind,ind);

%% Calculate the eigenvalue and eigenmode.
wvlen = 204;
omega_guess = 2*pi/wvlen;
eigval_guess = omega_guess^2;

opts.disp = 2;
opts.isreal = 0;

[x eigval] = eigs(A, 1, eigval_guess, opts);  % x: the E-field
clear A

y = HE*x;  % y: the H-field
clear HE


%% Plot the eigenmode.
e = create_e(x, gi);
h = create_h(y, gi);

s3d = e{Yy};
%s3d = h{Xx};  % to visualize the H-field

normal_dir = Yy;
intercept = 1;
s2d = s3d.get_slice(normal_dir, intercept);

s2d.margin = gi.Npml;  % set to [0 0; 0 0; 0 0] to draw the PML region
%s2d.draw_abs = true;  % to draw the abs value
s2d.plot();

%% Display the eigenvalue.
eig_wvlen = 2*pi/sqrt(eigval);
disp(['eigen wvlen = ', num2str(eig_wvlen)]);
