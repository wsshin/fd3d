function x = read_solution(filename)
% Read the FD3D solution file (*.E or *.H) into MATLAB's single precision array.
% PETSc does not support writing single-precision floating point numbers, which means that 
% PetscBinaryRead.m in $PETSC_DIR/bin/matlab/ cannot read *.sp.E and *.sp.H files.  So this 
% script is written without using PetscBinaryRead.m

% Open the given file and read double-precision elements that are written in big-endian format.
fid = fopen(filename, 'r');
if isequal(filename(end-4:end-1),'.sp.')
    type = '2*float=>float';
else
    type = '2*double=>float';
end

% Skip the PETSc header for PETSc's Vec.
fread(fid, 1, type);  % this does not read two floats even though type is '2*float=>float'; it reads just one float.

% Read the solution.
x = fread(fid, type, 'ieee-be');


% Make a pair of double-precision elements into a single complex number.
x = reshape(x, 2, [])';
x = complex(x(:,1),x(:,2));

% Close the file.
fclose(fid);
