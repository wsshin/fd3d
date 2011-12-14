function x = read_solution2d(filename, normal, intercept, gi)
% Read a slice of the FD3D solution file (*.E or *.H) into MATLAB's single precision array.

const;

Nx = gi.N(Xx);
Ny = gi.N(Yy);
Nz = gi.N(Zz);

if isequal(filename(end-4:end-1),'.sp.')
    type_base = '*float=>float';  % a number will be prepended to become, e.g., '10*float=>float'
    elem_size = 4;  % # of bytes per element (= float)
else
    type_base = '*double=>float';  % a number will be prepended to become, e.g., '10*float=>float'
    elem_size = 8;  % # of bytes per element (= double)
end

fid = fopen(filename, 'r');

% Remove the PETSc header for PETSc's Vec.
fread(fid, 1, strcat(num2str(2), type_base));  % this does not read two floats even though type is '2*float=>float'; it reads just one float.

if normal==Xx
    offset = 2*Naxis*(intercept-1);  % # of offset elements of "type"
    type = strcat(num2str(2*Naxis), type_base);
    count = 2*Naxis*Ny*Nz;  % # of elements to read
    skip = 2*Naxis*(Nx-1) * elem_size;  % # of bytes to skip between elements
elseif normal==Yy
    offset = 2*Naxis*Nx*(intercept-1);  % # of offset elements of "type"
    type = strcat(num2str(2*Naxis*Nx), type_base);
    count = 2*Naxis*Nz*Nx;  % # of elements to read
    skip = 2*Naxis*Nx*(Ny-1) * elem_size;  % # of bytes to skip between elements
else
    assert(normal==Zz);
    offset = 2*Naxis*Nx*Ny*(intercept-1);  % # of offset elements of "type"
    type = strcat(num2str(2*Naxis*Nx*Ny), type_base);
    count = 2*Naxis*Nx*Ny;  % # of elements to read
    skip = 0;  % # of bytes to skip between elements
end

fread(fid, offset, type);  % default skip is 0 byte.
[x, read_count] = fread(fid, count, type, skip, 'ieee-be');

assert(read_count==count);
fclose(fid);


% Make a pair of real numbers into a single complex number.
x = reshape(x, 2, [])';
x = complex(x(:,1),x(:,2));

% Ensure the right-hand rule of the 3D Cartesian coordinates.
if normal==Yy
    x = reshape(x, Naxis, Nx, Nz);
    x = permute(x, [1 3 2]);  % now the fastest-varying index is the z-index rather than the x-index.
    x = x(:);
end
