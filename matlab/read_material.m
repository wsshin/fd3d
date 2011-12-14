function x = read_material(filename)

const;

% Open the given file and read double-precision elements that are written in big-endian format.
fid = fopen(filename, 'r');
x = fread(fid, '*float');
%[x count] = fread(fid, '*float', 'ieee-le');


% Close the file.
fclose(fid);

% Make a pair of double-precision elements into a singel complex number.
x = reshape(x, 2, [])';
x = complex(x(:,1),x(:,2));