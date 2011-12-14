function p = read_power_box(file_base_name, box, gi)
% Calculate power flux going out of a box.
% "box" is [x, y, z, Nx, Ny, Nz], where x, y, z are the primary grid indices
% and Nx, Ny, Nz are the number of primary grid edges on the sides of the box.
const;

assert(isequal(size(box), [1 6]));
x = box(1); y = box(2); z = box(3);
Nx = box(4); Ny = box(5); Nz = box(6);

px_neg = read_power(file_base_name, Xx, x, [y z Ny Nz], gi);
px_pos = read_power(file_base_name, Xx, x+Nx, [y z Ny Nz], gi);

py_neg = read_power(file_base_name, Yy, y, [z x Nz Nx], gi);
py_pos = read_power(file_base_name, Yy, y+Ny, [z x Nz Nx], gi);

pz_neg = read_power(file_base_name, Zz, z, [x y Nx Ny], gi);
pz_pos = read_power(file_base_name, Zz, z+Nz, [x y Nx Ny], gi);

p = (px_pos - px_neg) + (py_pos - py_neg) + (pz_pos - pz_neg);
