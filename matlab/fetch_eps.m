function eps = fetch_eps(file_base_name, gi)

const; 

%% Read eps
eps_array = read_material(strcat(file_base_name, '.eps'));

% Extract each directional component, and encapsulate it by scalar3d class.
eps = scalar3d('eps', eps_array, [Prim Prim Prim], gi);
