function eps = fetch_eps_tensor(file_base_name, gi)

const; 

%% Read eps
eps_xyz = read_material(strcat(file_base_name, '.3eps'));
eps_xyz = reshape(eps_xyz, Naxis, []);

% Extract each directional component, and encapsulate it by scalar3d class.
eps_xx = scalar3d('eps_xx', eps_xyz(Xx,:), [Dual Prim Prim], gi);
eps_yy = scalar3d('eps_yy', eps_xyz(Yy,:), [Prim Dual Prim], gi);
eps_zz = scalar3d('eps_zz', eps_xyz(Zz,:), [Prim Prim Dual], gi);

clear eps_xyz;

eps = {eps_xx; eps_yy; eps_zz};
