function [e h] = read_eh2d(file_base_name, normal, intercept, gi, comp_to_get)

const; 

if nargin <= 4
    comp_to_get = [true true true; true true true];
end

ex = []; ey = []; ez = [];
hx = []; hy = []; hz = [];

%% Read E field.
if any(comp_to_get(Efield,:))
    E = read_solution2d(strcat(file_base_name, '.E'), normal, intercept, gi);
    E = reshape(E, Naxis, []);

    % Extract each directional component, and encapsulate it by scalar3d class.
    if comp_to_get(Efield, Xx)
        ex = scalar2d('Ex', E(Xx,:), normal, intercept, [Dual Prim Prim], gi);
    end

    if comp_to_get(Efield, Yy)
        ey = scalar2d('Ey', E(Yy,:), normal, intercept, [Prim Dual Prim], gi);
    end

    if comp_to_get(Efield, Zz)
        ez = scalar2d('Ez', E(Zz,:), normal, intercept, [Prim Prim Dual], gi);
    end

    clear E;
end
e = {ex; ey; ez};

%% Read H field.
if any(comp_to_get(Hfield,:))
    H = read_solution2d(strcat(file_base_name, '.H'), normal, intercept, gi);
    H = reshape(H, Naxis, []);

    % Extract each directional component, and reshape it into 3-dimensional array.
    if comp_to_get(Hfield, Xx)
        hx = scalar2d('Hx', H(Xx,:), normal, intercept, [Prim Dual Dual], gi);
    end
    
    if comp_to_get(Hfield, Yy)
        hy = scalar2d('Hy', H(Yy,:), normal, intercept, [Dual Prim Dual], gi);
    end
    
    if comp_to_get(Hfield, Zz)
        hz = scalar2d('Hz', H(Zz,:), normal, intercept, [Dual Dual Prim], gi);
    end

    clear H;
end
h = {hx; hy; hz};