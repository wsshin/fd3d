function e = create_e(x, gi, comp_to_get)

const; 

if nargin <= 2
    comp_to_get = [true true true];
end

ex = []; ey = []; ez = [];

%% Create E field.
if any(comp_to_get)
    E = reshape(x, Naxis, []);

    % Extract each directional component, and encapsulate it by scalar3d class.
    if comp_to_get(Xx)
        ex = scalar3d('Ex', E(Xx,:), [Dual Prim Prim], gi);
    end

    if comp_to_get(Yy)
        ey = scalar3d('Ey', E(Yy,:), [Prim Dual Prim], gi);
    end

    if comp_to_get(Zz)
        ez = scalar3d('Ez', E(Zz,:), [Prim Prim Dual], gi);
    end
end
e = {ex; ey; ez};
