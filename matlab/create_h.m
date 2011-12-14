function h = create_h(x, gi, comp_to_get)

const; 

if nargin <= 2
    comp_to_get = [true true true];
end

hx = []; hy = []; hz = [];

%% Create H field.
if any(comp_to_get)
    H = reshape(x, Naxis, []);

    % Extract each directional component, and reshape it into 3-dimensional array.
    if comp_to_get(Xx)
        hx = scalar3d('Hx', H(Xx,:), [Prim Dual Dual], gi);
    end
    
    if comp_to_get(Yy)
        hy = scalar3d('Hy', H(Yy,:), [Dual Prim Dual], gi);
    end
    
    if comp_to_get(Zz)
        hz = scalar3d('Hz', H(Zz,:), [Dual Dual Prim], gi);
    end
end
h = {hx; hy; hz};