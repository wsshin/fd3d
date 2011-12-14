function pr = read_power(file_base_name, normal, intercept, rect, gi)
% Calculate the power flux through a rectangular patch "rect" normal to 
% "normal" direction at the index "intercept" in the "normal" axis.
% The rectangular patch is defined on the primary grid.  
% "file_base_name" is the input file name without ".py".
% "rect" is [p q Np Nq], where p and q are the primary grid indices of the 
% lower-left corner of the rectangle, and Np and Nq are the number of grid
% edges on the sides of the rectangle.
% (normal, p, q) forms the cyclic permutation of (Xx, Yy, Zz).  For example, 
% if normal = Yy, then p = Zz and q = Xx.
const;

assert(isequal(size(rect), [1 4]));
p = rect(1); q = rect(2); Np = rect(3); Nq = rect(4);

sr = read_poynting(file_base_name, normal, intercept, gi);
sr_patch = sr.array(p:p+Np-1,q:q+Nq-1);

[Pp, Qq, Rr] = cycle_axis(normal);
dLp_patch = gi.dL{Prim, Pp}; dLp_patch = dLp_patch(p:p+Np-1);
dLq_patch = gi.dL{Prim, Qq}; dLq_patch = dLq_patch(q:q+Nq-1);

area = dLp_patch.' * dLq_patch;
pr = sr_patch .* area;
pr = sum(pr(:));
