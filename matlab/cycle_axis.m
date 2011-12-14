function [Pp Qq Rr] = cycle_axis(normal)

const; 

Rr = normal;
Pp = mod(normal, Naxis) + 1;
Qq = mod(normal+1, Naxis) + 1;