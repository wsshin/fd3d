function relerr = compare_solutions(basename1, basename2, gi)

const;

e1 = read_eh(strcat(FD3D_ROOT, basename1), gi, [true true true; false false false]);
e2 = read_eh(strcat(FD3D_ROOT, basename2), gi, [true true true; false false false]);

Npml = gi.Npml;

% Ex
a1x = e1{Xx}.array;
a2x = e2{Xx}.array;
assert(all(size(a1x)==size(a2x)));

a1x = a1x(1+Npml(Xx,Neg):end-Npml(Xx,Pos)+1, 1+Npml(Yy,Neg):end-Npml(Yy,Pos)+1, 1+Npml(Zz,Neg):end-Npml(Zz,Pos)+1);
a2x = a2x(1+Npml(Xx,Neg):end-Npml(Xx,Pos)+1, 1+Npml(Yy,Neg):end-Npml(Yy,Pos)+1, 1+Npml(Zz,Neg):end-Npml(Zz,Pos)+1);

a1x = a1x(:);
a2x = a2x(:);

% Ey
a1y = e1{Yy}.array;
a2y = e2{Yy}.array;
assert(all(size(a1y)==size(a2y)));

a1y = a1y(1+Npml(Xx,Neg):end-Npml(Xx,Pos)+1, 1+Npml(Yy,Neg):end-Npml(Yy,Pos)+1, 1+Npml(Zz,Neg):end-Npml(Zz,Pos)+1);
a2y = a2y(1+Npml(Xx,Neg):end-Npml(Xx,Pos)+1, 1+Npml(Yy,Neg):end-Npml(Yy,Pos)+1, 1+Npml(Zz,Neg):end-Npml(Zz,Pos)+1);

a1y = a1y(:);
a2y = a2y(:);

% Ez
a1z = e1{Zz}.array;
a2z = e2{Zz}.array;
assert(all(size(a1z)==size(a2z)));

a1z = a1z(1+Npml(Xx,Neg):end-Npml(Xx,Pos)+1, 1+Npml(Yy,Neg):end-Npml(Yy,Pos)+1, 1+Npml(Zz,Neg):end-Npml(Zz,Pos)+1);
a2z = a2z(1+Npml(Xx,Neg):end-Npml(Xx,Pos)+1, 1+Npml(Yy,Neg):end-Npml(Yy,Pos)+1, 1+Npml(Zz,Neg):end-Npml(Zz,Pos)+1);

a1z = a1z(:);
a2z = a2z(:);

a1 = [a1x; a1y; a1z];
a2 = [a2x; a2y; a2z];

relerr = norm(a1-a2,inf) / norm(a2,inf);