function gi = retrieve_gi(basename)

const;

inputfile = strcat(basename, '.py');

L0 = str2num(python('retrieve_gi.py', inputfile, 'L0'));
wvlen = str2num(python('retrieve_gi.py', inputfile, 'wvlen'));
BC = str2num(python('retrieve_gi.py', inputfile, 'BC'));
Npml = str2num(python('retrieve_gi.py', inputfile, 'Npml'));

dx_prim = str2num(python('retrieve_gi.py', inputfile, 'dx_prim'));
dy_prim = str2num(python('retrieve_gi.py', inputfile, 'dy_prim'));
dz_prim = str2num(python('retrieve_gi.py', inputfile, 'dz_prim'));

gi = gridinfo(L0, wvlen, BC, Npml, dx_prim, dy_prim, dz_prim);
