#!/usr/bin/env python
import sys
import commands
sys.path.insert(0, commands.getoutput('source ~/.fd3d;echo $FD3D_ROOT') + '/bin')
from const import *

'''Read the input file.'''
name = sys.argv[1]
item = sys.argv[2]
basename = name[0:-3]
file = open(name, 'r')
exec(file)

if item == 'L0':
	value = sim.grid.get_length_unit()
elif item == 'wvlen':
	value = sim.get_wvlen()
elif item == 'BC':
	value = '[[' + str(sim.get_BC(Xx,Neg)+1) + ', ' + str(sim.get_BC(Xx,Pos)+1) + ']; [' + str(sim.get_BC(Yy,Neg)+1) + ', ' + str(sim.get_BC(Yy,Pos)+1) + ']; [' + str(sim.get_BC(Zz,Neg)+1) + ', ' + str(sim.get_BC(Zz,Pos)+1) + ']]'  # str2num in MATLAB requires ';' to separate rows
elif item == 'Npml':
	value = '[[' + str(sim.get_Npml(Xx,Neg)) + ', ' + str(sim.get_Npml(Xx,Pos)) + ']; [' + str(sim.get_Npml(Yy,Neg)) + ', ' + str(sim.get_Npml(Yy,Pos)) + ']; [' + str(sim.get_Npml(Zz,Neg)) + ', ' + str(sim.get_Npml(Zz,Pos)) + ']]'  # str2num in MATLAB requires ';' to separate rows
elif item == 'dx_prim':
	value = sim.grid.d_prim[Xx]
elif item == 'dy_prim':
	value = sim.grid.d_prim[Yy]
elif item == 'dz_prim':
	value = sim.grid.d_prim[Zz]
else:
	assert('Not a vaild item name')

sys.stdout.write(str(value))
