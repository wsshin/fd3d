from math import floor, ceil 
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import Box
from material import Material
from object import Object
from source import PlaneSrc
from math import cos, pi

grid = Grid()
grid.set_length_unit(1e-9)
wvlen = 100
grid.set_wvlen(wvlen)

## Set up dx_prob, dy_prob, dz_prob
dL = 1.0
dx0 = dL
dy0 = dL
dz0 = dL
dx_prob = [dx0]*100
dy_prob = [dy0]*1
dz_prob = [dz0]*100

grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)

grid.set_Npml( (0, 0, 10) )
grid.set_BC( (Bloch, Bloch, PEC) )

grid.initialize()

'''The name of Simulation object must be sim.  This is assumed in fd3d.c, so the 
name should not be changed.'''
sim = Simulation(grid)
sim.set_BiCG_tol(1e-6)

## Set up objects.
# (No object)


## Set up sources.
z_src = 10
src = PlaneSrc(grid, Yy, Zz, 1, grid.get_N(Xx), grid.get_N(Yy))
src = src.translate(0, 0, grid.get_N(Zz)/2)
sim.append_source(src)
