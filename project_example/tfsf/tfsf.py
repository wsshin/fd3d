from math import floor, ceil 
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import Box, Circular
from material import Material
from object import Object
from source import PlaneSrc
from math import cos, pi
from sourcetfsf import PlaneIncidentE

grid = Grid()
grid.set_length_unit(1e-9)
wvlen = 200
grid.set_wvlen(wvlen)

## Set up dx_prob, dy_prob, dz_prob
dL = 10.0
dx0 = dL
dy0 = dL
dz0 = dL
dx_prob = [dx0]*100
dy_prob = [dy0]*1
dz_prob = [dz0]*100
#dx_prob = [dx0]*1
#dy_prob = [dy0]*1
#dz_prob = [dz0]*2000

grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)

grid.set_Npml( (10, 0, 10) )
#grid.set_Npml( (0) )
grid.set_BC( (PEC, Bloch, PEC) )

grid.initialize()

Nx = grid.get_N(Xx)
Ny = grid.get_N(Yy)
Nz = grid.get_N(Zz)

'''The name of Simulation object must be sim.  This is assumed in fd3d.c, so the 
name should not be changed.'''
sim = Simulation(grid)
sim.set_BiCG_tol(1e-6)
#sim.set_BiCG_max_iter(3*Nx*Ny*Nz+10)
#sim.set_BiCG_tol(1e-6)
#sim.set_sol_reference("plane_normal.E")

## Set up objects.
n_Ag = 0.469-9.32j  # refractive index of silver
eps_Ag = n_Ag * n_Ag
#eps_Ag = -1e6
Ag = Material('Silver', 2, eps_Ag)

radius = 10
cylinder = Circular(grid, radius).cylinder(Yy, Ny)
cylinder = cylinder.center_at_middle()
cylinder_Ag = Object(cylinder, Ag)

sim.append_object(cylinder_Ag)

## Set up sources.
#z_src = 10
#src = PlaneSrc(grid, Yy, Zz, 1, grid.get_N(Xx), grid.get_N(Yy))
#src = src.translate(0, 0, grid.get_N(Zz)/2)
#sim.append_source(src)
box_tf = Box(grid, Nx-60, Ny, Nz-60).center_at_middle()
#box_tf = Box(grid, Nx, Ny, Nz).center_at_middle()
Einc = PlaneIncidentE(grid, Yy, [0.0, 0.0, 1.0], 1.0, box_tf)
sim.set_incidentE(Einc)

