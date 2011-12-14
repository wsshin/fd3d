from math import floor, ceil 
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import Box
from shape import Circular
from material import Material
from object import Object
from source import PointSrc

grid = Grid()
grid.set_length_unit(1e-9)
grid.set_wvlen(1550)

# Set up dx_prob, dy_prob, dz_prob
dl = 1.0
dL = 20.0

Nx_fine = 20
Ny_fine = 20
Nz_fine = 20

dx_prob = [dl]*Nx_fine
dy_prob = [dl]*Ny_fine
dz_prob = [dl]*Nz_fine

grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)

grid.set_Npml( (0, 0, 10) )
grid.set_BC( (Bloch, Bloch, PEC) )

grid.initialize()

'''Create a Simulation object named <sim>.  CAUTION: the object must be named <sim> and should not be named differently, because this is assumed in fd3d.c.'''
sim = Simulation(grid)
sim.set_BiCG_tol(1e-6)
sim.set_BiCG_max_iter(-1)

# Set up objects.
n_Ag = 0.469-9.32j  # refractive index of silver
eps_Ag = n_Ag * n_Ag
Ag = Material('Silver', 2, eps_Ag)

vac = Material('Vacuum', 0)  # default values of eps and mu: 1.0

thickness = 5
slab = Box(grid, grid.get_N(Xx), grid.get_N(Yy), thickness)
slab = slab.center_at_middle()

radius = 5
hole = Circular(grid, radius).cylinder(Zz, thickness)
hole = hole.center_at_middle()

slab = slab - hole

slab_Ag = Object(slab, Ag)
sim.append_object(slab_Ag)

## Set up sources.
z_src = 5
src = PointSrc(grid, Xx, 1)
src = src.translate(grid.get_N(Xx)/2, grid.get_N(Yy)/2, grid.get_Npml(Zz,Neg)+z_src)
sim.append_source(src)
