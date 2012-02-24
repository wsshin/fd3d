"""Import modules."""
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import *
from material import Material, Vac
from object import Object
from source import *

"""Specify a finite-difference grid."""
grid = Grid()
grid.set_length_unit(1e-9)
wvlen = 1550
grid.set_wvlen(wvlen)

dL = 50.0
dx0 = dL
dy0 = dL
dz0 = dL
dx_prob = [dx0]*100
dy_prob = [dy0]*100
dz_prob = [dz0]*1

grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)

grid.set_Npml( (10, 0, 0) )
grid.set_BC( (PEC, Bloch, Bloch) )

grid.initialize()

"""Create an instance of Simulation."""
sim = Simulation(grid)  # the name of the Simulation instance must be sim, as assumed in fd3d.c

"""Specify basic solver properties."""
sim.set_BiCG_tol(1e-6)

"""Specify materials."""
silica = sim.get_material('Palik/SiO2', 1)
n_Ag = 0.469-9.32j
eps_Ag = n_Ag**2
Ag = Material('Silver', 2, eps_Ag)

"""Specify objects."""
# background
bg = Box(grid)
bg_Vac = Object(bg, Vac)
sim.append_object(bg_Vac)

# silver cylinder
thickness = grid.get_N(Zz)
radius = 15
cylinder = Circular(grid, radius).cylinder(Zz, thickness)
cylinder = cylinder.center_at_middle()
cylinder_Ag = Object(cylinder, Ag)
sim.append_object(cylinder_Ag)

# silica rectangular box
box = Box(grid, 20, 10, thickness)
box = box.translate(grid.get_N(Xx)/2, grid.get_N(Yy)/2, 0)
box_silica = Object(box, silica)
sim.append_object(box_silica)

"""Set up sources."""
src = PlaneSrc(grid, Zz, Xx, 1, grid.get_N(Yy), grid.get_N(Zz))
src = src.translate(grid.get_N(Xx)/2, 0, 0)  # equivalent to src = src.center_at_middle(Xx)
sim.append_source(src)
