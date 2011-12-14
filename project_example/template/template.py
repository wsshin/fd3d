from math import floor, ceil 
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import Box
from material import Material
from object import Object
from source import PointSrc

grid = Grid()
grid.set_length_unit(###)
grid.set_wvlen(###)

# dx_prob = 
# dy_prob = 
# dz_prob = 

grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)

grid.set_Npml(###)
grid.set_BC(###)
grid.initialize()

'''Create a Simulation object named <sim>.  CAUTION: the object must be named <sim> and should not be named differently, because this is assumed in fd3d.c.'''
sim = Simulation(grid)
sim.set_BiCG_tol(###)
sim.set_BiCG_max_iter(###)

# n_Ag = 0.469-9.32j  # refractive index of silver at wvlen = 1550 nm
# eps_Ag = n_Ag * n_Ag
# Ag = Material('Silver', 2, eps_Ag)
# w_slot = 7;
# h_slot = 5;
# film = Box(grid, grid.get_N(Xx), grid.get_N(Yy), h_slot)
# film = film.center_at_middle()
# film_Ag = Object(film, Ag)

sim.append_object(###)

# src = PointSrc(grid, Zz, 1.0)
# src = src.translate(grid.get_N(Xx)/2, grid.get_N(Yy)/2, grid.get_N(Zz)/2)

sim.append_source(###)
