from math import floor, ceil 
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import Box
from material import Material
from object import Object
from source import PointSrc

'''Create a Grid object named <grid>.'''
grid = Grid()


'''Set the length unit to <grid>.'''
grid.set_length_unit(1e-9)  # 1 nm


'''Set the wavelength to <grid> in terms of the previously set unit.'''
grid.set_wvlen(1550)


'''Set the dimensions of the Yee's grid in the problem domain to <grid>.   Note that the problem domain does not include PML regions.'''
# Create a list describing the number of Yee's cells and their sizes in each direction.
dx_prob = [1.0]*5  # 5 uniform Yee's cells at the center in the x direction
dx = 1.0
for j in xrange(2):  # 2 increasing-sized Yee's cells on both sides of the above uniform region
	dx *= 1.2
	dx_prob = [dx] + dx_prob + [dx]

dy_prob = [1.0]*9  # 9 uniform Yee's cells in the y direction
dz_prob = [1.0]*10  # 10 uniform Yee's cells in the z direction

# Set the created lists to <grid>.
grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)


'''Set the number of PML layers to <grid>.'''
grid.set_Npml(3)  # 3 PML layers on all six faces
# other usages:
# grid.set_Npml( (3, 4, 5) )  # 3, 4, 5 layers on faces normal to x, y, z, respectively
# grid.set_Npml( (3, (4,5), 6) )  # 3, 4, 5, 6 layers on faces normal to x, -y, +y, z, respectively


'''Set boundary conditions to <grid>.'''
grid.set_BC(PEC)  # PEC on all six faces
# other usages:
# grid.set_BC( (PEC, PMC, PEC) )  # PEC on faces normal to x, z, and PMC on faces normal to y
# grid.set_BC( (Bloch, (PMC, PEC), PEC) )  # Bloch, PMC, PEC, PEC on faces normal to x, -y, +y, z, respectively


'''Initialize <grid> after setting all the necessary parameters.'''
grid.initialize()


'''Create a Simulation object named <sim>.  CAUTION: the object must be named <sim> and should not be named differently, because this is assumed in fd3d.c.'''
sim = Simulation(grid)


'''Set the tolerance of the BiCG iterative solver to <sim>.  This is the target relative residual error.'''
sim.set_BiCG_tol(1e-6)


'''Set the maximum number of iterations of the BiCG iterative solver <sim>.  The vaule -1 makes it run forever until the tolerance is satisfied.'''
sim.set_BiCG_max_iter(-1)


'''Set phisical objects to <sim>.'''
# Create materials to use.
# silver
n_Ag = 0.469-9.32j  # refractive index of silver at wvlen = 1550 nm
eps_Ag = n_Ag * n_Ag
Ag = Material('Silver', 2, eps_Ag)

# vacuum
vac = Material('Vacuum', 0)  # default values of eps and mu: 1.0

# Create shapes, and create objects with shapes and materials.  Note that the dimensions of shapes are in terms of the number of cells, not the physical length.

# film shape
w_slot = 7;  # 7 cells, not 7 nm
h_slot = 5;  # 5 cells, not 5 nm
film = Box(grid, grid.get_N(Xx), grid.get_N(Yy), h_slot)
film = film.center_at_middle()

# film object
film_Ag = Object(film, Ag)

# Append the film object to <sim>.
sim.append_object(film_Ag)

# slot shape
slot = Box(grid, grid.get_N(Xx), w_slot, h_slot)
slot = slot.center_at_middle();

# slot object
slot_vac = Object(slot, vac) 

# Append the slot object to <sim>.
sim.append_object(slot_vac)


'''Set sources to <sim>.'''
z_src = 3
src = PointSrc(grid, Xx, 1)
src = src.translate(grid.get_N(Xx)/2, grid.get_N(Yy)/2, grid.get_Npml(Zz,Neg)+z_src)
sim.append_source(src)
