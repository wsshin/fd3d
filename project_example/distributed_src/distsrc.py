from math import floor, ceil 
from mio import loadmat
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import Box
from object import Object
from source import PlaneDistributedSrc

grid = Grid()
grid.set_length_unit(1e-9)
grid.set_wvlen(1550)

# Set up dx_prob, dy_prob, dz_prob
dl = 1.0
dL = 20.0

Nx_fine = 100
Ny_fine = 100
Nx_coarse = 10
Ny_coarse = 10
Nz_coarse = 40
N_transition = 16
r = (dL/dl)**(1.0/(N_transition+1))  # scaling factor of increasing grid edge length

dx_prob = [dl]*Nx_fine
dx = dl
for i in xrange(N_transition):
	dx *= r
	dx_prob += [dx]
dx_prob += [dL]*Nx_coarse

dy_prob = [dl]*Ny_fine
dy = dl
for j in xrange(N_transition):
	dy *= r
	dy_prob += [dy]
dy_prob += [dL]*Ny_coarse

dz_prob = [dL]*Nz_coarse

grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)

grid.set_Npml( ((0,10), (0,10), 10) )
grid.set_BC( (PEC, (PMC,PEC), PEC) )
grid.initialize()

'''Create a Simulation object named <sim>.  CAUTION: the object must be named <sim> and should not be named differently, because this is assumed in fd3d.c.'''
sim = Simulation(grid)
sim.set_BiCG_tol(1e-6)
sim.set_BiCG_max_iter(-1)

# Set up objects.
Ag = sim.get_material('Johnson/Ag', 2)
silica = sim.get_material('Palik/SiO2', 1)

Nx_slot_half = 24.5
Ny_slot_half = 25

bg = Box(grid)  # background
bg_silica = Object(bg, silica)
sim.append_object(bg_silica)

film = Box(grid, grid.get_N(Xx), Ny_slot_half, grid.get_N(Zz))
film_Ag = Object(film, Ag)
sim.append_object(film_Ag)

slot = Box(grid, Nx_slot_half, Ny_slot_half, grid.get_N(Zz))
slot_silica = Object(slot, silica) 
sim.append_object(slot_silica)

## Set up sources.
src_data = loadmat('distsrc_in.mat')
hx = src_data['hx']
hy = src_data['hy']

src_Kxy = PlaneDistributedSrc(grid, Zz, hy, -hx)

Nz_src = 5
Npml_zn = grid.get_Npml(Zz, Neg)

src_Kxy = src_Kxy.translate(0, 0, Npml_zn+Nz_src)

sim.append_source(src_Kxy)
