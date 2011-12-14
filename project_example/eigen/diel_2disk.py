from math import floor, ceil 
from const import *  # constants such as Xx, Yy, Zz, PEC, PMC, Periodic
from simulation import Simulation
from grid import Grid
from shape import *
from material import Material
from object import Object
from source import PointSrc

grid = Grid()
grid.set_length_unit(1e-9)
#grid.set_wvlen(197)
grid.set_wvlen(float('inf'))
grid.set_wvlen_resonance(197)

dL = 1.0
dx0 = dL
dy0 = dL
dz0 = dL
dx_prob = [dx0]*500
dy_prob = [dy0]*1
dz_prob = [dz0]*400

grid.set_d_prob(Xx, dx_prob)
grid.set_d_prob(Yy, dy_prob)
grid.set_d_prob(Zz, dz_prob)

grid.set_Npml( (10, 0, 10) )
grid.set_BC( (PEC, Bloch, PEC) )

grid.initialize()

'''Create a Simulation object named <sim>.  CAUTION: the object must be named <sim> and should not be named differently, because this is assumed in fd3d.c.'''

sim = Simulation(grid)
sim.set_BiCG_tol(1e-6)
sim.set_BiCG_max_iter(-1)

eps = 147.7 - 0.01477j 
diel = Material('diel', 2, eps)
vac = Material('Vaccum',0)
 
radius = 10
disk = Circular(grid, radius).cylinder(Yy, 1)
disk = disk.center_at_middle()
disk = disk.translate(-20,0,0)
disk_diel = Object(disk,diel)
sim.append_object(disk_diel)

disk = Circular(grid, radius).cylinder(Yy, 1)
disk = disk.center_at_middle()
disk = disk.translate(20,0,0) 
disk_diel = Object(disk,diel)
sim.append_object(disk_diel)

#src = PointSrc(grid, Yy, 1.0)
#src = src.translate(grid.get_N(Xx)/2+12, 0, grid.get_N(Zz)/2+12)
#src= src.translate(0,0,-20)
#sim.append_source(src)
