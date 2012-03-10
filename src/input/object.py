class Object:
	def __init__(self, shape, material):
		self.shape = shape
		self.material = material

	def get_material(self):
		return self.material
	
	def get_eps(self):
		return self.material.get_eps()

	def get_mu(self):
		return self.material.get_mu()
	
	def get_color_index(self):
		return self.material.get_color_index()

	def contains(self, x, y, z):
		return self.shape.contains(x,y,z)
	
	def translate(self, x0, y0, z0):
		return Object(self.shape.translate(x0, y0, z0), self.material)

	def draw(self, C, normal_dir, pos):
		self.shape.draw(C, normal_dir, pos, self.material.get_color_index())

	def draw3d(self, C):
		self.shape.draw3d(C, self.material.get_color_index())

if __name__ == '__main__':
	from viewer import view
	from numpy import zeros
	from material import Material
	from shape import Cube, Sphere
	from grid import Grid
	from const import *

	grid = Grid()
	grid.set_length_unit(1e-6)
	grid.set_wvlen(1.55)
	grid.set_d_prob(Xx, [1.0]*100)
	grid.set_d_prob(Yy, [1.0]*100)
	grid.set_d_prob(Zz, [1.0]*100)
	grid.set_BC(PEC)
	grid.set_Npml(0)
	grid.initialize()

	C = zeros((100,100),'f')
	
	Si = Material("Silicon", 1)
	Si.eps = 11.8

	box = Cube(grid, 50).translate(25.5, 25.5, 25.5)
	space = Sphere(grid, 10).translate(50, 50, 50)
	cavity = box - space

	cavity_Si = Object(cavity, Si).translate(10, 0, 0)

	cavity_Si.draw(C, Zz, 50)
	view(C, 'X', 'Y')
