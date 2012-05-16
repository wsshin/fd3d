# shape.py
"""
This module defines various classes that are used to specify the shapes of physical objects put 
in the simulation domain.
Two coordinate systems are used to specify the sizes and locations of shapes: the real coordinate system and the index coordinate system. 
Note that the origin of the real coordinate system coincides with the origin of the index coordinate system.
"""

__docformat__ = 'restructuredtext en'

from math import pi
from numpy import zeros, array, fromfile
from const import *

class Shape:
	def __init__(self, grid, isinside, on_real_axes = False, center = [0,0,0], translatable = True):
		"""
		Set an inequality describing the inside of this shape using lambda calculus.
		isinside should take three arguments and return a truth value.
		e.g. isinside = lambda x,y,z: 0<x<1 and 0<y<1 and 0<z<1
		Currently, on_real_axes is accepted only for objects with curved surfaces.
		"""
		self.grid = grid
		self.isinside_original = isinside
		if on_real_axes:
			self.isinside = lambda x,y,z: isinside(grid.get_L_at(Xx,x), grid.get_L_at(Yy,y), grid.get_L_at(Zz,z))  # x,y,z that are passed to contains() are indices, not real locations.
		else:
			self.isinside = lambda x,y,z: isinside(float(x), float(y), float(z))  # Make sure the stored lambda calculus works for floating points.
		assert(len(center)==3)
		self.c = [float(center[Xx]), float(center[Yy]), float(center[Zz])]  # real locations if on_real_axes == True
		self.on_real_axes = on_real_axes
		self.translatable = translatable
		
	def __add__(self, other):
		if (not self.on_real_axes and other.on_real_axes) or (self.on_real_axes and not other.on_real_axes):  
			"""If one is on real axes but the other is not, the result is not translatable.  
			Also it is in the index coordinate system."""
			return Shape(self.grid, lambda x,y,z: self.isinside(x,y,z) or other.isinside(x,y,z), False, [0,0,0], False)
		else: 
			"""If both are on real axes, or both are in the index coordinate system, the result is 
			translatable.  Also, the shared on_real_axes property is preserved."""
			return Shape(self.grid, lambda x,y,z: self.isinside_original(x,y,z) or other.isinside_original(x,y,z), self.on_real_axes, [0,0,0], True)

	def __neg__(self):
		return Shape(self.grid, lambda x,y,z: not self.isinside(x,y,z), self.on_real_axes, self.c, self.translatable)

  	def __sub__(self, other):
		if (not self.on_real_axes and other.on_real_axes) or (self.on_real_axes and not other.on_real_axes):  
			"""If one is on real axes but the other is not, the result is not translatable.  
			Also it is in the index coordinate system."""
			return Shape(self.grid, lambda x,y,z: self.isinside(x,y,z) and not other.isinside(x,y,z), False, [0,0,0], False)
		else: 
			"""If both are on real axes, or both are in the index coordinate system, the result is 
			translatable.  Also, the shared on_real_axes property is preserved."""
			return Shape(self.grid, lambda x,y,z: self.isinside_original(x,y,z) and not other.isinside_original(x,y,z), self.on_real_axes, [0,0,0], True)

	def intersect(self, other):
		if (not self.on_real_axes and other.on_real_axes) or (self.on_real_axes and not other.on_real_axes):  
			"""If one is on real axes but the other is not, the result is not translatable.  
			Also it is in the index coordinate system."""
			return Shape(self.grid, lambda x,y,z: self.isinside(x,y,z) and other.isinside(x,y,z), False, [0,0,0], False)
		else: 
			"""If both are on real axes, or both are in the index coordinate system, the result is 
			translatable.  Also, the shared on_real_axes property is preserved."""
			return Shape(self.grid, lambda x,y,z: self.isinside_original(x,y,z) and other.isinside_original(x,y,z), self.on_real_axes, [0,0,0], True)

	def contains(self, x, y, z):
		return self.isinside(x,y,z)

	def translate(self, dx, dy, dz):
		"""
		dx,dy,dz are translation in real locations if on_real_axes == True.
		Shape with on_real_axes == True should not be translated after it is added to, 
		intersected with, or subtracted from Shape with on_real_axes == False, and vice versa.  
		This is because Shape with on_real_axes == True loses the property after mixed with Shape 
		with on_real_axes == False."""
		assert(self.translatable)
		return Shape(self.grid, lambda x,y,z: self.isinside_original(x-dx, y-dy, z-dz), self.on_real_axes, [self.c[Xx]+dx, self.c[Yy]+dy, self.c[Zz]+dz], True)

	def center_at_orig(self, axis = -1):
		if axis < 0:
			return self.translate(-self.c[Xx], -self.c[Yy], -self.c[Zz])
		elif axis == Xx:
			return self.translate(-self.c[Xx], 0, 0)
		elif axis == Yy:
			return self.translate(0, -self.c[Yy], 0)
		else:
			assert(axis == Zz)
			return self.translate(0, 0, -self.c[Zz])

	def center_at_middle(self, axis = -1):
		temp = self.center_at_orig(axis)		
		if self.on_real_axes:
			if axis < 0:
				return temp.translate(self.grid.get_L(Xx)/2, self.grid.get_L(Yy)/2, self.grid.get_L(Zz)/2)
			elif axis == Xx:
				return temp.translate(self.grid.get_L(Xx)/2, 0, 0)
			elif axis == Yy:
				return temp.translate(0, self.grid.get_L(Yy)/2, 0)
			else:
				assert(axis == Zz)
				return temp.translate(0, 0, self.grid.get_L(Zz)/2)
		else:
			if axis < 0:
				return temp.translate(self.grid.get_N_float(Xx)/2, self.grid.get_N_float(Yy)/2, self.grid.get_N_float(Zz)/2)
			elif axis == Xx:
				return temp.translate(self.grid.get_N_float(Xx)/2, 0, 0)
			elif axis == Yy:
				return temp.translate(0, self.grid.get_N_float(Yy)/2, 0)
			else:
				assert(axis == Zz)
				return temp.translate(0, 0, self.grid.get_N_float(Zz)/2)
	
	def draw(self, C, normal_dir, intercept, val):
		rows, cols = C.shape
		for i in xrange(rows):
			for j in xrange(cols):
				args = []
				if normal_dir == Xx:
					args = [intercept, i, j]
				elif normal_dir == Yy:
					args = [j, intercept, i]
				else:
					assert(normal_dir == Zz)
					args = [i, j, intercept]
				if apply(self.isinside, args):
					C[i,j] = val

	def draw3d(self, C, val):
		Nx,Ny,Nz = C.shape
		for k in xrange(Nz):
			for j in xrange(Ny):
				for i in xrange(Nx):
					args = [i,j,k]
					if apply(self.isinside, args):
						C[i,j,k] = val

class Box(Shape):
  	def __init__(self, grid, Lx=-1.0, Ly=-1.0, Lz=-1.0, on_real_axes = False):
		if Lx < 0:
			self.Lx = grid.get_N_float(Xx)
			self.Ly = grid.get_N_float(Yy)
			self.Lz = grid.get_N_float(Zz)
		else: 
			assert(Lx>=0 and Ly>=0 and Lz>=0)
			self.Lx = float(Lx)
			self.Ly = float(Ly)
			self.Lz = float(Lz)

		Shape.__init__(self, grid, lambda x,y,z: 0<=x<=self.Lx and 0<=y<=self.Ly and 0<=z<=self.Lz, on_real_axes, [self.Lx/2, self.Ly/2, self.Lz/2])

class Cube(Box):
  	def __init__(self, grid, L):
		Box.__init__(self, grid, L, L, L)

class Ellipsoid(Shape):
	def __init__(self, grid, a, b, c, on_real_axes = False):
		Shape.__init__(self, grid, lambda x,y,z: (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1, on_real_axes)

class Sphere(Ellipsoid):
  	def __init__(self, grid, r, on_real_axes = False):
		Ellipsoid.__init__(self, grid, r, r, r, on_real_axes)

class Point(Shape):
	def __init__(self, grid, x0=0, y0=0, z0=0):
		Shape.__init__(self, grid, lambda x,y,z: x==x0 and y==y0 and z==z0)

class Slab(Shape):
  	def __init__(self, grid, normal, t, on_real_axes = False):
		"""
		normal: normal direction, t: thickness
		"""
		c = []
		if normal == Xx:
			c = [float(t)/2, 0, 0]
		elif normal == Yy:
			c = [0, float(t)/2, 0]
		else:
			assert(normal == Zz)
			c = [0, 0, float(t)/2]

		Shape.__init__(self, grid, lambda x,y,z: (normal==Xx and 0<=x<=t) or (normal==Yy and 0<=y<=t) or (normal==Zz and 0<=z<=t), on_real_axes, c)
		self.normal = normal
		self.t = float(t)

class Plane(Shape):
	def __init__(self, grid, normal):
		Shape.__init__(self, grid, lambda x,y,z: (normal==Xx and x==0.0) or (normal==Yy and y==0.0) or (normal==Zz and z==0.0))
		self.normal = normal

class CrossSectional:
	"""
	Abstract factory for planar and cylindrical shapes of a specific cross section.
	This is not Shape, but has factory methods for planar and cylindrical shapes
	"""
	def __init__(self, grid, isinside, on_real_axes = False, center = [0,0]):
		"""
		isinside is a lambda calculus object that determines if a given 2D point 
		(p,q) is inside this specific cross sectional area.  
		e.g. isinside = lambda p,q: 0<p<1 and 0<q<1
		"""
		self.grid = grid
		self.isinside = lambda p,q: isinside(float(p), float(q))  # Make sure the stored lambda calculus works for floating points.
		assert(len(center)==2)
		self.on_real_axes = on_real_axes
		self.c = [float(center[0]), float(center[1])]

	def plane(self, normal):
		plane = Plane(self.grid, normal)
		if normal == Xx:
			shape = plane.intersect(Shape(self.grid, lambda x,y,z: self.isinside(y,z), self.on_real_axes))
			shape.c = [0.0, self.c[0], self.c[1]]
		elif normal == Yy:
			shape = plane.intersect(Shape(self.grid, lambda x,y,z: self.isinside(z,x), self.on_real_axes))
			shape.c = [self.c[1], 0.0, self.c[0]]
		else:
			assert(normal == Zz)
			shape = plane.intersect(Shape(self.grid, lambda x,y,z: self.isinside(x,y), self.on_real_axes))
			shape.c = [self.c[0], self.c[1], 0.0]
		return shape

	def cylinder(self, normal, t):
		slab = Slab(self.grid, normal, t, self.on_real_axes)
		if normal == Xx:
			shape = slab.intersect(Shape(self.grid, lambda x,y,z: self.isinside(y,z), self.on_real_axes))
			shape.c = [float(t)/2, self.c[0], self.c[1]]
		elif normal == Yy:
			shape = slab.intersect(Shape(self.grid, lambda x,y,z: self.isinside(z,x), self.on_real_axes))
			shape.c = [self.c[0], float(t)/2, self.c[1]]
		else:
			assert(normal == Zz)
			shape = slab.intersect(Shape(self.grid, lambda x,y,z: self.isinside(x,y), self.on_real_axes))
			shape.c = [self.c[0], self.c[1], float(t)/2]
		return shape

class Elliptical(CrossSectional):
	def __init__(self, grid, a, b, on_real_axes = False):
		CrossSectional.__init__(self, grid, lambda p,q: (p/a)**2 + (q/b)**2 <= 1, on_real_axes)

class Circular(Elliptical):
	def __init__(self, grid, r, on_real_axes = False):
		Elliptical.__init__(self, grid, r, r, on_real_axes)

class Rectangular(CrossSectional):
	def __init__(self, grid, Lp, Lq):
		CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq, False, [float(Lp)/2,float(Lq)/2])

class Square(Rectangular):
	def __init__(self, grid, L):
		Rectangular.__init__(self, grid, L, L)

class RightTriangular(CrossSectional):
	NE, NW, SE, SW, = range(4)
	def __init__(self, grid, Lp, Lq, orientation=SW, include_diag=True):
		if orientation==RightTriangular.SW:
			if include_diag:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp+q/Lq<=1)
			else:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp+q/Lq<1)
		elif orientation==RightTriangular.NE:
			if include_diag:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp+q/Lq>=1)
			else:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp+q/Lq>1)
		elif orientation==RightTriangular.SE:
			if include_diag:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp>=q/Lq)
			else:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp>q/Lq)
		else:
			assert(orientation==RightTriangular.NW)
			if include_diag:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp<=q/Lq)
			else:
				CrossSectional.__init__(self, grid, lambda p,q: 0<=p<=Lp and 0<=q<=Lq and p/Lp<q/Lq)

class SquareTriangular(RightTriangular):
	def __init__(self, grid, L, lower=True):
		RightTriangular.__init__(self, grid, L, L, lower)

class Planar(CrossSectional):
	"""
	Assume that the given file represents a 2D array of booleans indicating whether each 
	integral point is contained in a cross section.
	"""
	def __init__(self, grid, normal, filename):
		self.normal = normal
		if self.normal == Xx:
			Lp, Lq = grid.get_N(Yy), grid.get_N(Zz)
		elif self.normal == Yy:
			Lp, Lq = grid.get_N(Zz), grid.get_N(Xx)
		else:
			assert(self.normal==Zz)
			Lp, Lq = grid.get_N(Xx), grid.get_N(Yy)
		cross_section = fromfile(filename, dtype=bool).reshape(Lp+1,Lq+1)
		CrossSectional.__init__(self,  grid, lambda p,q: cross_section[p,q], False, [float(Lp)/2,float(Lq)/2])

	def plane(self, normal):
		assert(normal==self.normal)
		return CrossSectional.plane(self, normal)

	def cylinder(self, normal, t):
		assert(normal==self.normal)
		return CrossSectional.cylinder(self, normal, t)

if __name__ == '__main__':
   	from viewer import view
   	from numpy import zeros
	from grid import Grid

	grid = Grid()
	grid.set_length_unit(1e-6)
	grid.set_wvlen(1.55)
	grid.set_d_prob(Xx, [1.0]*100)
	grid.set_d_prob(Yy, [1.0]*100)
	grid.set_d_prob(Zz, [1.0]*100)
	grid.set_BC(PEC)
	grid.set_Npml(0)
	grid.initialize()
									
   	C = zeros((grid.get_N(Xx), grid.get_N(Yy)),'f')
	box = Cube(grid, 50).translate(25, 25, 25)
	space = Sphere(grid, 10).translate(50, 50, 50)
	space = Circular(grid, 20).cylinder(Zz, 50)
	#space = space.center_at_orig()
	#space = space.translate(50, 50, 50)
	space = space.translate(50, 50, 25)
	cavity = box - space

	cavity.draw(C, Zz, 50, 1.0)
  	view(C, 'x', 'y')
