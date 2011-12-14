from shape import *
from numpy import ndarray
from cmath import exp
from itertools import imap
from operator import mul
from const import *

class Source:
	def __init__(self, grid, shape):
		if not grid.isinitialized():
			raise ValueError('Uninitialized grid given')
		self.grid = grid
		self.shape = shape
	
	def is_valid_at(self, axis, i, j, k):
		return self.shape.contains(i,j,k)
	
	def get_src_at(self, axis, i, j, k):
		'''In principle, the source value should be determined for a given position, because Jx, 
		Jy, Jz of the same cell are located at different positions.  For example, when the source 
		value at a position (3.5, 1, 2) is requested, Jx of the cell (3, 1, 2) should be returned.
		Therefore, to retrieve the source value, it seems like a position is sufficient 
		information.
		We do not follow this way, however.  The reason is that the equality of floating point
		numbers is not credible.  When the position (3.5, 1, 2) is passed, there is no easy way to
		decide whether 3.5 is truly 3+(1/2); the result can be different from one CPU architecture
		to another.
		Therefore, we do not merge the polarization information into the position.'''
		if self.is_valid_at(axis, i, j, k):
			return self.get_src_at_kernel(axis, i, j, k)
		else:
			return 0.0
	
	def get_src_at_kernel(self, axis, i, j, k):
		raise NotImplemented

	def translate(self, x0, y0, z0):
		return TranslateSrc(self, x0, y0, z0)

class PolarizedSource(Source):
	'''Source with a specific polarization'''
	def __init__(self, grid, polarization, shape):
		Source.__init__(self, grid, shape)
		self.polarization = polarization

	def is_valid_at(self, axis, i, j, k):
		return axis == self.polarization and Source.is_valid_at(self, axis, i, j, k)

class TranslateSrc(Source):
	def __init__(self, src, x0, y0, z0):
		Source.__init__(self, src.grid, src.shape.translate(x0,y0,z0))
		self.src = src
		self.x0 = x0
		self.y0 = y0
		self.z0 = z0
	
	def is_valid_at(self, axis, i, j, k):
		return self.src.is_valid_at(axis, i-self.x0, j-self.y0, k-self.z0)
	
	def get_src_at_kernel(self, axis, i, j, k):
		return self.src.get_src_at_kernel(axis, i-self.x0, j-self.y0, k-self.z0)
	
class PointSrc(PolarizedSource):
	def __init__(self, grid, polarization, value):
		PolarizedSource.__init__(self, grid, polarization, Point(grid))
		self.value = value
	
	def get_src_at_kernel(self, axis, i, j, k):
		return self.value

class PlaneSrc(PolarizedSource):
	def __init__(self, grid, polarization, normal, value, Lp=None, Lq=None):
		'''(normal, p, q) is a cyclic permutation of (x, y, z).  Therefore, if normal = Yy for 
		example, then Lp = Lz and Lq = Lx.'''
		self.value = value
		self.Lp = Lp
		self.Lq = Lq
		if Lp == None:
			Pp, Qq, Rr = cyclic_axis(normal)
			self.Lp = grid.get_N(Pp)
			self.Lq = grid.get_N(Qq)
		plane = Rectangular(grid, self.Lp, self.Lq).plane(normal)
		PolarizedSource.__init__(self, grid, polarization, plane)
	
	def get_src_at_kernel(self, axis, i, j, k):
		ind = [i, j, k]
		ind[self.polarization] += 0.5  # axis == self.polarization
		k_Bloch = [self.grid.get_k_Bloch(Xx), self.grid.get_k_Bloch(Yy), self.grid.get_k_Bloch(Zz)]
		'''Because dual grid points are exactly at the center of primary edges, applying get_L_at()
		to (ind += 0.5) gives the exact dual grid points.'''
		r = [self.grid.get_L_at(Xx,ind[Xx]), self.grid.get_L_at(Yy,ind[Yy]), self.grid.get_L_at(Zz,ind[Zz])]
		return self.value * exp(-1j * sum(imap(mul, k_Bloch, r)))

class PlaneDistributedSrc(Source):
	def __init__(self, grid, normal, value_array_p, value_array_q):
		'''(normal, p, q) is a cyclic permutation of (x, y, z).  Therefore, if normal = Yy for 
		example, then value_array_p = value_array_z and value_array_q = value_array_x.'''
		assert(value_array_p!=None or value_array_q!=None)
		if value_array_p!=None:
			assert(isinstance(value_array_p, ndarray) and value_array_p.ndim==2)
			Lp, Lq = value_array_p.shape
		if value_array_q!=None:
			assert(isinstance(value_array_q, ndarray) and value_array_q.ndim==2)
			Lp, Lq = value_array_q.shape
		if value_array_p!=None and value_array_q!=None:
			assert(value_array_p.shape==value_array_q.shape)
		Lp -= 1
		Lq -= 1
		self.Pp, self.Qq, self.normal = cyclic_axis(normal)
		self.value_array_p = value_array_p
		self.value_array_q = value_array_q
		plane = Rectangular(grid, Lp, Lq).plane(normal)
		Source.__init__(self, grid, plane)
	
	def get_src_at_kernel(self, axis, i, j, k):
		ind = [i, j, k]
		p = ind[self.Pp]
		q = ind[self.Qq]
		if axis == self.Pp:
			if self.value_array_p != None:
				return self.value_array_p[p][q]
			else:
				return 0.0
		elif axis == self.Qq:
			if self.value_array_q != None:
				return self.value_array_q[p][q]
			else:
				return 0.0
		else:
			assert(axis == self.normal)
			return 0.0
