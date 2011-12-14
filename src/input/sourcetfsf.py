from shape import Box
from const import *
from cmath import exp
from itertools import imap
from operator import mul

class IncidentE:
	def __init__(self, grid, shape):
		if not grid.isinitialized():
			raise ValueError('Uninitialized grid given')
		self.grid = grid
		self.shape = shape

	def is_valid_at(self, axis, i, j, k):
		return self.shape.contains(i,j,k)
	
	def get_E_at(self, axis, i, j, k):
		'''In principle, the E-field value should be determined for a given position, because Ex, 
		Ey, Ez of the same cell are located at different positions.  For example, when the E-field 
		value at a position (3.5, 1, 2) is requested, Ex of the cell (3, 1, 2) should be returned.
		Therefore, to retrieve the source value, it seems like a position is sufficient 
		information.
		We do not follow this way, however.  The reason is that the equality of floating point
		numbers is not credible.  When the position (3.5, 1, 2) is passed, there is no easy way to
		decide whether 3.5 is truly 3+(1/2); the result can be different from one CPU architecture
		to another.
		Therefore, we do not include the polarization information into the position.'''
		if self.is_valid_at(axis, i, j, k):
			return self.get_E_at_kernel(axis, i, j, k)
		else:
			return 0.0

	def get_E_at_kernel(self, axis, i, j, k):
		raise NotImplemented

class PolarizedIncidentE(IncidentE):
	'''IncidentE with a specific polarization'''
	def __init__(self, grid, polarization, shape):
		IncidentE.__init__(self, grid, shape)
		self.polarization = polarization

	def is_valid_at(self, axis, i, j, k):
		return axis == self.polarization and IncidentE.is_valid_at(self, axis, i, j, k)

class PlaneIncidentE(PolarizedIncidentE):
	def __init__(self, grid, polarization, k_normal, value, shape):
		PolarizedIncidentE.__init__(self, grid, polarization, shape)
		assert(len(k_normal)==3)
		assert(not isinstance(k_normal[Xx],complex) and not isinstance(k_normal[Yy],complex) and not isinstance(k_normal[Zz],complex))  # Make sure the components of normal are not complex
		assert(k_normal[polarization]==0)  # k_normal[Xx]==0 for the x-polarized plane wave
		norm_k = sqrt(k_normal[Xx]**2 + k_normal[Yy]**2 + k_normal[Zz]**2)  # norm_k is float
		k_unit_normal = [k_normal[Xx]/norm_k, k_normal[Yy]/norm_k, k_normal[Zz]/norm_k]  # k_unit_normal is float even if the components of normal are int
		k = 2*pi/self.grid.get_wvlen()
		self.k_normal = [k*k_unit_normal[Xx], k*k_unit_normal[Yy], k*k_unit_normal[Zz]]
		self.value = value
	
	def get_E_at_kernel(self, axis, i, j, k):
		ind = [i, j, k]
		ind[self.polarization] += 0.5  # half-integer indicates a dual grid position
		r = [self.grid.get_L_at(Xx,ind[Xx]), self.grid.get_L_at(Yy,ind[Yy]), self.grid.get_L_at(Zz,ind[Zz])]
		return self.value * exp(-1j * sum(imap(mul, self.k_normal, r)))
