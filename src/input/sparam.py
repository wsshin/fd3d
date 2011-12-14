from const import *
from math import log, pow

class Sparam:
	def __init__(self, grid, grading=None):
		if not grid.isinitialized():
			raise ValueError('Uninitialized Grid given')
		self.grid = grid
		self.grading = grading
		if grading==None:
			self.grading = PolynomialGrading(grid)
		self.s_prim = [[1.0]*grid.get_N(Xx), [1.0]*grid.get_N(Yy), [1.0]*grid.get_N(Zz)]
		self.s_dual = [[1.0]*grid.get_N(Xx), [1.0]*grid.get_N(Yy), [1.0]*grid.get_N(Zz)]
		self.initialize()
	
	def get_s_prim(self, axis, n):
		return self.s_prim[axis][n]
	
	def get_s_dual(self, axis, n):
		return self.s_dual[axis][n]
	
	def initialize(self):
		for axis in xrange(Naxis):
			Npml_n = self.grid.get_Npml(axis, Neg)
			Npml_p = self.grid.get_Npml(axis, Pos)
			Ngrid = self.grid.get_N(axis)
			for ind in xrange(self.grid.get_N(axis)):
				if ind < Npml_n:
					depth_prim = Npml_n - ind
					depth_dual = depth_prim - 0.5
					self.s_prim[axis][ind] = self.grading.s_param_at(axis, Neg, depth_prim)
					self.s_dual[axis][ind] = self.grading.s_param_at(axis, Neg, depth_dual)
				elif ind >= Ngrid - Npml_p:
					depth_prim = ind - (Ngrid - Npml_p)
					depth_dual = depth_prim + 0.5
					if ind != Ngrid-Npml_p:
						self.s_prim[axis][ind] = self.grading.s_param_at(axis, Pos, depth_prim)
					self.s_dual[axis][ind] = self.grading.s_param_at(axis, Pos, depth_dual)

class Grading:
	Polynomial, Geometric = range(2)
	def __init__(self, grid):
		self.grid = grid

	def s_param_at(self, axis, sign, depth):
		raise NotImplementedError

class ConstantGrading(Grading):
	def __init__(self, grid, s0):
		Grading.__init__(self, grid)
		if (isinstance(s0, complex)):
			self.s0 = s0
		else:
			self.s0 = s0 - 0j

	def s_param_at(self, axis, sign, depth):
		return self.s0

class PolynomialGrading(Grading):
	#def __init__(self, grid, lnR=-1e9, m=11.2):
	#def __init__(self, grid, lnR=-1e5, m=7.0):  # 700nm
	#def __init__(self, grid, lnR=-1e8, m=10.9):  # 700nm
	#def __init__(self, grid, lnR=-1e10, m=12.0):  # 1550nm
	#def __init__(self, grid, lnR=-1e10, m=11.99):
	#def __init__(self, grid, lnR=-1e10, m=11.9):
	#def __init__(self, grid, lnR=-10000, m=8.0):
	#def __init__(self, grid, lnR=-1000, m=7.0):
	#def __init__(self, grid, lnR=-100, m=6.0):
	def __init__(self, grid, lnR=-16, m=4.0):  # currently used with Npml=10
	#def __init__(self, grid, lnR=-8, m=4.0):  # used with Npml=5
	#def __init__(self, grid, lnR=-17, m=4.6):  # least reflection for Npml=5
	#def __init__(self, grid, lnR=-13, m=4):  # least reflection for Npml=5 with m=4
		Grading.__init__(self, grid)
		self.lnR = float(lnR)
		self.m = float(m)
		self.ma = float(m)
		self.sigma_poly = [['',''], ['',''], ['','']]
		self.kappa_poly = [['',''], ['',''], ['','']]
		self.a_poly = [['',''], ['',''], ['','']]
		self.initialize()

	def initialize(self):
		omega = self.grid.get_omega()
		for axis in xrange(Naxis):
			for sign in xrange(Nsign):
				Npml = self.grid.get_Npml(axis,sign)
				if Npml > 0:
					#Npml = 5  # for extrapolation of PML parameters from their optimal value for Npml = 5
					if sign == Neg:
						dL = self.grid.get_d_prim(axis, 0)
					else:
						assert(sign == Pos)
						dL = self.grid.get_d_prim(axis, -1)
					#kappa_max = 20.0
					#sigma_max = - (self.m+1) * self.lnR / (2 * 3 * dL * Npml)
					kappa_max = 1.0
					sigma_max = - (self.m+1) * self.lnR / (2 * dL * Npml)
					sigma_poly = lambda x, npml=Npml, sigma0=sigma_max: sigma0 * pow(float(x)/npml, self.m)
					kappa_poly = lambda x, npml=Npml, kappa0=kappa_max: 1.0 + (kappa0-1) * pow(float(x)/npml, self.m)
					a_max = 0.0
					#if omega < sigma_poly(0.5):
					#	a_max = sigma_poly(0.5) * 10
					#a_poly = lambda x, npml=Npml, a0=a_max: a0 * pow(1.0 - float(x)/npml, self.ma)a  # when Npml is forced to 5, x>5 causes an error
					a_poly = lambda x: 0.0
					self.sigma_poly[axis][sign] = sigma_poly
					self.kappa_poly[axis][sign] = kappa_poly
					self.a_poly[axis][sign] = a_poly

	def s_param_at(self, axis, sign, depth):
		sigma = self.sigma_poly[axis][sign](depth)
		kappa = self.kappa_poly[axis][sign](depth)
		a = self.a_poly[axis][sign](depth)
		omega = self.grid.get_omega()
		assert(omega != 0.0)
	 	return kappa + sigma/(a + 1j*self.grid.get_omega())

class GeometricGrading(Grading):
	def __init__(self, grid, lnR=-2000, g=3):
		Grading.__init__(self, grid)
		self.lnR = float(lnR)
		self.g = float(g)
		self.sigma0 = [['',''], ['',''], ['','']]
		self.kappa_max = [['',''], ['',''], ['','']]
		self.initialize()
	
	def initialize(self):
		for axis in xrange(Naxis):
			for sign in xrange(Nsign):
				Npml = self.grid.get_Npml(axis,sign)
				if Npml > 0:
					if sign == Neg:
						dL = self.grid.get_d_prim(axis, 0)
					else:
						assert(sign == Pos)
						dL = self.grid.get_d_prim(axis, -1)
					self.sigma0[axis][sign] = - self.lnR * log(self.g) / (2 * dL * pow(self.g, self.grid.get_Npml(axis,sign)) - 1.0)
					self.kappa_max[axis][sign] = 1.0

	def s_param_at(self, axis, sign, depth):
		sigma = self.sigma0[axis][sign] * pow(self.g, depth)
		kappa = pow(self.kappa_max[axis][sign], (float(depth)/self.grid.get_Npml(axis,sign)))
		return kappa + 1 * sigma/1j/self.grid.get_omega()
