from const import *
from numpy import array
from cmath import exp

class Grid:
	def __init__(self):
		self.d_prob = ['', '', '']
		self.d_prim = ['', '', '']
		self.d_dual = ['', '', '']
		self.dL_default = ['', '', '']
		self.k_Bloch = [0.0, 0.0, 0.0]
		self.d_prob_set = ['', '', '']
		self.initialized = False
	
	def confirm_initialized(self):
		if not self.isinitialized():
			raise Exception('Grid not initialized')

	def confirm_not_initialized(self):
		if self.isinitialized():
			raise Exception('Grid already initialized')

	def get_length_unit(self):
		self.confirm_initialized()
		return self.a0
		
	def set_length_unit(self, a0):
		self.confirm_not_initialized()
		self.a0 = a0
		self.length_unit_set= True
	
	def get_omega(self):
		self.confirm_initialized()
		return self.omega

	def get_wvlen(self):
		self.confirm_initialized()
		return self.wvlen

	def set_wvlen(self, wvlen):
		self.confirm_not_initialized()
		self.wvlen = wvlen
		self.omega = 2*pi/wvlen
		self.wvlen_set = True
	
	def set_dL_default(self, dL_default):
		self.confirm_not_initialized()
		assert(len(dL_default)==3)
		self.dL_default = [float(dL_default[Xx]), float(dL_default[Yy]), float(dL_default[Zz])]
	
	def get_dL_default(self, axis):
		return self.dL_default[axis]
	
	def get_d_prim(self, axis, n):
		self.confirm_initialized()
		return self.d_prim[axis][n]
	
	def get_d_dual(self, axis, n):
		self.confirm_initialized()
		return self.d_dual[axis][n]
	
	def set_d_prob(self, axis, d_prob):
		self.confirm_not_initialized()
		assert(isinstance(d_prob,tuple) or isinstance(d_prob,list))
		self.d_prob[axis] = list(d_prob)
		self.d_prob_set[axis] = True
	
	def get_N(self, axis):
		self.confirm_initialized()
		return len(self.d_prim[axis])

	def get_N_float(self, axis):
		return float(self.get_N(axis))

	def get_Nprob(self, axis):
		self.confirm_initialized()
		return len(self.d_prob[axis])

	def get_Npml(self, axis, sign):
		self.confirm_initialized()
		return self.Npml[axis][sign]
	
	def set_Npml(self, Npml):
		self.confirm_not_initialized()
		self.Npml = make_list2d(Npml, 2, 3)
		self.Npml_set = True
	
	def get_L(self, axis):
		self.confirm_initialized()
		return sum(self.d_prim[axis])

	def get_L_at(self, axis, n):
		self.confirm_initialized()
		assert(n>=0 and n<=self.get_N(axis))
		n_int = int(n)
		if n != n_int:
			p = n - n_int
			return sum(self.d_prim[axis][:n_int]) + p*self.d_prim[axis][n_int]
		else:
			return sum(self.d_prim[axis][:n_int])

	def get_exp_neg_ikL(self, axis):
		self.confirm_initialized()
		return exp(-1j * self.get_k_Bloch(axis) * self.get_L(axis))
	
	def get_k_Bloch(self, axis):
		self.confirm_initialized()
		return self.k_Bloch[axis]
		
	def set_k_Bloch(self, k_Bloch):
		self.confirm_not_initialized()
		assert(len(k_Bloch)==3)
		self.k_Bloch = [float(k_Bloch[Xx]), float(k_Bloch[Yy]), float(k_Bloch[Zz])]
	
	def get_BC(self, axis, sign):
		self.confirm_initialized()
		return self.BC[axis][sign]
	
	def set_BC(self, BC):
		self.confirm_not_initialized()
		self.BC = make_list2d(BC, 2, 3)
		self.BC_set = True
	
	def initialize(self):
		if not self.isreadyforinit():
			raise Exception, 'Grid not ready for initialization'
		self.init_d_prim()
		self.init_d_dual()
		self.initialized = self.isreadyforinit()

	def isreadyforinit(self):
		return self.length_unit_set and self.wvlen_set and self.d_prob_set[Xx] and self.d_prob_set[Yy] and self.d_prob_set[Zz] and self.BC_set and self.Npml_set

	def isinitialized(self):
		return self.initialized 

	def init_d_prim(self):
		assert(self.isreadyforinit())
		for axis in xrange(Naxis):
			if self.d_prob[axis]:  # if self.d_prob[axis] is not empty
				self.d_prim[axis] = [self.d_prob[axis][0]] * self.Npml[axis][Neg] + self.d_prob[axis] + [self.d_prob[axis][-1]] * self.Npml[axis][Pos]
			else:
				self.d_prim[axis] = [self.dL_default[axis]] * self.Npml[axis][Neg] + [self.dL_default[axis]] * self.Npml[axis][Pos]
		
	def init_d_dual(self):
		assert(self.isreadyforinit())
		for axis in xrange(Naxis):
			except_last = array(self.d_prim[axis][0:-1])
			except_first = array(self.d_prim[axis][1:])
			self.d_dual[axis] = list(0.5 * (except_last + except_first))
			if self.BC[axis][Neg] == PMC:
				self.d_dual[axis] = [self.d_prim[axis][0]] + self.d_dual[axis]  # d_prim[axis][0] == 2 * (d_dual[0]-w_prim[0])
			else:  # For PEC or Bloch BC
				self.d_dual[axis] = [0.5 * (self.d_prim[axis][0] + self.d_prim[axis][-1])] + self.d_dual[axis]  # sum(d_prim[axis]) == sum(d_dual[axis])


if __name__ == '__main__':
	grid = Grid()
	grid.set_length_unit(1e-6)
	grid.set_wvlen(1.55)
	grid.set_k_Bloch([1, 2, 3])
	grid.set_d_prob(Xx, [1,1,1,0.5,0.5,0.5,1,1,1])
	grid.set_d_prob(Yy, [1,1,1,0.5,0.5,0.5,1,1,1])
	grid.set_d_prob(Zz, [1,1,1,0.5,0.5,0.5,1,1,1])
	grid.set_Npml( (3, 0, 3) )
	grid.set_BC( (PEC, PEC, PEC) )
	grid.initialize()
	
	print 'wavelength:', grid.get_wvlen()
	print 'omega:', grid.get_omega()
	print 'k_Bloch:', grid.get_k_Bloch(Xx), grid.get_k_Bloch(Yy), grid.get_k_Bloch(Zz)
	print 'exp(-ikL):', grid.get_exp_neg_ikL(Xx), grid.get_exp_neg_ikL(Yy), grid.get_exp_neg_ikL(Zz)
	print 'd_prim:'
	for axis in xrange(Naxis):
		print axis, ':',
		for n in xrange(grid.get_N(axis)):
			print grid.get_d_prim(axis, n),
		print
	print 'd_dual:'
	for axis in xrange(Naxis):
		print axis, ':',
		for n in xrange(grid.get_N(axis)):
			print grid.get_d_dual(axis, n),
		print
	print 'Lx, Ly, Lz:', grid.get_L(Xx), grid.get_L(Yy), grid.get_L(Zz)
	print 'Lx(Nx), Ly(Nx), Lz(Nz):', grid.get_L_at(Xx, grid.get_N(Xx)), grid.get_L_at(Yy, grid.get_N(Yy)), grid.get_L_at(Zz, grid.get_N(Zz))
	print 'Nx, Ny, Nz:', grid.get_N(Xx), grid.get_N(Yy), grid.get_N(Zz)
	print 'Nprob_x, Nprob_y, Nprob_z:', grid.get_Nprob(Xx), grid.get_Nprob(Yy), grid.get_Nprob(Zz)
	print 'Npml_xn, Npml_xp, Npml_yn, Npml_yp, Npml_zn, Npml_zp:', grid.get_Npml(Xx,Neg), grid.get_Npml(Xx,Pos), grid.get_Npml(Yy,Neg), grid.get_Npml(Yy,Pos), grid.get_Npml(Zz,Neg), grid.get_Npml(Zz,Pos)
	print 'BCxn, BCxp, BCyn, BCyp, BCzn, BCzp:', grid.get_BC(Xx,Neg), grid.get_BC(Xx,Pos), grid.get_BC(Yy,Neg), grid.get_BC(Yy,Pos), grid.get_BC(Zz,Neg), grid.get_BC(Zz,Pos)
