import struct
import os
from numpy import array
from grid import Grid
from sparam import Sparam
from object import Object
from shape import Box
from material import *
from const import *

class Simulation:
	def __init__(self, grid, grading=None):
		if not grid.isinitialized():
			raise ValueError('Uninitialized Grid given')
		self.grid = grid
		self.sparam = Sparam(grid, grading)
		self.objList = []
		self.bg_objList = []  # background object list for TF/SF
		self.srcList = []
		self.tol = 1e-6
		self.max_iter = -1

		box = Box(grid)
		box_vac = Object(box, Vac)
		self.bg_objList.append(box_vac)
		
		# PMLs
		'''The following object for the PML is only for visualization.  PML is functional without
		this object once the PML is set in the input file by grid.set_Npml().'''
		Nx = grid.get_N(Xx)
		Ny = grid.get_N(Yy)
		Nz = grid.get_N(Zz)
		Npml_xn = grid.get_Npml(Xx,Neg)
		Npml_xp = grid.get_Npml(Xx,Pos)
		Npml_yn = grid.get_Npml(Yy,Neg)
		Npml_yp = grid.get_Npml(Yy,Pos)
		Npml_zn = grid.get_Npml(Zz,Neg)
		Npml_zp = grid.get_Npml(Zz,Pos)

		in_box = Box(grid, grid.get_Nprob(Xx), grid.get_Nprob(Yy), grid.get_Nprob(Zz))
		in_box = in_box.translate(Npml_xn, Npml_yn, Npml_zn)
		pml_shape = -in_box
		if Npml_xn > 0:
			pml_xn_box = Box(grid, Npml_xn, Ny, Nz)
			pml_shape = pml_shape + pml_xn_box
		if Npml_xp > 0:
			pml_xp_box = Box(grid, Npml_xp, Ny, Nz).translate(Nx-Npml_xp, 0, 0)
			pml_shape = pml_shape + pml_xp_box
		if Npml_yn > 0:
			pml_yn_box = Box(grid, Nx, Npml_yn, Nz)
			pml_shape = pml_shape + pml_yn_box
		if Npml_yp > 0:
			pml_yp_box = Box(grid, Nx, Npml_yp, Nz).translate(0, Ny-Npml_yp, 0)
			pml_shape = pml_shape + pml_yp_box
		if Npml_zn > 0:
			pml_zn_box = Box(grid, Nx, Ny, Npml_zn)
			pml_shape = pml_shape + pml_zn_box
		if Npml_zp > 0:
			pml_zp_box = Box(grid, Nx, Ny, Npml_zp).translate(0, 0, Nz-Npml_zp)
			pml_shape = pml_shape + pml_zp_box
		pml_object = Object(pml_shape, PML)
		self.pml_object = pml_object

		self.eps_file = None
		self.x0_file = None
		self.xref_file = None

		# TF box for TF/SF
		self.incidentE = None
	
	def get_material(self, name, color):
		if name=='Vacuum':
			return Vac
		else:
			return create_material(name, color, self.grid.get_eV())
	
	def append_bg_object(self, obj):
		self.bg_objList.append(obj)
	
	def append_object(self, obj):
		self.objList.append(obj)
	
	def append_source(self, src):
		self.srcList.append(src)

	def set_eps_file(self, eps_file):
		self.eps_file = eps_file
	
	def get_sol_guess(self):
		if self.x0_file != None and not os.path.exists(self.x0_file):
			raise UninitializedError('Guess solution file does not exist')
		return self.x0_file
	
	def set_sol_guess(self, x0_file):
		'''Set the file name of a guess of the solution, which is used as the starting 
		point of the iterative solver.'''
		self.x0_file = x0_file
	
	def get_sol_reference(self):
		if self.xref_file != None and not os.path.exists('../in/'+self.xref_file):
			raise UninitializedError('Reference solution file does not exist')
		return self.xref_file
	
	def set_sol_reference(self, xref_file):
		'''Set the file name of a guess of the solution, which is used as the starting 
		point of the iterative solver.'''
		self.xref_file = xref_file
	
	def get_BiCG_tol(self):
		'''Should change this to get_err_tol, because I also use iterative methods other than 
		BiCG.''' 
		return self.tol

	def set_BiCG_tol(self, tol):
		'''Set the tolerance of BiCG.'''
		self.tol = tol
	
	def get_BiCG_max_iter(self):
		return self.max_iter

	def set_BiCG_max_iter(self, max_iter):
		'''Set the maximum number of iterations of BiCG.'''
		self.max_iter = max_iter
	
	def get_eps_at(self, axis, i, j, k, bg_only=False):
		'''bg_only is for TF/SF.'''
		assert(isinstance(i,int) and isinstance(j,int) and isinstance(k,int))
		pos1 = [i, j, k]
		eps1 = self.get_eps_at_kernel(pos1, bg_only)
		pos2 = [i, j, k]
		pos2[axis] += 1
		eps2 = self.get_eps_at_kernel(pos2, bg_only)
		denom = (1.0/eps1 + 1.0/eps2)
		if denom==0.0:
			return float('inf')
		else:
			return 2.0 / (1.0/eps1 + 1.0/eps2)  # for the continuity of the normal component of E at the interface.  See Hwang and Cangellaris, IEEE Microwave and Wireless Components Letters, 11 (4), 2001

	def get_eps_node_at(self, i, j, k, bg_only=False):
		'''Return the value of eps at a node regardless of axis.'''
		'''bg_only is for TF/SF.'''
		assert(isinstance(i,int) and isinstance(j,int) and isinstance(k,int))
		pos = [i, j, k]
		return self.get_eps_at_kernel(pos, bg_only)

	def get_eps_at_kernel(self, pos, bg_only=False):
		'''bg_only is for TF/SF.'''
		assert(len(pos)==3)
		'''If eps_file is set, then use it; TF/SF is not supported.'''
		if self.eps_file != None:
			raise NotImplemented

		'''First, iterate through the foreground object list, because the foreground objects always
		have priority to the background objects.'''
		if not bg_only:
			for n in xrange(len(self.objList)-1, -1, -1):
				obj = self.objList[n]
				if obj.contains(pos[Xx], pos[Yy], pos[Zz]):
					return obj.get_eps(self.grid.wvlen)
		'''Next, iterate through the background object list.'''
		for n in xrange(len(self.bg_objList)-1, -1, -1):
			obj = self.bg_objList[n]
			if obj.contains(pos[Xx], pos[Yy], pos[Zz]):
				return obj.get_eps(self.grid.wvlen)
		assert('Should not reach here.')
			 
	def get_mu_at(self, axis, i, j, k, bg_only=False):
		'''bg_only is for TF/SF.'''
		assert(isinstance(i,int) and isinstance(j,int) and isinstance(k,int))
		pos1 = [i, j, k]
		mu1 = self.get_mu_at_kernel(pos1, bg_only)
		pos2 = [i, j, k]
		pos2[(axis+1)%Naxis] += 1
		mu2 = self.get_mu_at_kernel(pos2, bg_only)
		pos3 = [i, j, k]
		pos3[(axis+2)%Naxis] += 1
		mu3 = self.get_mu_at_kernel(pos3, bg_only)
		pos4 = [i, j, k]
		pos4[(axis+1)%Naxis] += 1
		pos4[(axis+2)%Naxis] += 1
		mu4 = self.get_mu_at_kernel(pos4, bg_only)
		return (mu1+mu2+mu3+mu4)/4.0  # for the continuity of the tangential component of H at the interface.  See Hwang and Cangellaris, IEEE Microwave and Wireless Components Letters, 11 (4), 2001

	def get_mu_at_kernel(self, pos, bg_only=False):
		'''bg_only is for TF/SF.'''
		assert(len(pos)==3)
		'''First, iterate through the foreground object list, because the foreground objects always
		have priority to the background objects.'''
		if not bg_only:
			for n in xrange(len(self.objList)-1, -1, -1):
				obj = self.objList[n]
				if obj.contains(pos[Xx], pos[Yy], pos[Zz]):
					return obj.get_mu(self.grid.wvlen)
		'''Next, iterate through the background object list.'''
		for n in xrange(len(self.bg_objList)-1, -1, -1):
			obj = self.bg_objList[n]
			if obj.contains(pos[Xx], pos[Yy], pos[Zz]):
				return obj.get_mu(self.grid.wvlen)
		assert('Should not reach here.')
	
	def get_src_at(self, polarization, i, j, k):
		for n in xrange(len(self.srcList)-1, -1, -1):
			src = self.srcList[n]
			if src.is_valid_at(polarization, i, j, k):
				return src.get_src_at(polarization, i, j, k)
		return 0.0
	
	def has_incidentE(self):
		return self.incidentE != None

	def get_incidentE_at(self, axis, i, j, k):
		assert(self.has_incidentE())
	 	return self.incidentE.get_E_at(axis, i, j, k)
	
	def set_incidentE(self, incidentE):
		self.incidentE = incidentE
	
	def get_wvlen(self):
		return self.grid.get_wvlen()

	def get_omega(self):
		return self.grid.get_omega()

	def get_k_Bloch(self, axis):
		return self.grid.get_k_Bloch(axis)

	def get_exp_neg_ikL(self, axis):
		return self.grid.get_exp_neg_ikL(axis)
	
	def get_d_prim(self, axis, n):
		return self.grid.get_d_prim(axis, n)

	def get_d_dual(self, axis, n):
		return self.grid.get_d_dual(axis, n)

	def get_BC(self, axis, sign):
		return self.grid.get_BC(axis, sign)

	def get_N(self, axis):
		return self.grid.get_N(axis)

	def get_Npml(self, axis, sign):
		return self.grid.get_Npml(axis, sign)

	def get_s_prim(self, axis, n):
		return self.sparam.get_s_prim(axis, n)

	def get_s_dual(self, axis, n):
		return self.sparam.get_s_dual(axis, n)
	
	def runtest(self, material_probe=None, src_probe=None):
		print 'Guess solution file:', self.get_sol_guess()
		print 'BiCG tolerance:', self.get_BiCG_tol()
		print 'BiCG maximum # of iterations:', self.get_BiCG_max_iter()
		print 'normalized wavelength:', self.get_wvlen()
		print 'normalized omega:', self.get_omega()
		print 'k_Bloch:', [self.get_k_Bloch(Xx), self.get_k_Bloch(Yy), self.get_k_Bloch(Zz)]
		print 'exp(-ikL):', [self.get_exp_neg_ikL(Xx), self.get_exp_neg_ikL(Yy), self.get_exp_neg_ikL(Zz)]
		print 'BC: [[' + BCName[self.get_BC(Xx,Neg)] + ', ' + BCName[self.get_BC(Xx,Pos)] + ']; [' + BCName[self.get_BC(Yy,Neg)] + ', ' + BCName[self.get_BC(Yy,Pos)] + ']; [' + BCName[self.get_BC(Zz,Neg)] + ', ' + BCName[self.get_BC(Zz,Pos)] + ']]'
		print 'N:', [self.get_N(Xx), self.get_N(Yy), self.get_N(Zz)]
		print 'Nprob:', [self.grid.get_Nprob(Xx), self.grid.get_Nprob(Yy), self.grid.get_Nprob(Zz)]
		print 'Npml: [[' + str(self.get_Npml(Xx,Neg)) + ', ' + str(self.get_Npml(Xx,Pos)) + ']; [' + str(self.get_Npml(Yy,Neg)) + ', ' + str(self.get_Npml(Yy,Pos)) + ']; [' + str(self.get_Npml(Zz,Neg)) + ', ' + str(self.get_Npml(Zz,Pos)) + ']]'
		print 'dL_default:', [self.grid.get_dL_default(Xx), self.grid.get_dL_default(Yy), self.grid.get_dL_default(Zz)]
		print
		print 'd_prim:' 
		for axis in xrange(Naxis):
			print AxisName[axis] + ': \n[', 
			for n in xrange(self.get_N(axis)):
				print self.grid.get_d_prim(axis, n),
			print ']'
		print
		print 'd_dual:'
		for axis in xrange(Naxis):
			print AxisName[axis], ': \n[',
			for n in xrange(self.get_N(axis)):
				print self.grid.get_d_dual(axis, n),
			print ']'
		print
		print 's_prim:' 
		for axis in xrange(Naxis):
			print AxisName[axis], ': \n[', 
			for n in xrange(self.get_N(axis)):
				print self.get_s_prim(axis, n),
			print ']'
		print
		print 's_dual:'
		for axis in xrange(Naxis):
			print AxisName[axis], ': \n[',
			for n in xrange(self.get_N(axis)):
				print self.get_s_dual(axis, n),
			print ']'
		print
		if material_probe==None:
			material_probe = [self.get_N(Xx)/2, self.get_N(Yy)/2, self.get_N(Zz)/2]
		print 'eps at', str(tuple(material_probe))+':', self.get_eps_at_kernel(material_probe)
		print 'mu at', str(tuple(material_probe))+':', self.get_mu_at_kernel(material_probe)
		if src_probe==None:
			src_probe = [Zz, self.get_N(Xx)/2, self.get_N(Yy)/2, self.get_N(Zz)/2]
		print 'source at', str(tuple(src_probe[1:])) + ' in ' + AxisName[src_probe[0]] +':', self.get_src_at(src_probe[0], src_probe[1], src_probe[2], src_probe[3])
		if self.has_incidentE():
			print 'incident E at', str(tuple(src_probe[1:])) + ' in ' + AxisName[src_probe[0]] +':', self.get_incidentE_at(src_probe[0], src_probe[1], src_probe[2], src_probe[3])
		

if __name__ == '__main__':
	from source import PointSrc
	grid = Grid()
	grid.set_length_unit(1e-6)
	grid.set_wvlen(1.55)
	grid.set_d_prob(Xx, [1.0/20]*80)
	grid.set_d_prob(Yy, [1.0/20]*80)
	grid.set_d_prob(Zz, [1.0/20]*80)
	grid.set_BC(PEC)
	grid.set_Npml(10)
	grid.set_k_Bloch([0.5, 0.1, 0.2])
	grid.initialize()

	sim = Simulation(grid)
	Si = Material('Silicon', 1, 11.8)
	box = Box(grid, 20, 20, 20).center_at_middle()
	box_Si = Object(box, Si)
	sim.append_object(box_Si)

	x_src = 3
	src = PointSrc(grid, Zz, 1)
	src = src.translate(grid.get_Npml(Xx,Neg)+x_src, grid.get_N(Yy)/2, grid.get_N(Zz)/2)
	sim.append_source(src)

	sim.runtest()
