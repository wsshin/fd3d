from mio import loadmat
from numpy import interp
from const import FD3D_ROOT, heV, c0

class Material:
	def __init__(self, name, color_index, eps=1.0, mu=1.0):
		self.name = name
		self.color_index = color_index
		self.eps = eps
		self.mu = mu

	def get_eps(self, omega):
		return self.eps

	def get_mu(self, omega):
		return self.mu
	
	def get_color_index(self):
		return self.color_index

Vac = Material('Vacuum', 0)
PML = Material('PML', -1)
TruePEC = Material('PEC', -2, float('inf'))

#def create_material(name, color, wvlen0):
#	param_dir = FD3D_ROOT + '/material/'
#	param_file = name + '.mat'
#	param = loadmat(param_dir + param_file)
#	wvlen = param['wvlen'].ravel()
#	if wvlen0 < wvlen[0] or wvlen0 > wvlen[-1]:
#		raise ValueError('wvlen0 not in the range described by ' + param_file)
#	n = param['n'].ravel()
#	k = param['k'].ravel()
#	n0 = interp(wvlen0, wvlen, n)
#	k0 = interp(wvlen0, wvlen, k)
#	eps = n0 - 1j * k0
#	eps = eps * eps
#	return Material(name, color, eps)

def create_material(name, color, eV0):
	param_dir = FD3D_ROOT + '/material/'
	param_file = name + '.mat'
	param = loadmat(param_dir + param_file)
	eV = param['eV'].ravel()
	if eV0 < eV[0] or eV0 > eV[-1]:
		raise ValueError('eV0 not in the range described by ' + param_file)
	n = param['n'].ravel()
	k = param['k'].ravel()
	n0 = interp(eV0, eV, n)
	k0 = interp(eV0, eV, k)
	eps = n0 - 1j * k0
	eps = eps * eps
	return Material(name, color, eps)
