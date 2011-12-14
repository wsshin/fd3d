from math import pi, sqrt
import commands

FD3D_ROOT = commands.getoutput('source ~/.bashrc;echo $FD3D_ROOT')  #FD3D_ROOT = os.environ['FD3D_ROOT'] is not supported in MATLAB 2010
c0 = 2.99792458e8
mu0 = pi*4.0e-7
eps0 = 1.0 / (mu0*c0*c0)
eta0 = sqrt(mu0/eps0)

Xx, Yy, Zz , Naxis = range(4)
AxisName = ['X', 'Y', 'Z']
Neg, Pos, Nsign = range(3)
PEC, PMC, Bloch, hPEC = range(4)
BCName = ['PEC', 'PMC', 'Bloch', 'hPEC']

def isnum(elem):
	return isinstance(elem, int) or isinstance(elem, float) or isinstance(elem, complex)

def isindexed(elem):
	return isinstance(elem, list) or isinstance(elem, tuple)

def make_list2d(elem, num_inner, num_outer):
	if isnum(elem):
		return [[elem]*num_inner]*num_outer
	else:
		if not (isindexed(elem) and len(elem)==num_outer):
			raise ValueError('Not an indexed, or len(elem) != num_outer')
		elem_tuple = []
		for i in xrange(num_outer):
			if isnum(elem[i]):
				elem_tuple.append([elem[i]]*num_inner)
			else:
				assert(isindexed(elem[i]))
				assert(len(elem[i])==num_inner)
				for j in xrange(num_inner):
					assert(isnum(elem[i][j]))
				elem_tuple.append(list(elem[i]))
		return elem_tuple

def cyclic_axis(normal):
	Rr = normal
	Pp = (normal+1) % Naxis
	Qq = (normal+2) % Naxis
	return [Pp, Qq, Rr]

def update_minmax(minmax_prev, value_curr):
	assert(len(minmax_prev)==2)
	min_curr, max_curr = minmax_prev
	if value_curr < min_curr:
		min_curr = value_curr
	if value_curr > max_curr:
		max_curr = value_curr
	return [min_curr, max_curr] 

class UninitializedError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)
