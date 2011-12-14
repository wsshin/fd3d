import Gnuplot

def view(A, h_label, v_label, c_range=None, h_prim=None, v_prim=None, on_real_axes=False):
	nrow, ncol = A.shape
	hmin, hmax = 0, nrow-1
	vmin, vmax = 0, ncol-1
	portion_margin = 0.05
	if h_prim != None:
		hmin, hmax = h_prim[0], h_prim[-1]
	if v_prim != None:
		vmin, vmax = v_prim[0], v_prim[-1]
	margin = portion_margin * max(hmax-hmin, vmax-vmin)
	g = Gnuplot.Gnuplot(persist=True)
	g('set size ratio -1')
	if c_range != None:
		g('set cbrange [' + str(c_range[0]) + ':' + str(c_range[1]) + ']')
	g('set xlabel \"' + h_label + '\"')
	g('set ylabel \"' + v_label + '\"')
	g('set pm3d map')
	g('set xrange [' + str(hmin-margin) + ':' + str(hmax+margin) + ']')
	g('set yrange [' + str(vmin-margin) + ':' + str(vmax+margin) + ']')
	#g('set pm3d corners2color min')
	#g('set palette color')
	#g('set palette defined (0 "blue", 1 "white", 2 "red")')
	#g('set palette rgbformulae 7,5,15')
	data = Gnuplot.GridData(A, h_prim, v_prim, binary=0)  # if h_prim and v_prim are None, indices of the elements of A are used
	g.splot(data)
