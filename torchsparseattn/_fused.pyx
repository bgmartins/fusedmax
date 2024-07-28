cimport numpy as np
from cython cimport floating

cpdef prox_tv1d(np.ndarray[ndim=1, dtype=floating] w, floating stepsize):
    cdef long width, k, k0, kplus, kminus
    cdef floating umin, umax, vmin, vmax, twolambda, minlambda
    width = w.size
    if width > 0 and stepsize >= 0:
        k, k0 = 0, 0			# k: current sample location, k0: beginning of current segment
        umin = stepsize  # u is the dual variable
        umax = - stepsize
        vmin = w[0] - stepsize
        vmax = w[0] + stepsize	  # bounds for the segment's value
        kplus = 0
        kminus = 0 	# last positions where umax=-lambda, umin=lambda, respectively
        twolambda = 2.0 * stepsize  # auxiliary variable
        minlambda = -stepsize		# auxiliary variable
        while True:				# simple loop, the exit test is inside
            while k >= width-1: 	# we use the right boundary condition
                if umin < 0.0:			# vmin is too high -> negative jump necessary
                    while True:
                        w[k0] = vmin
                        k0 += 1
                        if k0 > kminus: break
                    k = k0
                    kminus = k
                    vmin = w[kminus]
                    umin = stepsize
                    umax = vmin + umin - vmax
                elif umax > 0.0:    # vmax is too low -> positive jump necessary
                    while True:
                        w[k0] = vmax
                        k0 += 1
                        if k0 > kplus: break
                    k = k0
                    kplus = k
                    vmax = w[kplus]
                    umax = minlambda
                    umin = vmax + umax -vmin
                else:
                    vmin += umin / (k-k0+1)
                    while True:
                        w[k0] = vmin
                        k0 += 1
                        if k0 > k: break
                    return
            umin += w[k + 1] - vmin
            if umin < minlambda:       # negative jump necessary
                while True:
                    w[k0] = vmin
                    k0 += 1
                    if k0 > kminus: break
                k = k0
                kminus = k
                kplus = kminus
                vmin = w[kplus]
                vmax = vmin + twolambda
                umin = stepsize
                umax = minlambda
            else:
                umax += w[k + 1] - vmax
                if umax > stepsize:
                    while True:
                        w[k0] = vmax
                        k0 += 1
                        if k0 > kplus: break
                    k = k0
                    kminus = k
                    kplus = kminus
                    vmax = w[kplus]
                    vmin = vmax - twolambda
                    umin = stepsize
                    umax = minlambda
                else:                   # no jump necessary, we continue
                    k += 1
                    if umin >= stepsize:		# update of vmin
                        kminus = k
                        vmin += (umin - stepsize) / (kminus - k0 + 1)
                        umin = stepsize
                    if umax <= minlambda:	    # update of vmax
                        kplus = k
                        vmax += (umax + stepsize) / (kplus - k0 + 1)
                        umax = minlambda
