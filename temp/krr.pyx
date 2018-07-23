from __future__ import print_function, division
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ITYPE = np.int
ctypedef np.int_t ITYPE_t

def krr(fname=None, maxactive=None, lreturn=False, lprint=True,
        x0=None, y0=None, xp=None, lreturn_w=False, coef=None):
    '''Simplified KRR with no regularlization and a simple boolean
    kernel function.'''

    from scipy.linalg import solve
    from scipy.optimize import lsq_linear
    import progressbar

    cdef int i

    # read from file
    if fname is not None:
        if lprint: print ('Reading input data...')

        # check for number of columns
        with open(fname, 'r') as f:
            first_line = f.readline().strip()
            num_col = len(first_line.split())

        # read the data
        if num_col > 2:
            x0 = np.loadtxt(fname, dtype=bool, unpack=True, usecols=range(num_col-1)).transpose()
        else:
            xs = np.loadtxt(fname, unpack=True, usecols=0, dtype=str)
            x0 = np.array([[int(xs[i][j]) for j in range(len(xs[i]))]
                           for i in range(len(xs))], dtype=bool)
            del(xs)
        y0 = np.loadtxt(fname, unpack=True, usecols=num_col-1)
    elif x0 is None or y0 is None:
        print ('ERROR: Must give one of "fname" or both "x0" and "y0"')
        return

    if x0.ndim == 1:
        if lprint: print ('Energy = {0:16.12f}'.format(y0))
        if lreturn:
            return y0, y0
        else:
            return

    ehf = y0[0]
    y0 = y0[1:] - ehf
    x0 = x0[1:]

    maxactive = 1

    if maxactive is not None:
        xx = np.array(x0, dtype='int8')
        xx = xx.sum(axis=1)
        idx = np.where(xx <= maxactive)
        xt = x0[idx]
        yt = y0[idx]
        idx = np.where(xx == maxactive+1)
        xv = x0[idx]
        yv = y0[idx]
    else:
        xt = np.copy(x0)
        yt = np.copy(y0)
        xv = np.copy(x0)
        yv = np.copy(x0)

    cdef float dmin = yt.min() + ehf

    cdef int ndata = xt.shape[0]

    # calculate kernel matrix
    K = gen_kernel(xt, coef=coef, lprint=lprint)

    if lprint: print ('Solving linear equations...')

    w = solve(K, yt, overwrite_a=True,
        overwrite_b=True, check_finite=False)
#    w = gauss_jordan(K, yt)

    print ('KERNELS ', xt.shape, xv.shape)
    kv = gen_kernel(xt, x1=xv, coef=coef, lprint=lprint)
    print (kv)
    yp = np.dot(kv.T, w)
    error = (np.abs(yv - yp)).max()

    if lreturn_w: return xt, w

    if lprint: print ('Calculating energy...')
    if xp is None:
        p = w.sum() + ehf
    else:
        Kp = gen_kernel(xp, xt)
        p = np.dot(Kp, w) + ehf

    if lprint: print ('Energy = {0:16.12f}'.format(p))

    if lreturn:
        return p, dmin


def gen_kernel(x0, x1=None, coef=None, lprint=False):
    '''Generates the kernel matrix from the data.'''

    import numpy as np
    import progressbar

    if lprint: print ('Calculating kernel matrix...')

    if coef is None: coef = 1e4

    x0 = np.array(x0, dtype=int)
    if x1 is None:
        x1 = np.copy(x0)
    else:
        x1 = np.array(x1, dtype=int)

    print (x0.shape, x1.shape)

    cdef int n0 = x0.shape[0]
    cdef int n1 = x1.shape[0]

    K = np.zeros((n0, n1), dtype=float)

    if lprint:
        bar = progressbar.ProgressBar(maxval=n0,
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    for i in xrange(n0):
        K[i] = np.dot(x0[i,:], x1.T[:,:])
        K[i] = np.tanh( coef * K[i] )
        if lprint: bar.update(i+1)

    if lprint: bar.finish()

    return K


def gauss_jordan(np.ndarray A, np.ndarray b):
    """Puts given matrix (2D array) into the Reduced Row Echelon Form.
    Returns True if successful, False if 'm' is singular.
    NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
    Written by Jarno Elonen in April 2005, released into Public Domain"""

    import progressbar as pb

    cdef int h = A.shape[0]
    cdef int y, y2
    cdef int c, temp

    bar_a = pb.ProgressBar(maxval=h,
        widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    bar_a.start()

    for y in xrange(0, h): 
        # Eliminate column y
        for y2 in xrange(y+1, h):
            c = A[y2,y] / A[y,y]
            if c == 1:
                A[y2,y:] -= A[y,y:]
                b[y2] -= b[y]
            elif c == -1:
                A[y2,y:] += A[y,y:]
                b[y2] += b[y]

        bar_a.update(y+1)
    bar_a.finish()

    # Backsubstitute
    bar_b = pb.ProgressBar(maxval=h,
        widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    bar_b.start()

    for y in xrange(h-1, 0-1, -1):
        c  = A[y,y]
        for y2 in xrange(0, y):
            temp = A[y2,y] * c
            if temp == 1:
                A[y2,y:] -= A[y,y:]
                b[y2] -= b[y]
            elif temp == -1:
                A[y2,y:] += A[y,y:]
                b[y2] += b[y]
        A[y,y] /= c

        # Normalize row y
        b[y] /= c

        bar_b.update(h-y)

    bar_b.finish()

    return b

