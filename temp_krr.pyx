from __future__ import print_function, division
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
ITYPE = np.int
ctypedef np.int_t ITYPE_t

def temp_krr(fname=None, icol=0, eone=0.0):
    '''Simplified KRR with no regularlization and a simple boolean
    kernel function.'''

    from scipy.linalg import solve
    from scipy.optimize import lsq_linear
    import progressbar

    cdef int i

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

    ehf = y0[0]
    y0 = y0[1:] - eone
    x0 = x0[1:]

    y1 = []
    x1 = []
    for i in range(len(y0)):
        if x0[i][icol] == 1 and x0[i].sum() > 1:
            if icol < x0.shape[1]-1:
                temp = np.append(x0[i][:icol], x0[i][icol+1:])
                x1.append(temp)
            else:
                x1.append(x0[i][:icol])
            y1.append(y0[i])

    x1 = np.array(x1, dtype=int)
    y1 = np.array(y1, dtype=float)

    cdef int ndata = x1.shape[0]

    # calculate kernel matrix
    K = gen_kernel(x1, False)

    w = solve(K, y1, assume_a='sym', overwrite_a=True,
        overwrite_b=True, check_finite=False)
#    w = gauss_jordan(K, y0)

    p = w.sum() + eone
    return p


def gen_kernel(x0, lprint=False):
    '''Generates the kernel matrix from the data.'''

    import numpy as np
    import progressbar

    if lprint: print ('Calculating kernel matrix...')

    if x0.dtype != bool:
        x0 = np.array(x0, dtype=bool)

    cdef int ndata = x0.shape[0]

#    K = np.memmap('kernel.matrix', mode='w+', shape=(ndata, ndata), dtype=int)
    K = np.empty((ndata, ndata), dtype='int8', order='C')

    if lprint:
        bar = progressbar.ProgressBar(maxval=ndata,
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

    for i in xrange(ndata):
        K[i,i:] = np.dot(x0[i,:], x0.T[:,i:])
        K[i:,i] = K[i,i:]
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

def check(fname, permut, icol, ncol):

    from scipy.linalg import solve
    from scipy.optimize import lsq_linear
    import progressbar

    cdef int i

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

    ehf = y0[0]
    y0 = y0[1:] - ehf 
    x0 = x0[1:]

    y1 = []
    x1 = []
    for i in range(len(y0)):
        if x0[i][icol] == 1:
            if icol < ncol-1:
                x1.append(x0[i][:icol] + x0[i][icol+1:])
            else:
                x1.append(x0[i][:icol])
            y1.append(y0[i])

    x1 = np.array(x1, dtype=int)
    y1 = np.array(y1, dtype=float)

    cdef int ndata = x1.shape[0]

    # calculate kernel matrix
    K = gen_kernel(x1, False)

    w = solve(K, y1, assume_a='sym', overwrite_a=True,
        overwrite_b=True, check_finite=False)
#    w = gauss_jordan(K, y0)

    p = w.sum()
    return p
