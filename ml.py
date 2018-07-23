from __future__ import print_function, division
import numpy as np

def ml(fname=None, maxactive=None, lreturn=False, lprint=True,
        x0=None, y0=None, xp=None, lreturn_w=False):
    '''Simplified KRR with no regularlization and a simple boolean
    kernel function.'''

    from scipy.linalg import solve
    from scipy.optimize import lsq_linear
    import progressbar
    import sklearn
    from sklearn import svm

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

    if maxactive is not None:
        xx = np.array(x0, dtype='int8')
        xx = xx.sum(axis=1)
        idx = np.where(xx <= maxactive)
        x0 = x0[idx]
        y0 = y0[idx]

    dmin = y0.min() + ehf

    ndata = x0.shape[0]

    # initialize machine learning method
    SVR = sklearn.svm.SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0,
        tol=0.00001, C=1.0, epsilon=0.1, verbose=True, max_iter=-1)

    # fit/train
    SVR.fit(x0, y0)

    # predict FCI values
    if lprint: print ('Calculating energy...')
    if xp is None:
        xp = np.zeros((1,x0.shape[1]), dtype=int) + 1
    p = SVR.predict(xp)[0]  + ehf
    if lprint: print ('Energy = {0:16.12f}'.format(p))

    if lreturn:
        return p, dmin


def gen_kernel(x0, lprint=False):
    '''Generates the kernel matrix from the data.'''

    import numpy as np
    import progressbar

    if lprint: print ('Calculating kernel matrix...')

    if x0.dtype != bool:
        x0 = np.array(x0, dtype=bool)

    ndata = x0.shape[0]

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

def gen_kernel_p(x0, xp):

    if x0.dtype != bool:
        x0 = np.array(x0, dtype=bool)
    if xp.dtype != bool:
        xp = np.array(xp, dtype=bool).reshape(1,x0.shape[1])

    K = np.dot(xp, x0.T)
    K = np.array(K, dtype=int)
    return K

def gauss_jordan(A, b):
    """Puts given matrix (2D array) into the Reduced Row Echelon Form.
    Returns True if successful, False if 'm' is singular.
    NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
    Written by Jarno Elonen in April 2005, released into Public Domain"""

    import progressbar as pb

    h = A.shape[0]

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

