#! /usr/bin/env python
from __future__ import print_function, division

def main(prefix, occ, vir, lreturn=False, lprint=True):
    '''The main program.'''

    import pyscf
    from pyscf import gto, scf, mcscf, ci, lo
    import numpy as np
    import scipy as sp
    import os
    import pickle
    from sympy.utilities.iterables import multiset_permutations
    import glob
    from krr import krr
    from scipy.linalg import solve

    # glob all data files
    files = glob.glob(prefix + '_' + '*.data')
    files.sort()
    nocc = len(files[0].split('.')[0].split('_')[1])

    # return HF results if requested
    if occ == 0 and vir == 0:
        ehf = np.loadtxt(files[0].split('_')[0] + '_' + 
                         '0'*nocc + '.data', unpack=True, usecols=[-1])
        return ehf, ehf

    # initialize data
    xt = []
    yt = []

    xv = []
    yv = []

    dmin = 1e6

    for i in range(len(files)):

        # get feature vector for occupied space
        nf = np.array(list(files[i].split('.')[0].split('_')[1]), dtype=int)

        # include HF results
        if i == 0:
            ehf = np.loadtxt(files[i], unpack=True, usecols=[-1])
            continue

        # check for number of columns
        with open(files[i], 'r') as f:
            first_line = f.readline().strip()
            num_col = len(first_line.split())

        # read the data
        if num_col > 2:
            x0 = np.loadtxt(files[i], dtype=bool, unpack=True, usecols=range(num_col-1)).transpose()
        else:
            xs = np.loadtxt(files[i], unpack=True, usecols=0, dtype=str)
            x0 = np.array([[int(xs[i][j]) for j in range(len(xs[i]))]
                           for i in range(len(xs))], dtype=bool)
            del(xs)
        y0 = np.loadtxt(files[i], unpack=True, usecols=num_col-1)

        if nf.sum() <= occ:

            xxt = []
            yyt = []
            xxv = []
            yyv = []

            for j in range(len(x0)):

#                temp = np.append(nf, x0[j])
                temp = x0[j].copy()

                if x0[j].sum() == 0:
                    pass

                elif x0[j].sum() <= vir:

                    xxt.append(temp)
                    yyt.append(y0[j] - ehf)

                elif x0[j].sum() == vir+1:

                    xxv.append(temp)
                    yyv.append(y0[j] - ehf)

            xxt = np.array(xxt, dtype=int)
            yyt = np.array(yyt)

            xxv = np.array(xxv, dtype=int)
            yyv = np.array(yyv)

            # train algorithm
            K = kernel(xxt, xxt, args.coef)
            w = solve(K, yyt, overwrite_a=True, overwrite_b=True, check_finite=False)

            # validate algorithm
            K = kernel(xxv, xxt, args.coef)
            yp = np.dot(K, w)

            print ('VALIDATION ERROR: {0:12.8f}'.format((np.abs(yyv - yp)).sum()/len(yp)))

        elif nf.sum() == vir+1:

            for j in range(len(x0)):

                temp = np.append(nf, x0[j])

                if x0[j].sum() == occ+1:

                    xv.append(temp)
                    yv.append(y0[j] - ehf)

    xt = np.array(xt, dtype=int)
    yt = np.array(yt)

    xv = np.array(xv, dtype=int)
    yv = np.array(yv)

    # train algorithm
    K = kernel(xt, xt, args.coef)
    w = solve(K, yt, overwrite_a=True, overwrite_b=True, check_finite=False)

    # validate algorithm
    K = kernel(xv, xt, args.coef)
    yp = np.dot(K, w)

    print ('VALIDATION ERROR: {0:12.8f}'.format((np.abs(yv - yp)).sum()/len(yp)))

    # prediction for the full system
    xp = np.zeros((1,xt.shape[1]), dtype=int) + 1
    K = kernel(xp, xt, args.coef)
    yp = np.dot(K, w)
    print ('FULL PREDICTION: {0:12.8f}'.format(yp[0]+ehf))


def kernel(x1, x2, coef):
    import numpy as np
    K = np.zeros((x1.shape[0], x2.shape[0]))
    for i in xrange(x1.shape[0]):
        K[i] = np.dot(x1[i], x2.T)
        K[i] = np.tanh(coef * K[i])
    return K


def arguments():
    '''Gets the options based on the arguments passed in.'''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys 

    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('prefix', nargs=1, default=sys.stdin,
                        help='The input files to submit.')
    parser.add_argument('-o', '--occ', help='The max number of active '
                        'occupied orbitals', type=int, default=3)
    parser.add_argument('-v', '--vir', help='The max number of active '
                        'virtual orbitals.', type=int, default=3)
    parser.add_argument('-c', '--coef', help='The coefficient for the TANH '
                        'function.', type=float, default=1e4)

    args = parser.parse_args()
    args.prefix = args.prefix[0]
    return args


if __name__ == '__main__':
    args = arguments()
    main(args.prefix, args.occ, args.vir)

