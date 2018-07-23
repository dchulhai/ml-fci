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
#    from krr import krr
    from ml import ml

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
    x0 = []
    y0 = []
    dmin = 1e6

    for i in range(len(files)):

        # get feature vector for occupied space
        nf = np.array(list(files[i].split('.')[0].split('_')[1]), dtype=int)
        if nf.sum() > occ: continue
        x0.append(nf)

        # include HF results
        if i == 0:
            y = np.loadtxt(files[i], unpack=True, usecols=[-1])
            y0.append(y)
            continue

        # predict values for full virtual space
#        p, pmin = krr(files[i], vir, lreturn=True, lprint=False)
        p, pmin = ml(files[i], vir, lreturn=True, lprint=False)
        y0.append(p)
        if pmin < dmin: dmin = pmin

    x0 = np.array(x0, dtype=int)
    y0 = np.array(y0)

    # predict values for full occupied space
#    p, pmin = krr(x0=x0, y0=y0, maxactive=None, lreturn=True, lprint=lprint)
    p, pmin = ml(x0=x0, y0=y0, maxactive=None, lreturn=True, lprint=lprint)

    if lreturn: return p, dmin


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

    args = parser.parse_args()
    args.prefix = args.prefix[0]
    return args


if __name__ == '__main__':
    args = arguments()
    main(args.prefix, args.occ, args.vir)

