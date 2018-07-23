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

    # glob all data files
    files = glob.glob(prefix + '_' + '*.data')
    files.sort()
    nocc = len(files[0].split('.')[0].split('_')[1])

    # return HF results if requested
    if occ == 0 and vir == 0:
        dip_hf = np.loadtxt(files[0].split('_')[0] + '_' + 
                         '0'*nocc + '.data', unpack=True, usecols=[1,2,3])
        if lprint:
            print ('HF dipole moment (Debye):')
            print (repr(dip_hf))
        if lreturn:
            return dip_hf
        else:
            return

    # initialize data
    x0 = []
    y0 = []

    for i in range(len(files)):

        # get feature vector for occupied space
        nf = np.array(list(files[i].split('.')[0].split('_')[1]), dtype=int)
        if nf.sum() > occ: continue
        x0.append(nf)

        # include HF results
        y1 = np.loadtxt(files[i], unpack=True, usecols=[1,2,3])
        if i == 0:
            y0.append(y1)
            continue

        # predict values for full virtual space
        x = np.loadtxt(files[i], unpack=True, dtype=str, usecols=[0])
        if x[0][0] == 'b':
            x1 = np.array([[int(j) for j in k[2:-1]] for j in x], dtype=int)
        else:
            x1 = np.array([[int(j) for j in k] for k in x], dtype=int)

        dip_new = np.zeros((3))
        for j in range(3):
            dip_new[j], temp = krr(x0=x1, y0=y1[j], maxactive=vir, lreturn=True,
                lprint=False)

        y0.append(dip_new)

    x0 = np.array(x0, dtype=int)
    y0 = np.array(y0).T

    # predict values for full occupied space
    dip_tot = np.zeros((3))
    for i in range(3):
        dip_tot[i], temp = krr(x0=x0, y0=y0[i], maxactive=None, lreturn=True, lprint=False)

    if lprint:
        print ('Learned dipole moments (Debye):')
        print (repr(dip_tot))

    if lreturn: return dip_tot


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

