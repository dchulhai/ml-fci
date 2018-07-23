#! /usr/bin/env python
from __future__ import print_function, division

def main():
    '''The main program.'''

    import pyscf
    from pyscf import gto, scf, mcscf, ci, lo
    import numpy as np
    import scipy as sp
    import os
    import pickle
    from sympy.utilities.iterables import multiset_permutations

    # get arguments
    args = arguments()

    # create Mole object
    mol = pyscf.gto.Mole()
    mol.atom = get_atoms(args.coord_file)
    mol.basis = args.basis
    mol.charge = args.charge
    mol.verbose = 0
    mol.build()

    # create SCF object and do HF calculation
    mf = pyscf.scf.RHF(mol)
    mf.init_guess = 'atom'
    dmfile = args.prefix + 'dm.pickle'
    dm0 = None
    if os.path.isfile(dmfile):
        dm0 = pickle.load(open(dmfile, 'rb'))
    ehf = mf.kernel(dm0=dm0)

    # set some defauls
    nao = mol.nao_nr()
    ncore = args.ncore
    nocc = mol.nelectron // 2
    nvir = nao - nocc
    S = mf.get_ovlp()

    # read MOs from file
    mofile = args.prefix + 'mo.pickle'
    if os.path.isfile(mofile):
        mo = pickle.load(open(mofile, 'rb'))
        mf.mo_coeff = mo.copy()
    else:

        # localize occupied MOs
        mo = mf.mo_coeff.copy()
        mo[:,ncore:nocc] = lo.ER(mol).kernel(mf.mo_coeff[:,ncore:nocc], verbose=4)
        mf.mo_coeff = mo.copy()

    # save data to file
    if not os.path.isfile(dmfile):
        dm0 = mf.make_rdm1(mo_coeff=mo)
        pickle.dump(dm0, open(dmfile, 'wb'))
    if not os.path.isfile(mofile):
        pickle.dump(mo, open(mofile, 'wb'))

    # some variable
    mc_occ = dps(nocc-ncore,args.occ)
    ic_occ = mc_occ // args.range[1] + 1
    icount = 0

    # CASCI for all permutations
    for iocc in range(args.occ+1):

        # print HF results
        if iocc == 0:
            fname = args.prefix
            for i in range(nocc-ncore):
                fname += '0'
            fname += '.data'
            fout = open(fname, 'w')
            p = np.zeros((nvir), dtype=int)
            print_features(fout, p, ehf)
            fout.close()
            continue

        # permute over all occupied orbitals
        x = np.zeros((nocc-ncore), dtype=int)
        x[:iocc] = 1
        for p in multiset_permutations(x):

            icount += 1
            if icount<args.range[0]*ic_occ or icount>=(args.range[0]+1)*ic_occ: continue

            # open file for writting
            fname = args.prefix
            for i in range(nocc-ncore):
                fname += str(p[i])
            fname += '.data'
            if os.path.isfile(fname): continue
            fout = open(fname, 'w')

            # permute over virtual orbitals
            for ivir in range(min(nvir+1,args.vir+1)):

                if ivir == 0:
                    q = np.zeros((nvir), dtype=int)
                    print_features(fout, q, ehf)
                    continue

                y = np.zeros((nvir), dtype=int)
                y[:ivir] = 1

                for q in multiset_permutations(y):

                    # reset MO coefficients
                    mf.mo_coeff = mo.copy()

                    # generate new list of active orbitals and run
                    r = np.zeros((nao), dtype=int)
                    r[ncore:nocc] = p[:]
                    r[nocc:] = q[:]
                    eci = custom_cas(r, mf)

                    print_features(fout, q, eci)

                    print (list_2_str(p) + '|' + list_2_str(q))

            fout.close()


def arguments():
    '''Gets the options based on the arguments passed in.
    Use "--help" to view all available options.'''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys 

    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('coord_file', nargs=1, default=sys.stdin,
                        help='The input files to submit.')
    parser.add_argument('-b', '--basis', help='Basis set to use. '
                        'Default is "STO-3G"', default='sto3g')
    parser.add_argument('-o', '--occ', help='The max number of correlated '
                        'occupied orbitals.', type=int, default=4)
    parser.add_argument('-v', '--vir', help='The max number of correlated '
                        'virtual orbitals.', type=int, default=4)
    parser.add_argument('-c', '--ncore', help='The number of frozen core '
                        'orbitals.', type=int, default=0)
    parser.add_argument('-q', '--charge', help='The charge of the molecule.',
                        type=int, default=0)
    parser.add_argument('-p', '--prefix', help='The prefix string from which '
                        'all data will be saved/read.', default=None)
    parser.add_argument('-r', '--range', help='For large calculations, this '
                        'defines the range of the calculations. Made up of '
                        'two values, one is the index, the other is the maximum '
                        'value of the range.', type=int, nargs=2, default=[0,1])

    args = parser.parse_args()
    args.coord_file = args.coord_file[0]
    if args.prefix is None: args.prefix = args.coord_file[:-4] + '_'
    assert (args.range[1] > args.range[0])
    return args


def get_atoms(fname):
    '''Returns an atom string from an XYZ file.'''
    flines = open(fname, 'r').readlines()
    st = flines[2][:-1]
    for f in flines[3:]:
        st += '; ' + f[:-1]
    return st


def dps(a,b):
    '''Counts the number of combinations.'''
    from scipy.misc import comb
    c = sum([comb(a,i) for i in range(b+1)])
    return int(c)


def print_features(fout, x, val):
    '''Simple function to print the features to a file "fout".'''
    st = list_2_str(x)
    st += ' {0:18.12f}\n'.format(val)
    fout.write(st)


def list_2_str(l):
    st = "".join([str(i) for i in l])
    return st


def custom_cas(x, mf):
    '''Perform a CASCI calculation with
    a custom defined CAS.'''

    import pyscf
    from pyscf import mcscf, fci
    import numpy as np

    nao = mf.mo_occ.shape[0]
    assert(nao == len(x))

    # get CAS orbitals and number of CAS electrons
    cas_list = []
    nel = 0
    for i in range(len(x)):
        if x[i] == 1:
            cas_list.append(i)
            nel += mf.mo_occ[i]
    ncas = len(cas_list)
    if ncas == 0:
        return mf.e_tot
    nel = int(nel)

    # change high spin to low spin (spin-flip)
    assert (nel % 2 == 0)
    mf.mol.spin = 0
#    if mf.mol.spin != 0:
#        nel = (nel // 2, nel // 2)

    mc = pyscf.mcscf.CASCI(mf, ncas, nel)
    mc.fcisolver = pyscf.fci.direct_spin0.FCISolver(mf)
    if x.sum() < len(x):
        mo = mcscf.addons.sort_mo(mc, np.copy(mf.mo_coeff), cas_list, 0)
    else:
        mo = np.copy(mf.mo_coeff)
    energy = mc.kernel(mo)[0]

    return energy


if __name__ == '__main__':
    main()

