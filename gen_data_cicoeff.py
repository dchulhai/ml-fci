#!/usr/bin/env python
from __future__ import print_function, division

def main():
    '''Main Program.'''

    import pyscf
    from pyscf import gto, scf, mcscf, ci, lo
    import numpy as np
    from sympy.utilities.iterables import multiset_permutations
    import os
    from sklearn.kernel_ridge import KernelRidge
    import scipy as sp
    import pickle

    # get arguments
    args = arguments()

    # create Mole object
    mol = pyscf.gto.Mole()
    mol.atom = get_atoms(args.coord_file)
    mol.basis = args.basis
    mol.verbose = 0
    mol.build()

    # create scf object and do HF calculation
    mf = pyscf.scf.RHF(mol)
    mf.init_guess = 'atom'
    dmfile = args.prefix + 'dm.pickle'
    dm0 = None
    if os.path.isfile(dmfile):
        dm0 = pickle.load(open(dmfile, 'rb'))
    ehf = mf.kernel(dm0=dm0)

    # get some variables
    nao = mol.nao_nr()
    ncore = args.ncore
    nocc = mol.nelectron // 2
    nvir = nao - nocc

    # read MOs from file
    mofile = args.prefix + 'mo.pickle'
    if os.path.isfile(mofile):
        mo = pickle.load(open(mofile, 'rb'))
        mf.mo_coeff = np.copy(mo)
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

    mffile = args.prefix + 'mf.pickle'
    if not os.path.isfile(mffile):
        mf.stdout = None
        mf.mol.stdout = None
        mf._chkfile.close()
        mf._chkfile = None
        pickle.dump(mf, open(mffile, 'w'))

    # some variable
    mc_vir = dps(nvir,args.vir)
    ic_vir = mc_vir // args.range[1] + 1
    icount = 0

    # CASCI for permutations of virtual space (all occupied included)
    for ivir in range(min(args.vir+1, nvir)):

        x = np.zeros((nvir), dtype=int)
        x[:ivir] = 1 

        for p in multiset_permutations(x):

            icount += 1
            if icount<args.range[0]*ic_vir or icount>=(args.range[0]+1)*ic_vir: continue

            svir = ''
            for i in range(len(p)):
                svir += str(p[i])
            fname = args.prefix + svir + '.pdata'
            if os.path.isfile(fname): continue

            mf.mo_coeff = np.copy(mo)

            r = np.zeros((nao), dtype=int)
            r[ncore:nocc] = 1 
            r[nocc:] = np.copy(p[:])

            # do CASCI claculations
            mc, ncas, nel = custom_cas(r, mf) 

            # dump civec to pickle file
            civec = mc.fcisolver.large_ci(mc.ci, ncas, nel, tol=1e-6)
            civec = civec_2_fcivec(r, civec)
            pickle.dump(civec, open(fname, 'wb'))

            print (list_2_str(r[:ncore]) + '|' + list_2_str(r[ncore:nocc]) + 
                   '|' + list_2_str(p))


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
#    nel = (nel // 2, nel // 2)

    mc = pyscf.mcscf.CASCI(mf, ncas, nel)
    mc.fcisolver = pyscf.fci.direct_spin1.FCISolver(mf.mol)
    if x.sum() < len(x):
        mo = mcscf.addons.sort_mo(mc, np.copy(mf.mo_coeff), cas_list, 0)
    else:
        mo = np.copy(mf.mo_coeff)
    energy = mc.kernel(mo)[0]

    return mc, ncas, nel


def get_atoms(fname):
    '''Returns an atom string from an XYZ file.'''
    flines = open(fname, 'r').readlines()
    st = flines[2][:-1]
    for f in flines[3:]:
        st += '; ' + f[:-1]
    return st


def civec_2_fcivec(active, civec):
    '''converts CAS space ci vectors to fci vectors.'''

    civec = list(civec)
    for i in range(len(civec)):
        civec[i] = list(civec[i])
        civec[i][1] = change_string(active, civec[i][1])
        civec[i][2] = change_string(active, civec[i][2])
        civec[i] = tuple(civec[i])
    return tuple(civec)


def change_string(x, st1):
    '''Extend the string "st1" from a CASCI calculation
    for a full CI representation, where the active orbitals
    were taken from the vector "x". A "0" in "x" suggest that
    an orbital was not in the CAS space.'''

    st1 = st1[2:][::-1]
    st2 = ''
    j = 0
    for i in range(len(x)):
        if x[i] == 0:
            st2 += '0'
        elif j < len(st1):
            st2 += st1[j]
            j += 1
        else:
            st2 += '0'
    return st2[::-1]


def dps(a,b):
    '''Counts the number of combinations.'''
    from scipy.misc import comb
    c = sum([comb(a,i) for i in range(b+1)])
    return int(c)


def list_2_str(l):
    st = "".join([str(i) for i in l]) 
    return st


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
    parser.add_argument('-v', '--vir', help='The max number of correlated '
                        'virtual orbitals.', type=int, default=4)
    parser.add_argument('-c', '--ncore', help='The number of frozen core '
                        'orbitals.', type=int, default=0)
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


if __name__=='__main__':
    main()

