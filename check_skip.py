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
    from krr import krr
    from temp_krr import temp_krr

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

    # coulomb energy for screening
    J = mol.intor('int2e_sph').reshape(nao,nao,nao,nao)
    JJ = np.zeros((nao,nao))
    for i in range(nao):
        di = np.outer(mo[:,i], mo[:,i])
        for j in range(i+1,nao):
            dj = np.outer(mo[:,j], mo[:,j])
            JJ[i][j] = np.einsum('ijkl,ij,kl', J, di, dj)
            JJ[j][i] = JJ[i][j]

    # make temp directory and copy over files
    os.system("rm -rf __TEMP__ > /dev/null 2>&1")
    os.system("mkdir __TEMP__")
    os.system("cp *.data __TEMP__/ > /dev/null 2>&1")

    # permute over all occupied orbitals
    for iocc in range(2,args.occ+1):

        x = np.zeros((nocc-ncore), dtype=int)
        x[:iocc] = 1
        for p in multiset_permutations(x):

            icount += 1
            if icount<args.range[0]*ic_occ or icount>=(args.range[0]+1)*ic_occ: continue

            ldo = True
            if iocc > 1:

                # check if I should screen this interaction
                val = 1.0 
                for ii in range(len(p)):
                    if p[ii] == 1:
                        for ij in range(ii+1, len(p)):
                            if p[ij] == 1:
                                val *= JJ[ii+ncore][ij+ncore]
                if val < args.thresh: ldo = False

            if not ldo:
                os.system("rm __TEMP__/{0}{1}.data".format(args.prefix, list_2_str(p)))
                continue

    # permute over virtual orbitals
    for ivir in range(1,min(nvir+1,args.vir+1)):

        y = np.zeros((nvir), dtype=int)
        y[:ivir] = 1 

        for q in multiset_permutations(y):

            q = np.array(q, dtype=int)

            ldo = True
            if ivir > 1:

                # check if I should screen this interaction
                val = 1.0
                for ii in range(len(q)):
                    if q[ii] == 1:
                        for ij in range(ii+1, len(q)):
                            if q[ij] == 1:
                                val *= JJ[ii+nocc][ij+nocc]
                if val < args.thresh: ldo = False

            # do CASCI calculation
            if not ldo:
                os.system('sed -i "/{0}/d" __TEMP__/*.data'.format(list_2_str(q)))
                continue


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
                        'occupied orbitals.', type=int, default=6)
    parser.add_argument('-v', '--vir', help='The max number of correlated '
                        'virtual orbitals.', type=int, default=6)
    parser.add_argument('-c', '--ncore', help='The number of frozen core '
                        'orbitals.', type=int, default=0)
    parser.add_argument('-q', '--charge', help='The charge of the molecule.',
                        type=int, default=0)
    parser.add_argument('-t', '--thresh', help='The threshold for screening.',
                        type=float, default=0.0001)
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


def do_casci(fout, mf, mo, nao, ncore, nocc, nvir, p, q):

    import numpy as np

    ldo = True
    ic = 0 
    leci = np.array([], dtype=float)
    while (ldo or ic < 5): 

        ic += 1

        # reset MO coefficients
        mf.mo_coeff = mo.copy()

        # generate new list of active orbitals and run
        r = np.zeros((nao), dtype=int)
        r[ncore:nocc] = p[:]
        r[nocc:] = q[:]
        eci = custom_cas(r, mf) 

        # check whether two CASCI calculations are within 1e-6
        # this is a fidelity test for the CASCI calculations
        leci = np.append(leci, [eci])
        if ic > 1:
            deci = np.array([[abs(i - j) for j in leci] for i in leci])
            deci += np.diag(np.zeros((len(leci))) + 1e2)
            idx = np.where(deci < 1e-6)[0]
            if idx.shape[0] > 0:
                eci = leci[idx[0]]
                ldo = False

    if ldo:
        ferr.write(list_2_str(p) + '|' + list_2_str(q) + '\n')
    print_features(fout, q, eci)
    print (list_2_str(p) + '|' + list_2_str(q))
    return eci


if __name__ == '__main__':
    main()

