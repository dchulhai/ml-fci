#!/usr/bin/env python
from __future__ import print_function, division

def main():
    '''Main Program.'''

    import numpy as np
    import scipy as sp
    from matplotlib import pyplot as plt
    from sklearn.kernel_ridge import KernelRidge
    from sympy.utilities.iterables import multiset_permutations
    import os
    import pickle
    import glob

    # glob data files
    files = glob.glob("*.pdata")
    files.sort()
    prefix = files[0].split('_')[0] + '_'

    # get feature vectors
    x0 = []
    for i in range(len(files)):
        f = files[i].split('_')[1].split('.')[0]
        x0.append(np.array([int(j) for j in f], dtype=int))
    x0 = np.array(x0, dtype=int)
    y0 = np.zeros((len(x0)))

    # read from all pickle files
    DATA = {'X': x0}
    for i in range(len(x0)):

        fname = files[i]
        if not os.path.isfile(fname): continue
        data = pickle.load(open(fname))

        for j in range(len(data)):

            # skey is the string of the CI coefficient
            skey = data[j][1] + data[j][2]

            # add data with key to DATA
            if not skey in DATA.keys():
                DATA[skey] = y0.copy()
            DATA[skey][i] = data[j][0]

    pickle.dump(DATA, open(prefix + 'data_all.pickle', 'w'))


if __name__ == '__main__':
    main()
