#!/usr/bin/env python
from __future__ import print_function, division

def main(prefix, vir):
    '''Main Program.'''

    import numpy as np
    import scipy as sp
    import pickle
    from krr import krr
    import progressbar

    # load pickle data
    data = pickle.load(open(prefix+'_data_all.pickle'))
    keys = data.keys()
    CI = [[], []]

    # initialize progress bar
    bar = progressbar.ProgressBar(maxval=len(keys),
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    # cycle over all CI coefficients
    for i in xrange(len(keys)):

        if keys[i] == 'X': continue

        # predict the CI coefficient for the full virtual space
        e, temp = krr(maxactive=vir, lreturn=True, lprint=False,
                      x0=data['X'], y0=data[keys[i]])

        CI[0].append(keys[i])
        CI[1].append(e)
        bar.update(i+1)

    # save CI vectors to file
    CI[0] = np.array(CI[0])
    CI[1] = np.array(CI[1])
    pickle.dump(CI, open(prefix + '_CI_vir{0}.pickle'.format(vir), 'w'))

    bar.finish()


def arguments():
    '''Gets the options based on the arguments passed in.'''

    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent
    import sys 

    parser = ArgumentParser(description=dedent(main.__doc__),
                            formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument('prefix', nargs=1, default=sys.stdin,
                        help='The input files to submit.')
    parser.add_argument('-v', '--vir', help='The max number of active '
                        'virtual orbitals.', type=int, default=3)

    args = parser.parse_args()
    args.prefix = args.prefix[0]
    return args


if __name__ == '__main__':
    args = arguments()
    main(args.prefix, args.vir)
