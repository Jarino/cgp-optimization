"""
CCGP, pso, pygmo
"""
from configparser import ConfigParser

import numpy as np

from pso_def import run_benchmark


if __name__ == '__main__':
    import sys
    import random
    import pickle
    import os
    from time import time

    random.seed(42)

    cp = ConfigParser()
    cp.read(sys.argv[1])
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    from experiment_settings import nguyen7_funset, pagie_funset, keijzer_funset, korns12_funset, vlad_funset
    import symreg

    benchmarks = [
        ('keijzer6', symreg.get_benchmark_keijzer(random, 6), keijzer_funset),
    ]

    start = time()

    for name, (x_train, y_train, x_test, y_test), funset in benchmarks:
        x_train = np.c_[np.ones(len(x_train)), x_train]

        max_trials = cp.getint('DEFAULT', 'trials')

        for i in range(max_trials):
            print(f'{name}-{i}')
            logs = run_benchmark(cp, x_train, y_train, funset)

        with open(os.path.join(output_dir, f'{name}.log'), 'wb') as f:
            pickle.dump(logs, f)

    print(f'finised, wall time: {time() - start}')
