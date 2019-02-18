from configparser import ConfigParser
import sys
import random
import pickle
import os
from time import time
import pygmo as pg
import numpy as np
from tengp_eval.utils import get_keijzer_data
from tengp_eval.utils.function_sets import keijzer_fixed_funset


def run(algos, output_dir, configs_dir, default_ini=None):
    random.seed(42)
    pg.set_global_rng_seed(42)

    np.warnings.filterwarnings('ignore')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time()

    for bench_id in  range(1,16):

        x_train, y_train, x_test, y_test = get_keijzer_data(random, bench_id)
        print(bench_id, x_test.shape)

        chosen_run = algos[0]


#        if x_test.shape[0] > 10000:
#            print('volim si coevoluciu')
#            chosen_run = algos[1]


        cp = ConfigParser()

        if default_ini:
            cp.read(os.path.join(configs_dir, default_ini))
        else:
            cp.read(os.path.join(configs_dir, f'hp-keijzer{bench_id}.ini'))

        max_trials = 50

        all_logs = []

        for i in range(max_trials):
            print(f'keijzer{bench_id}-{i}')
            log = chosen_run(cp, x_test, y_test, keijzer_fixed_funset)
            all_logs.append(log)

        with open(os.path.join(output_dir, f'keijzer{bench_id}.log'), 'wb') as f:
            pickle.dump(all_logs, f)

    print(f'finised, wall time: {time() - start}')


if __name__ == '__main__':

    output_dir = sys.argv[2]
    configs_dir = sys.argv[3]
    default_ini = None
    if len(sys.argv) == 5:
        default_ini = sys.argv[4]

    if sys.argv[1] == 'pso':
        from .pso import RUNNERS
    elif sys.argv[1] == 'sa':
        from .sa import RUNNERS

    algos = RUNNERS

    run(algos, output_dir, configs_dir, default_ini)
