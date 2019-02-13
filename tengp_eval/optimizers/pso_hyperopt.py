import os
import random
from configparser import ConfigParser

import pygmo as pg
from GPyOpt.methods import BayesianOptimization
import numpy as np
from numpy.random import seed

from tengp_eval.utils.configmock import ConfigMock
from tengp_eval.optimizers.pso import run_benchmark
from tengp_eval.utils.function_sets import keijzer_fixed_funset
from tengp_eval.utils import get_keijzer_data

def cost_function_fc(cp, funset, x, y, fitness_fn):
    def cost_function(X):
        cp['CGPPARAMS']['max_back'] = str(int(X[0][0]))
        cp['OPTIMPARAMS']['omega'] = str(X[0][1])
        cp['OPTIMPARAMS']['eta1'] = str(X[0][2])
        cp['OPTIMPARAMS']['eta2'] = str(X[0][3])
        pg.set_global_rng_seed(seed = 42)
        log = run_benchmark(cp, x, y, funset)
        return fitness_fn(log)
        #return np.sum(np.diff(log) != 0)
    return cost_function

def tune_hyperparameters(output_dir):
    for bench_id in range(1,16):
        print(f'creating hyperparameters for benchmark keijzer{bench_id}')

        x_train, y_train, x_test, y_test = get_keijzer_data(random, bench_id)

        cp = ConfigParser()
        cp['DEFAULT'] = {
            'gens': '10',
            'population_size': '10',
            'n_nodes': '50',
            'max_back': '20'
        }
        cp['CGPPARAMS'] = {
            'n_nodes': '50',
            'max_back': '20'
        }
        cp['OPTIMPARAMS'] = {}

        domain = [
            {'name': 'max_back', 'type': 'discrete', 'domain': (1, 5, 10, 15, 20, 25, 30)},
            {'name': 'omega', 'type': 'continuous', 'domain': (0.01, 1)},
            {'name': 'eta1', 'type': 'continuous', 'domain': (0.01, 4)},
            {'name': 'eta2', 'type': 'continuous', 'domain': (0.01, 4)},
        ]

        seed(42)
        random.seed(42)
        myBopt = BayesianOptimization(
                f=cost_function_fc(
                    cp, keijzer_fixed_funset, x_train, y_train,
                    lambda log: min(log)),
                domain=domain,
                acquisition_jitter=0.1)

        myBopt.run_optimization(max_iter=5)

        best = myBopt.X[np.argmin(myBopt.Y)]

        cp['CGPPARAMS']['max_back'] = str(int(best[0]))
        cp['OPTIMPARAMS']['omega'] = str(best[1])
        cp['OPTIMPARAMS']['eta1'] = str(best[2])
        cp['OPTIMPARAMS']['eta2'] = str(best[3])
        cp['DEFAULT']['gens'] = str(2000)
        cp['DEFAULT']['population_size'] = str(50)


        output_file = os.path.join(output_dir, f'hyperparams-keijzer{bench_id}.ini')

        with open(f'{output_dir}//hp-keijzer{bench_id}-pso.ini', 'w') as f:
            cp.write(f)


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tune_hyperparameters(output_dir)
