import random
from configparser import ConfigParser

import pygmo as pg
from GPyOpt.methods import BayesianOptimization
import numpy as np
from numpy.random import seed

from configmock import ConfigMock
from experiment_settings import keijzer_fixed_funset
from ccgp_pso import run_benchmark

from tengp_eval.utils import get_keijzer_data

def cost_function_fc(cp, funset, x, y):
    def cost_function(X):
        cp.add('max_back', X[0][0])
        cp.add('omega', X[0][1])
        cp.add('eta1', X[0][2])
        cp.add('eta2', X[0][3])
        pg.set_global_rng_seed(seed = 42)
        log = run_benchmark(cp, x, y, funset)
        return min(log)
        #return np.sum(np.diff(log) != 0)
    return cost_function


for bench_id in range(10,16):
    print(f'creating hyperparameters for benchmark keijzer{bench_id}')

    x_train, y_train, x_test, y_test = get_keijzer_data(random, bench_id)

    cp = ConfigMock()
    cp.add('n_generations', 10)
    cp.add('population_size', 25)
    cp.add('n_nodes', 50)

    domain = [
        {'name': 'max_back', 'type': 'discrete', 'domain': (1, 5, 10, 15, 20, 25, 30)},
        {'name': 'omega', 'type': 'continuous', 'domain': (0.01, 1)},
        {'name': 'eta1', 'type': 'continuous', 'domain': (0.01, 4)},
        {'name': 'eta2', 'type': 'continuous', 'domain': (0.01, 4)},
    ]

    seed(42)
    random.seed(42)
    myBopt = BayesianOptimization(
            f=cost_function_fc(cp, keijzer_fixed_funset, x_train, y_train),
            domain=domain,
            acquisition_jitter=0.1)

    myBopt.run_optimization(max_iter=5)

    best = myBopt.X[np.argmin(myBopt.Y)]

    cp.add('max_back', best[0])
    cp.add('omega', best[1])
    cp.add('eta1', best[2])
    cp.add('eta2', best[3])
    cp.add('n_generations', 2000)

    real_cp = ConfigParser()

    real_cp['DEFAULT'] = cp.parameters

    with open(f'hyperparams-keijzer{bench_id}-pso.ini', 'w') as f:
        real_cp.write(f)



