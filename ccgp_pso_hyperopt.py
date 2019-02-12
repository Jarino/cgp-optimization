import random

import pygmo as pg
from GPyOpt.methods import BayesianOptimization
import numpy as np
from numpy.random import seed
seed(42)

from configmock import ConfigMock
from experiment_settings import keijzer_funset
import symreg
from ccgp_pso import run_benchmark

x_train, y_train, x_test, y_test = symreg.get_benchmark_keijzer(random, 6)
x_train = np.c_[np.ones(len(x_train)), x_train]
x_test = np.c_[np.ones(len(x_test)), x_test]

cp = ConfigMock()
cp.add('n_nodes', 50)
cp.add('n_generations', 50)
cp.add('population_size', 25)

def func(x):
    print(x)
    cp.add('max_back', x[0][0])
    cp.add('omega', x[0][1])
    cp.add('eta1', x[0][2])
    cp.add('eta2', x[0][3])
    pg.set_global_rng_seed(seed = 42)
    log = run_benchmark(cp, x_train, y_train, keijzer_funset)
    return min(log)

domain = [
    {'name': 'max_back', 'type': 'discrete', 'domain': (1, 5, 10, 15, 20, 25, 30)},
    {'name': 'omega', 'type': 'continuous', 'domain': (0.01, 1)},
    {'name': 'eta1', 'type': 'continuous', 'domain': (0.01, 4)},
    {'name': 'eta2', 'type': 'continuous', 'domain': (0.01, 4)},
]

random.seed(42)
myBopt = BayesianOptimization(f=func, domain=domain)
myBopt.run_optimization(max_iter=10)

print(myBopt.X)
print(myBopt.Y)
print('best', myBopt.X[np.argmin(myBopt.Y)], np.min(myBopt.Y))

print('running on the test set')

cp.add('max_back', myBopt.X[0][0])
cp.add('omega', myBopt.X[0][1])
cp.add('eta1', myBopt.X[0][2])
cp.add('eta2', myBopt.X[0][3])

run_benchmark(cp, x_test, y_test, keijzer_funset)



