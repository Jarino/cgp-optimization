from configparser import ConfigParser

import numpy as np
from sklearn.metrics import mean_squared_error
import pygmo as pg

from tengp.individual import IndividualBuilder, NPIndividual
from tengp import Parameters, FunctionSet

class cost_function:
    def __init__(self, X, Y, params, bounds):
        self.params = params
        self.bounds = bounds
        self.X = X
        self.Y = Y

    def fitness(self, x):
        individual = NPIndividual(list(x), self.bounds, self.params)

        pred = individual.transform(self.X)

        try:
            return [mean_squared_error(pred, self.Y)]
        except ValueError:
            return [10e10]

    def get_bounds(self):
        return self.bounds

def define_cgp_system(n_nodes, n_inputs, n_outputs, funset, max_back):
    """
    define CCGP system

    Return:
        IndividualBuilder object
        Parameters
        bounds (tuple)
    """
    params = Parameters(n_inputs, n_outputs, 1, n_nodes, funset, real_valued=True, max_back=max_back)
    ib = IndividualBuilder(params)
    bounds = ib.create().bounds
    return ib, params, bounds

def run_benchmark(cp, x_train, y_train, funset):
    ib, params, bounds = define_cgp_system(
            cp.getint('CGPPARAMS', 'n_nodes'),
            x_train.shape[1] if len(x_train.shape) > 1 else 1,
            y_train.shape[1] if len(y_train.shape) > 1 else 1,
            funset,
            cp.getint('CGPPARAMS', 'max_back'))

    prob = pg.problem(cost_function(x_train, y_train, params, bounds))
    algo = pg.algorithm(pg.pso(
        gen=cp.getint('DEFAULT', 'gens'),
        omega=cp.getfloat('OPTIMPARAMS', 'omega'),
        eta1=cp.getfloat('OPTIMPARAMS', 'eta1'),
        eta2=cp.getfloat('OPTIMPARAMS', 'eta2')))
    algo.set_verbosity(100)
    pop = pg.population(prob, cp.getint('DEFAULT', 'population_size'))
    pop = algo.evolve(pop)
    uda = algo.extract(pg.pso)

    return [x[2] for x in uda.get_log()]


if __name__ == '__main__':
    import sys
    import random
    import pickle
    import os
    from time import time
    from tengp_eval.utils import get_keijzer_data
    from tengp_eval.utils.function_sets import keijzer_fixed_funset

    output_dir = sys.argv[1]
    configs_dir = sys.argv[2]

    random.seed(42)

    np.warnings.filterwarnings('ignore')


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time()

    for bench_id in  range(11,16):
        x_train, y_train, x_test, y_test = get_keijzer_data(random, bench_id)

        cp = ConfigParser()
        cp.read(os.path.join(configs_dir, f'hp-keijzer{bench_id}-pso.ini'))
        cp['DEFAULT']['gens'] = str(10)

        max_trials = 5

        all_logs = []

        for i in range(max_trials):
            print(f'keijzer{bench_id}')
            log = run_benchmark(cp, x_test, y_test, keijzer_fixed_funset)
            all_logs.append(log)


        with open(os.path.join(output_dir, f'keijzer{bench_id}.log'), 'wb') as f:
            pickle.dump(all_logs, f)

    print(f'finised, wall time: {time() - start}')
