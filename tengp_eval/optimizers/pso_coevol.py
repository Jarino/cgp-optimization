from configparser import ConfigParser

import numpy as np
from sklearn.metrics import mean_squared_error
import pygmo as pg

from tengp.individual import IndividualBuilder, NPIndividual
from tengp import Parameters, FunctionSet

from tengp_eval.coevolution import TrainersSet, GaPredictors

def fitness(individual, x, y):
    output = individual.transform(x)
    try:
        return mean_squared_error(output, y)
    except ValueError:
        return 1000000000

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

    # setup the coevolution elements
    ts = TrainersSet(ib, 4, fitness, x_train, y_train)
    predictors = GaPredictors(x_train, y_train, 100, 24)
    predictors.evaluate_fitness(ts)
    x_reduced, y_reduced = predictors.best_predictors_data()

    GENS = 100

    cf = cost_function(x_reduced, y_reduced, params, bounds)
    prob = pg.problem(cf)
    algo = pg.algorithm(pg.pso(
        gen=GENS ,
        omega=cp.getfloat('OPTIMPARAMS', 'omega'),
        eta1=cp.getfloat('OPTIMPARAMS', 'eta1'),
        eta2=cp.getfloat('OPTIMPARAMS', 'eta2'),
        memory=True))
    algo.set_verbosity(5)
    pop = pg.population(prob, 50)
    n_gens = GENS


    while n_gens < 2000: #cp.getint('DEFAULT', 'gens'):

        pop = algo.evolve(pop)

        # calculate exact fitness of champion and
        # add it to the trainers set
        champion = NPIndividual(pop.champion_x, cf.bounds, cf.params)
        try:
            champion.fitness = fitness(champion, x_train, y_train)
            ts.add_trainer(champion)
        except ValueError:
            print('unsuccessful adding of champion')

        # update random population
        ts.update_random_population()

        predictors.predictors_evolution_step(ts)
        print('changing the subset, best predictor: ', predictors.best_predictor.fitness)

        x_reduced, y_reduced = predictors.best_predictors_data()
        pop.problem.extract(object).X = x_reduced
        pop.problem.extract(object).Y = y_reduced
        n_gens += GENS

    uda = algo.extract(pg.pso)

    champion = NPIndividual(pop.champion_x, cf.bounds, cf.params)
    try:
        champion.fitness = fitness(champion, x_train, y_train)
        print('exact fitness of champion', champion.fitness)
    except ValueError:
        print('unsuccessful adding of champion')

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

    for bench_id in  range(11,12):
        x_train, y_train, x_test, y_test = get_keijzer_data(random, bench_id)

        cp = ConfigParser()
        cp.read(os.path.join(configs_dir, f'hp-keijzer{bench_id}-pso.ini'))

        max_trials = 1

        all_logs = []

        for i in range(max_trials):
            print(f'keijzer{bench_id}')
            log = run_benchmark(cp, x_test, y_test, keijzer_fixed_funset)
            all_logs.append(log)


        with open(os.path.join(output_dir, f'keijzer{bench_id}.log'), 'wb') as f:
            pickle.dump(all_logs, f)

    print(f'finised, wall time: {time() - start}')
