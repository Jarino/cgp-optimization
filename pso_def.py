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
            cp.getint('DEFAULT', 'n_nodes'),
            x_train.shape[1] if len(x_train.shape) > 1 else 1,
            y_train.shape[1] if len(y_train.shape) > 1 else 1,
            funset,
            cp.getint('DEFAULT', 'max_back'))

    prob = pg.problem(cost_function(x_train, y_train, params, bounds))
    algo = pg.algorithm(pg.pso(
        gen=cp.getint('DEFAULT', 'n_generations'),
        omega=cp.getfloat('DEFAULT', 'omega'),
        eta1=cp.getfloat('DEFAULT', 'eta1'),
        eta2=cp.getfloat('DEFAULT', 'eta2')
        ))
    algo.set_verbosity(1)
    pop = pg.population(prob, cp.getint('DEFAULT', 'population_size'))
    pop = algo.evolve(pop)
    uda = algo.extract(pg.pso)

    return [x[2] for x in uda.get_log()]
