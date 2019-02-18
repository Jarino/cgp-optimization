from configparser import ConfigParser

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pygmo as pg

from tengp.individual import IndividualBuilder, NPIndividual
from tengp import Parameters, FunctionSet
from tengp_eval.coevolution import TrainersSet, GaPredictors


def fitness_function(individual, x, y):
    output = individual.transform(x)
    try:
        #return adjusted_r2_score(y, output, len(x), len(individual.genes))
        return mean_squared_error(output, y)
    except ValueError:
        return 10e10

class cost_function:
    def __init__(self, X, Y, params, bounds):
        self.params = params
        self.bounds = bounds
        self.X = X
        self.Y = Y

    def fitness(self, x):
        individual = NPIndividual(list(x), self.bounds, self.params)

        fitness = fitness_function(individual, self.X, self.Y)

        return [fitness]


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
    params = Parameters(n_inputs, n_outputs, 1, 25, funset, real_valued=True, max_back=max_back)
    ib = IndividualBuilder(params)
    bounds = ib.create().bounds
    return ib, params, bounds

def run_benchmark_coevolution(cp, x_train, y_train, funset):
    ib, params, bounds = define_cgp_system(
            cp.getint('CGPPARAMS', 'n_nodes'),
            x_train.shape[1] if len(x_train.shape) > 1 else 1,
            y_train.shape[1] if len(y_train.shape) > 1 else 1,
            funset,
            cp.getint('CGPPARAMS', 'max_back'))

    # setup the coevolution elements
    ts = TrainersSet(ib, 16, fitness_function, x_train, y_train)
    predictors = GaPredictors(x_train, y_train, 10, 24)
    predictors.evaluate_fitness(ts)
    x_reduced, y_reduced = predictors.best_predictors_data()

    GENS_STEP = 50

    cf = cost_function(x_reduced, y_reduced, params, bounds)
    prob = pg.problem(cf)
    algo = pg.algorithm(pg.pso(
        gen=GENS_STEP,
        omega=cp.getfloat('OPTIMPARAMS', 'omega'),
        eta1=cp.getfloat('OPTIMPARAMS', 'eta1'),
        eta2=cp.getfloat('OPTIMPARAMS', 'eta2'),
        memory=True))
    algo.set_verbosity(1)
    pop = pg.population(prob, cp.getint('DEFAULT', 'population_size'))
    n_gens = GENS_STEP


    while n_gens < 500:

        pop = algo.evolve(pop)

        # calculate exact fitness of champion and
        # add it to the trainers set
        champion = NPIndividual(pop.champion_x, cf.bounds, cf.params)
        try:
            champion.fitness = fitness_function(champion, x_train, y_train)
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
        n_gens += GENS_STEP

    uda = algo.extract(pg.pso)

    champion = NPIndividual(pop.champion_x, cf.bounds, cf.params)
    champion.fitness = fitness_function(champion, x_train, y_train)


    fitnesses = [x[2] for x in uda.get_log()]
    fitnesses.append(champion.fitness)
    return fitnesses


def run_benchmark(cp, x_train, y_train, funset):
    ib, params, bounds = define_cgp_system(
            cp.getint('CGPPARAMS', 'n_nodes'),
            x_train.shape[1] if len(x_train.shape) > 1 else 1,
            y_train.shape[1] if len(y_train.shape) > 1 else 1,
            funset,
            cp.getint('CGPPARAMS', 'max_back'))
    cf = cost_function(x_train, y_train, params, bounds)
    prob = pg.problem(cf)
    algo = pg.algorithm(pg.pso(
        gen=cp.getint('DEFAULT', 'gens'),
        omega=cp.getfloat('OPTIMPARAMS', 'omega'),
        eta1=cp.getfloat('OPTIMPARAMS', 'eta1'),
        eta2=cp.getfloat('OPTIMPARAMS', 'eta2')))
    algo.set_verbosity(1)
    pop = pg.population(prob, cp.getint('DEFAULT', 'population_size'))
    pop = algo.evolve(pop)
    uda = algo.extract(pg.pso)

    return [x[2] for x in uda.get_log()]

RUNNERS = [run_benchmark, run_benchmark_coevolution]
