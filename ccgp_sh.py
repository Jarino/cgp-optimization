"""
CCGP, sh, solid
"""
from configparser import ConfigParser


import numpy as np
from sklearn.metrics import mean_squared_error
from Solid.StochasticHillClimb import StochasticHillClimb

from tengp.individual import IndividualBuilder, NPIndividual
from tengp import Parameters, FunctionSet

def fitness(individual, x_train, y_train):
    output = individual.transform(x_train)
    try:
        fitness = -mean_squared_error(output, y_train)
    except ValueError:
        fitness = -10e10

    return fitness


class Algorithm(StochasticHillClimb):
    """
    Optimize the CCGP individual
    """
    def custom_init(self,
                    mut_prob,
                    x_train,
                    y_train,
                    bounds,
                    params):
        self.mut_prob = mut_prob
        self.x_train = x_train
        self.y_train = y_train
        self.bounds = bounds
        self.params = params

    def _neighbor(self):
        new_genes = np.array(self.current_state.genes[:])

        chances = np.random.random(len(new_genes))
        mask = chances < self.mut_prob
        new_genes[mask] = np.random.normal(new_genes[mask], 2)
        new_genes = np.clip(new_genes, self.bounds[0], self.bounds[1])

        new_individual = NPIndividual(new_genes.tolist(), self.bounds, self.params)
        new_individual.fitness = self._objective(new_individual)

        return new_individual

    def _objective(self, member):
        if member.fitness is not None:
            return member.fitness

        member.fitness = fitness(member, self.x_train, self.y_train)

        return member.fitness

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

    initial_individual = ib.create()
    initial_individual.fitness = fitness(initial_individual, x_train, y_train)

    algo = Algorithm(initial_individual,
                    temp=cp.getfloat('DEFAULT', 'temperature'),
                    max_steps=cp.getint('DEFAULT', 'max_evals'),
                    max_objective=0)
    algo.custom_init(
            cp.getfloat('DEFAULT', 'mut_prob'),
            x_train,
            y_train,
            bounds,
            params)


    best_x, best_y = algo.run(verbose=False)

    return algo.log[:]


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
        ('nguyen4', symreg.get_benchmark_poly(random, 6), nguyen7_funset),
        ('nguyen7', symreg.get_benchmark_nguyen7(random, None), nguyen7_funset),
        ('pagie1', symreg.get_benchmark_pagie1(random, None), pagie_funset),
        ('keijzer6', symreg.get_benchmark_keijzer(random, 6), keijzer_funset),
        ('korns12', symreg.get_benchmark_korns(random, 12), korns12_funset),
        ('vladislasleva4', symreg.get_benchmark_vladislasleva4(random, None), vlad_funset)
    ]

    start = time()

    for name, (x_train, y_train, x_test, y_test), funset in benchmarks:
        x_train = np.c_[np.ones(len(x_train)), x_train]

        max_trials = cp.getint('DEFAULT', 'trials')

        for i in range(max_trials):
            print(f'{name}-{i}')
            logs = run_benchmark(cp, x_train, y_train, nguyen7_funset)

        with open(os.path.join(output_dir, f'{name}.log'), 'wb') as f:
            pickle.dump(logs, f)

    print(f'finised, wall time: {time() - start}')
