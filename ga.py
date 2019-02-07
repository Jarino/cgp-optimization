"""
Module containing functions for genetic algorithm """
import numpy as np
from random import choice, random, randint
from math import isinf

class GaIndividual:
    def __init__(self, genes):
        self.fitness = None
        self.genes = genes

def mutate(individual, prob, values):
    """
    Return a mutated copy of individual.

    Args:
        individual (GaIndividual): individual to mutated
        prop (float): probability of gene mutation (value between 0 and 1)
        values (array-like): possible gene values

    Returns:
        array-like: mutated copy of individual
    """
    child = GaIndividual(individual.genes[:])
    i = randint(0, len(child.genes) - 1)
    child.genes[i] = choice(values)
    return child


def tournament_selection(population, k):
    """
    Tournament selection of individuals, based on their fitness values.

    Args:
        population (list): list of GaIndividual
        k (integer): tournament size

    Returns:
        GaIndividual
    """
    best = choice(population)
    for i in range(1, k + 1):
        ind = choice(population)
        if ind.fitness < best.fitness:
            best = ind
    return best


def crossover(a, b):
    """
    Performs single-point crossover.

    Args:
        a (GaIndividual): first parent
        b (GaIndividual): second parent

    Returns:
        tuple: two GaIndividual instances
    """
    pivot = randint(0, len(a.genes) - 1)

    child_a = a.genes[0:pivot] + b.genes[pivot:]
    child_b = b.genes[0:pivot] + a.genes[pivot:]

    return GaIndividual(child_a), GaIndividual(child_b)

class GaPredictors:

    def __init__(self, train_x, train_y, fraction, size):
        """
        Initialize the GaPredictors population
	Args:
	    train_x (array): features
	    train_y (array): target vectors
	    fraction (int): fraction of data to use for predictors
	    size (int): population size
	"""
        self.train_x = train_x
        self.train_y = train_y
        self.population_size = size

        self.predictor_size = len(train_x)//fraction

        self.gene_values = list(range(len(train_x)))

        self.population = []

        for _ in range(size):
            self.population.append(GaIndividual([randint(0, len(train_x)-1) for _ in range(self.predictor_size)]))

    def predictor_data(self, genes):
        """
        Probably should be in utils or something
        """
        return self.train_x[np.ix_(genes)], self.train_y[np.ix_(genes)]

    def evaluate_fitness(self, trainers_set):
        """
        Assign fitness to predictors in populations

        Args:
            trainers (TrainersSet): trainers used to calculate predictors fitness

        """
        for predictor in self.population:
            total = 0
            for trainer in trainers_set.population:
                reduced_train_x, reduced_train_y = self.predictor_data(
                        predictor.genes)
                f_predicted = trainers_set.fitness(trainer, reduced_train_x, reduced_train_y)

                total += abs(trainer.fitness - f_predicted)
            predictor.fitness = total/len(trainers_set.population)


    def predictors_evolution_step(self, trainers):
        new_population = []

        while len(new_population) < self.population_size:
            parent_a = tournament_selection(self.population, 2)
            parent_b = tournament_selection(self.population, 2)

            children = crossover(parent_a, parent_b)
            for child in children:
                if random() > 0.9:
                    child = mutate(child, 0.1, self.gene_values)
                new_population.append(child)
        self.population = new_population
        self.evaluate_fitness(trainers)

    @property
    def best_predictor(self):
        best = min(self.population, key=lambda p: p.fitness)
        return best


    def best_predictors_data(self):
        best = self.best_predictor
        return self.predictor_data(best.genes)

