""" Module implementing the trainers set. """
from math import ceil, isinf


class TrainersSet:
    def __init__(self, individual_builder, size, fitness_fn, train_x, train_y):
        """
        Initialize with initial population

        Args:
            individual_builder (IndividualBuilder): used for generating individuals
            size (int): population size
            fintess_fn (callable): fitness function for trainers
            train_x (ndarray): full features
            train_y (ndarray): target vectors

        """
        self.train_x = train_x
        self.train_y = train_y
        self.ib = individual_builder
        self.fitness = fitness_fn
        initial_population = [self.ib.create() for _ in range(size)]
        pivot = ceil(size/2)
        self.best_pop = initial_population[0:pivot]
        self.random_pop = initial_population[pivot:]

        for trainer in self.best_pop + self.random_pop:
            trainer.fitness = self.fitness(trainer, train_x, train_y)
        self.best_pop.sort(key=lambda p: p.fitness)


    def add_trainer(self, trainer):
        if self.best_pop[0].fitness > trainer.fitness:
            self.best_pop.pop(0)
            self.best_pop.append(trainer)

    def update_random_population(self):
        return
        self.random_pop.pop(0)
        # prevent creation of trainers with invalid fitness
        fitness = float('inf')
        while isinf(fitness):
            rnd_individual = self.ib.create()
            fitness = self.fitness(rnd_individual, self.train_x, self.train_y)
        rnd_individual.fitness = fitness
        self.random_pop.append(rnd_individual)

    @property
    def population(self):
        return self.best_pop #+ self.random_pop


