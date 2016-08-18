from random import choice


class Raffle:
    def __init__(self, population, fitness_f):
        self.population = population
        self.tickets = []

        for i, individual in enumerate(population):
            self.tickets.extend([i] * fitness_f(individual))

    def get_individual(self, i):
        return self.population[i]

    def draw_index(self):
        return choice(self.tickets)

    def draw_pair(self):
        a = b = self.draw_index()

        while b == a:
            b = self.draw_index()

        return self.get_individual(a), self.get_individual(b)
