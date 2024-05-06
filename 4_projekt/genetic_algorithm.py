import numpy as np
import random
import time
from numpy import sin
from numpy import sqrt

class GeneticAlgorithm:
    def __init__(self, min_ss, max_ss, objective_function, num_variables, population_size=100, num_epochs=100,
                 crossover_prob=0.8, mutation_prob=0.1, elite_percentage=0.6, num_selected=4, selection_method="tournament", crossover_type="single_point", grain_size=2, precision=2,
                 mutation_method="single point"):
        self.max_ss = max_ss
        self.min_ss = min_ss
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.population_size = population_size
        self.num_epochs = num_epochs
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_percentage = elite_percentage
        self.num_selected = num_selected
        self.selection_method = selection_method
        self.crossover_type = crossover_type
        self.grain_size = grain_size
        self.precision = precision
        self.mutation_method = mutation_method
        self.population = self.initialize_population()
        self.population_masks = np.random.randint(2, size=(self.population_size,self.num_variables))

    def initialize_population(self):
        return np.random.uniform(self.min_ss, self.max_ss, size=(self.population_size, self.num_variables))

    def evaluate_population(self):
        return np.array([self.objective_function(individual) for individual in self.population])

    def evaluate_subject(self, subject):
        return self.objective_function(subject)

    def select_parents(self, fitness_values):
        if self.selection_method == "best":
            return self.select_best_parents(fitness_values)
        elif self.selection_method == "roulette_wheel":
            return self.select_roulette_wheel_parents(fitness_values)
        elif self.selection_method == "tournament":
            return self.select_tournament_parents(fitness_values)

    def select_best_parents(self, fitness_values):
        best_index = np.argmin(fitness_values)
        best_individual = self.population[best_index]
        return [best_individual] * len(self.population)

    def select_roulette_wheel_parents(self, fitness_values):
        total_fitness = np.sum(fitness_values)
        probabilities = fitness_values / total_fitness
        selected_indices = np.random.choice(len(self.population), size=len(self.population), p=probabilities)
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def select_tournament_parents(self, fitness_values):
        selected_indices = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(len(fitness_values), size=self.num_selected, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(winner_index)
        return self.population[selected_indices]

    def arithmetic_crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def linear_crossover(self, parent1, parent2):
        n = len(parent1)
        Z = np.empty(n)
        V = np.empty(n)
        W = np.empty(n)
        
        for i in range(n):
            Z[i] = parent1[i]/2 + parent2[i]/2
            V[i] = parent1[i]/2*3 - parent2[i]/2
            W[i] = -parent1[i]/2 + parent2[i]/2*3
        
        child1 = self.select_best_vector([Z, V, W])
        child2 = self.select_best_vector([Z, V, W])
        
        return child1, child2

    def select_best_vector(self, vectors):
        fitness_values = [self.evaluate_subject(v) for v in vectors]
        best_index = np.argmax(fitness_values)
        return vectors[best_index]

    def alpha_mixed_crossover(self, parent1, parent2):
        n = len(parent1)
        child1 = np.empty(n)
        child2 = np.empty(n)
        self.alpha = random.random() #TODO pomyslec czy w ten sposob czy podawac z innymi parametrami
        
        for i in range(n):
            d = parent1[i] - parent2[i]
            min_value = min(parent1[i], parent2[i]) - self.alpha * d
            max_value = max(parent1[i], parent2[i]) + self.alpha * d
            
            u1 = np.random.uniform(min_value, max_value)
            u2 = np.random.uniform(min_value, max_value)
            
            child1[i] = u1
            child2[i] = u2
        
        return child1, child2

    def alpha_beta_mixed_crossover(self, parent1, parent2):
        n = len(parent1)
        child1 = np.empty(n)
        child2 = np.empty(n)
        self.alpha = random.random() #TODO pomyslec czy w ten sposob czy podawac z innymi parametrami
        self.beta = random.random() #TODO pomyslec czy w ten sposob czy podawac z innymi parametrami
        
        for i in range(n):
            d = parent1[i] - parent2[i]
            if parent1[i] <= parent2[i]:
                min_value = parent1[i] - self.alpha * d
                max_value = parent2[i] + self.beta * d
            else:
                min_value = parent2[i] - self.beta * d
                max_value = parent1[i] + self.alpha * d
            
            u1 = np.random.uniform(min_value, max_value)
            u2 = np.random.uniform(min_value, max_value)
            
            child1[i] = u1
            child2[i] = u2
        
        return child1, child2
    
    def average_crossover(self, parent1, parent2):
        child = (parent1 + parent2) / 2
        return child
    
    #TODO to jakos?
    def schaffer_f2(self, x):
        return 0.5 + ((sin(x[0]**2 + x[1]**2))**2 - 0.5) / ((1 + 0.001*(x[0]**2 + x[1]**2))**2)

    def crossover_HX (self, P1, P2):
        C1 = []
        # alpha = random.random()
        alpha = 0.5
        if self.schaffer_f2(P1) >= self.schaffer_f2(P2):
            C1.append(alpha*(P2[0] - P1[0]) + P2[0])
            C1.append(alpha*(P2[1] - P1[1]) + P2[1])
        else:
            C1.append(alpha*(P1[0] - P2[0]) + P1[0])
            C1.append(alpha*(P1[1] - P2[1]) + P1[1])
        return C1
    
    def SX_version1(self, parent1, parent2):
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        n = len(parent1)
        child = []
        for i in range(n):
            child.append(sqrt((parent1[i] ** 2 + parent2[i] ** 2) / 2))
        
        return child

    def SX_version2(self, parent1, parent2):
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        n = len(parent1)
        alpha = random.uniform(0, 1)
        
        child = []
        for i in range(n):
            child.append(sqrt(alpha * parent1[i] ** 2 + (1 - alpha) * parent2[i] ** 2))
        
        return child
    
    def f1_PAX(self, parent1, parent2):
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")

        size = len(parent1)

        cp = random.randint(0, (size - 1))

        child1 = parent1[:]
        child2 = parent2[:]

        child1[cp] = (parent1[cp] + parent2[cp]) / 2
        child2[cp] = (parent1[cp] + parent2[cp]) / 2

        return child1, child2

    def crossover(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i+1]
            if random.random() < self.crossover_prob:
                if self.crossover_type == "arithmetic":
                    child1, child2 = self.arithmetic_crossover(parent1, parent2)
                elif self.crossover_type == "linear":
                    child1, child2 = self.linear_crossover(parent1, parent2)
                elif self.crossover_type == "alpha_mixed":
                    child1, child2 = self.alpha_mixed_crossover(parent1, parent2)
                elif self.crossover_type  == "alpha_beta_mixed":
                    child1, child2 = self.alpha_beta_mixed_crossover(parent1, parent2)
                elif self.crossover_type  == "average":
                    #TODO pomyslec czy 2 identyczne maja sens czy jakos to obejsc
                    child1= self.average_crossover(parent1, parent2)
                    child2= self.average_crossover(parent1, parent2)
                elif self.crossover_type  == "crossover_HX": 
                    #TODO przy wiecej niz 2 zmiennych sie wywala
                    child1= self.crossover_HX(parent1, parent2)
                    child2= self.crossover_HX(parent1, parent2)
                elif self.crossover_type  == "SX":
                    child1= self.SX_version1(parent1, parent2)
                    child2= self.SX_version2(parent1, parent2)
                elif self.crossover_type  == "f1_PAX":
                    child1, child2 = self.f1_PAX(parent1, parent2)
                #TODO dodac krzyzowanie Wojtka bo jak na nie patrze to mi słabo
            else:
                child1, child2 = parent1, parent2
                
            children.append(child1)
            children.append(child2)

            if len(parents) % 2 != 0:
                children.append(parents[-1])

        return np.array(children)[:self.population_size]

    def get_dimensions(self, lst):
        if isinstance(lst, list):
            return [len(lst)] + self.get_dimensions(lst[0])
        else:
            return []

    def uniform_mutation(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                individual[i] += np.random.uniform(-self.mutation_prob, self.mutation_prob)
                individual[i] = np.clip(individual[i], self.min_ss, self.max_ss)
        return individual

    def gaussian_mutation(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                individual[i] += np.random.normal(0, self.mutation_prob)
                individual[i] = np.clip(individual[i], self.min_ss, self.max_ss)
        return individual

    def mutate(self, population):
        mutated_population = []
        for individual in population:
            if self.mutation_method == "uniform":
                mutated_individual = self.uniform_mutation(individual)
            elif self.mutation_method == "gaussian":
                mutated_individual = self.gaussian_mutation(individual)
            mutated_population.append(mutated_individual)
        return np.array(mutated_population)

    def elitism(self, population, fitness_values):
        elite_size = int(self.population_size * self.elite_percentage)
        sorted_indices = np.argsort(fitness_values)
        sorted_population = population[sorted_indices]

        elite_population = sorted_population[:elite_size]

        return elite_population

    def inversion(self, population):
        inversed_population = population.copy()
        for i in range(len(inversed_population)):
            if random.random() < self.mutation_prob:
                mutation_point = random.randint(0, len(inversed_population[i]) - 1)
                selected_bit = random.randint(0, self.num_bits - 1)
                selected_bit2 = random.randint(selected_bit, self.num_bits - 1)
                inversed_population[i][mutation_point][selected_bit:selected_bit2+1] = np.flip(inversed_population[i][mutation_point][selected_bit:selected_bit2+1])
        return inversed_population

    #TODO zmienic strategie ewolucyjna na jakąś z jego wykładu jeśli trzeba?
    def evolve(self):
        best_values = []
        best_solutions = []
        average_values = []
        std_dev_values = []

        start_time = time.time()

        for epoch in range(self.num_epochs):
            fitness_values = self.evaluate_population()
            best_index = np.argmin(fitness_values)
            best_values.append(fitness_values[best_index])
            best_solution = self.population[best_index]
            best_solutions.append(best_solution)

            average_values.append(np.mean(fitness_values))
            std_dev_values.append(np.std(fitness_values))

            print(f"Epoch {epoch}: Best Value = {best_values[-1]}, Solution = {best_solution}")

            elite_population = self.elitism(self.population, fitness_values)

            parents = self.select_parents(fitness_values)
            children = self.crossover(parents)
            mutated_children = self.mutate(children)
            #TODO inwersja jesli trzeba
            #inversed_population = self.inversion(mutated_children)

            #new_population = np.vstack((elite_population, inversed_population))
            new_population = np.vstack((elite_population, mutated_children))
            self.population = new_population

        end_time = time.time()
        execution_time = end_time - start_time

        return best_values, best_solution, average_values, std_dev_values, execution_time