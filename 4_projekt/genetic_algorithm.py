import numpy as np
import random
import time

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
        self.alpha = random.random()
        
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
        self.alpha = random.random()
        self.beta = random.random()
        
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

    def crossover_HX (self, P1, P2):
        alpha = random.random()
        if self.objective_function(P1) >= self.objective_function(P2):
            # C1.append(alpha*(P2[0] - P1[0]) + P2[0])
            # C1.append(alpha*(P2[1] - P1[1]) + P2[1])
            C1 = [alpha*(p2 - p1) + p2 for p1,p2 in zip(P1,P2)]
        else:
            # C1.append(alpha*(P1[0] - P2[0]) + P1[0])
            # C1.append(alpha*(P1[1] - P2[1]) + P1[1])
            C1 = [alpha*(p1 - p2) + p1 for p1,p2 in zip(P1,P2)]
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

    def fitness_weighted_cross_for_real_numbers(self, pop_list):
        pop = np.array(pop_list)
        num_vars = pop.shape[1]
        pop_size = pop.shape[0]
        alfa = random.uniform(0, 1)

        parent_num = 0
        parent_tab = np.zeros((0, num_vars), dtype=float)
        for i in range(0, pop_size):
            beta = (self.objective_function(pop[i]) - self.min_adap_func(pop, pop_size)) / (
                        self.max_adap_func(pop, pop_size) - self.min_adap_func(pop, pop_size))
            if beta < alfa:
                pop_i_reshaped = pop[i].reshape(1, *pop[i].shape)
                parent_tab = np.vstack([parent_tab, pop_i_reshaped])
                parent_num = parent_num + 1

        W = np.zeros((1, parent_num), dtype=float)
        W_old = np.zeros((1, parent_num), dtype=float)

        denominator = 0
        for i in range(0, parent_num):
            denominator = denominator + self.objective_function(parent_tab[i])

        for i in range(0, parent_num):
            W_old[0][i] = self.objective_function(parent_tab[i]) / float(denominator)

        new_denominator = 0
        for i in range(0, parent_num):
            new_denominator = new_denominator+1/W_old[0][i]

        for i in range(0, parent_num):
            W[0][i] = (1/W_old[0][i])/new_denominator

        f_desc = np.zeros((1, num_vars), dtype=float)
        for v in range(0, num_vars):
            meter = 0
            denominator = 0
            for p in range(0, parent_num):
                meter = meter + W[0][p]*parent_tab[p][v]
                denominator = denominator + W[0][p]
            #TODO: potencjalny błąd wynikający z dzielenia przez "0"
            if denominator == 0:
                f_desc[0][v] = 0
            else:
                f_desc[0][v] = meter/denominator

        return f_desc

    def min_adap_func(self, pop, pop_size):
        minim = self.objective_function(pop[0])
        for i in range(0, pop_size):
            if self.objective_function(pop[i]) < minim:
                minim = self.objective_function(pop[i])
        return minim

    def max_adap_func(self, pop, pop_size):
        maxim = self.objective_function(pop[0])
        for i in range(0, pop_size):
            if self.objective_function(pop[i]) > maxim:
                maxim = self.objective_function(pop[i])
        return maxim

    def max_ff_parent(self, pop):
        maxim_id = 0
        maxim = -1
        # print("len of pop: ", len(pop))
        for i in range(0, len(pop)):
            if self.objective_function(pop[i]) > maxim:
                maxim_id = i
                maxim = self.objective_function(pop[i])
        return maxim_id

    def crossover(self, parents, num_elite: int):
        missing_pop = self.population_size-num_elite
        children_fwx = np.zeros((0, len(parents[0])))
        children = []
        print("parents at begining of loop: ", type(parents))
        i = 0
        while len(children) < missing_pop and children_fwx.shape[0] < missing_pop:
            parent1, parent2 = parents[i], parents[i+1]
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
                    child1= self.average_crossover(parent1, parent2)
                    child2= self.average_crossover(parent1, parent2)
                elif self.crossover_type  == "crossover_HX": 
                    child1= self.crossover_HX(parent1, parent2)
                    child2= self.crossover_HX(parent1, parent2)
                elif self.crossover_type  == "SX":
                    child1= self.SX_version1(parent1, parent2)
                    child2= self.SX_version2(parent1, parent2)
                elif self.crossover_type  == "f1_PAX":
                    child1, child2 = self.f1_PAX(parent1, parent2)
                elif self.crossover_type == "FWX":
                    children_fwx = np.concatenate((children_fwx, self.fitness_weighted_cross_for_real_numbers(parents)), axis=0)
            else:
                child1, child2 = parent1, parent2

            if self.crossover_type != "FWX":
                children.append(child1)
                children.append(child2)

            if self.crossover_type == "FWX":
                children_fwx_list = children_fwx.tolist()
                for _ in range(0, len(children_fwx_list)):
                    idx_to_delete = self.max_ff_parent(parents)
                    parents = np.delete(parents, idx_to_delete, axis=0)
                if type(parents) == np.ndarray:
                    #print("parents: ", type(parents))
                    parents_list = parents.tolist()
                else:
                    parents_list = parents
                #print("parents_list: ", type(parents_list))
                parents_list.extend(children_fwx_list)
                children = parents_list


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
            elif self.mutation_method == "inversion":
                mutated_individual = self.inversion()
            mutated_population.append(mutated_individual)
        return np.array(mutated_population)

    def elitism(self, population, fitness_values):
        elite_size = int(self.population_size * self.elite_percentage)
        sorted_indices = np.argsort(fitness_values)
        sorted_population = population[sorted_indices]

        elite_population = sorted_population[:elite_size]

        return elite_population

    def inversion(self, population):
        inversed_population = np.copy(population)
        for i in range(inversed_population.shape[0]):
            if random.random() < self.mutation_prob:
                start_index = np.random.randint(population.shape[1])
                end_index = np.random.randint(start_index, population.shape[1])
                inversed_population[i, start_index:end_index+1] = np.flip(inversed_population[i, start_index:end_index+1])
        return inversed_population
    
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
            children = self.crossover(parents, elite_population.shape[0])
            mutated_children = self.mutate(children)

            new_population = np.vstack((elite_population, mutated_children))
            self.population = new_population

        end_time = time.time()
        execution_time = end_time - start_time

        return best_values, best_solution, average_values, std_dev_values, execution_time