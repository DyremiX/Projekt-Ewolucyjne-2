import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import benchmark_functions as bf
import opfunu as opf

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
        self.num_bits = int(np.ceil(np.log2((self.max_ss-self.min_ss)*10**self.precision) + np.log2(1)))
        return np.random.randint(2, size=(self.population_size, self.num_variables, self.num_bits))

    def binary_to_decimal(self, binary_population, precision):
        if binary_population.ndim == 3:  # Dla całej populacji
            population_size, num_variables, num_bits = binary_population.shape
            decimal_population = np.zeros((population_size, num_variables))

            for i in range(population_size):
                for v in range(num_variables):
                    decimal_value = 0
                    for j in range(num_bits):
                        decimal_value += binary_population[i][v][j] * (2 ** (num_bits - j - 1))
                    decimal_population[i][v] = self.min_ss + decimal_value  / (10 ** precision) #TODO: Nie jest zgodne z wzorem, ale działa (prawie) idealnie
        elif binary_population.ndim == 2:  # Dla pojedynczego najlepszego rozwiązania
            num_variables, num_bits = binary_population.shape
            decimal_solution = np.zeros(num_variables)

            for v in range(num_variables):
                decimal_value = 0
                for j in range(num_bits):
                    decimal_value += binary_population[v][j] * (2 ** (num_bits - j - 1))
                decimal_solution[v] = self.min_ss + decimal_value  / (10 ** precision)#TODO: Nie jest zgodne z wzorem, ale działa (prawie) idealnie

            decimal_population = decimal_solution
        else:
            raise ValueError("Niepoprawny wymiar tablicy binarnej.")

        return decimal_population

    def evaluate_population(self):
        decimal_population = self.binary_to_decimal(self.population, self.precision)
        return np.array([self.objective_function(individual) for individual in decimal_population])

    def evaluate_subject(self, subject):
        decimal_subject = self.binary_to_decimal(subject, self.precision)
        return self.objective_function(decimal_subject)

    def select_parents(self, fitness_values):
        if self.selection_method == "best":
            return self.select_best_parents(fitness_values)
        elif self.selection_method == "roulette_wheel":
            return self.select_roulette_wheel_parents(fitness_values)
        elif self.selection_method == "tournament":
            return self.select_tournament_parents(fitness_values)

    def select_best_parents(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        return self.population[sorted_indices[-self.population_size:]]

    def select_roulette_wheel_parents(self, fitness_values):
        inverted_fitness_values = 1 / fitness_values
        total_inverted_fitness = np.sum(inverted_fitness_values)
        probabilities = inverted_fitness_values / total_inverted_fitness
        selected_indices = np.random.choice(
            range(len(fitness_values)),
            size=self.population_size,
            replace=True,
            p=probabilities
        )
        return self.population[selected_indices]


    def select_tournament_parents(self, fitness_values):
        selected_indices = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(len(fitness_values), size=self.num_selected, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(winner_index)
        return self.population[selected_indices]

    def single_point_crossover(self, parent1, parent2):
        children1 = []
        children2 = []
        for i in range(len(parent1)):
            crossover_point = random.randint(1, len(parent1[i]) - 1)
            child1 = np.concatenate((parent1[i][:crossover_point], parent2[i][crossover_point:]))
            child2 = np.concatenate((parent2[i][:crossover_point], parent1[i][crossover_point:]))
            children1.append(child1)
            children2.append(child2)
        return children1, children2

    def two_point_crossover(self, parent1, parent2):
        children1 = []
        children2 = []
        for i in range(len(parent1)):
            crossover_points = sorted(random.sample(range(1, len(parent1[i])), 2))
            child1 = np.concatenate((parent1[i][:crossover_points[0]], parent2[i][crossover_points[0]:crossover_points[1]], parent1[i][crossover_points[1]:]))
            child2 = np.concatenate((parent2[i][:crossover_points[0]], parent1[i][crossover_points[0]:crossover_points[1]], parent2[i][crossover_points[1]:]))
            children1.append(child1)
            children2.append(child2)
        return children1, children2

    def three_point_crossover(self, parent1, parent2):
        children1 = []
        children2 = []
        for i in range(len(parent1)):
            crossover_points = sorted(random.sample(range(1, len(parent1[i])), 3))
            child1 = np.concatenate((parent1[i][:crossover_points[0]],
                                    parent2[i][crossover_points[0]:crossover_points[1]],
                                    parent1[i][crossover_points[1]:crossover_points[2]],
                                    parent2[i][crossover_points[2]:]))
            child2 = np.concatenate((parent2[i][:crossover_points[0]],
                                    parent1[i][crossover_points[0]:crossover_points[1]],
                                    parent2[i][crossover_points[1]:crossover_points[2]],
                                    parent1[i][crossover_points[2]:]))
            children1.append(child1)
            children2.append(child2)
        return children1, children2

    def uniform_crossover(self, parent1, parent2):
        children1 = []
        children2 = []
        for i in range(len(parent1)):
            mask_shape = parent1[i].shape
            mask = np.random.randint(2, size=mask_shape)
            child1 = np.where(mask, parent1[i], parent2[i])
            child2 = np.where(mask, parent2[i], parent1[i])
            children1.append(child1)
            children2.append(child2)
        return children1, children2

    def grainy_crossover(self, parent1, parent2):
        children1 = []
        children2 = []
        for i in range(len(parent1)):
            child1, child2 = parent1[i].copy(), parent2[i].copy()
            for j in range(0, len(parent1[i]), self.grain_size):
                if random.random() <= 0.5:
                    child1[j] = parent1[i][j]
                    child2[j] = parent2[i][j]
                else:
                    child1[j] = parent2[i][j]
                    child2[j] = parent1[i][j]
            children1.append(child1)
            children2.append(child2)
        return children1, children2

    def RRC(self, A, B, n):
        children1 = []
        children2 = []
        for i in range(n):
            S = np.where(A[i] == B[i], A[i], None)
            mask_shape = A[i].shape
            mask = np.where(S != None, S, np.where(np.random.uniform(0, 1, size=mask_shape) <= 0.5, 1, 0))
            C = np.where(S != None, S, mask)
            D = np.where(S != None, S, np.where(np.random.uniform(0, 1, size=mask_shape) <= 0.5, 1, 0))
            children1.append(D)
            children2.append(C)
        return children1, children2


    def crossover_by_dominance(self, parent_A, parent_B, mask_A, mask_B):
        child_C = parent_A[:]
        child_D = parent_B[:]
        
        for i in range(len(parent_A)):
            if mask_B[i] == 1 and mask_A[i] == 0:  # dominacja B z uwagi na maskę
                child_C[i] = parent_B[i]
            if mask_B[i] == 0 and mask_A[i] == 1:  # dominacja A z uwagi na maskę
                child_D[i] = parent_A[i]
        self.update_population_mask(parents=[parent_A,parent_B],children=[child_C,child_D])
        return child_C, child_D

    def DIS(self, ind1, ind2, q, number_of_features):
        size = len(ind1[0])
        new_inds = []

        for feature in range(number_of_features):
            new_ind = []
            for i in range(q):
                if ind1[feature][i] != ind2[feature][i]:
                    new_ind.append(ind1[feature][i])
                else:
                    new_ind.append(np.random.randint(0, 2))
            for i in range(q, size):
                if ind1[feature][i] != ind2[feature][i]:
                    new_ind.append(ind2[feature][i])
                else:
                    new_ind.append(np.random.randint(0, 2))
            new_inds.append(np.array(new_ind))

        return np.array(new_inds)
    
    def adaption_weighted_cross(self, pop):
        n = pop.shape[2]
        num_vars = pop.shape[1]
        pop_size = pop.shape[0]
        alfa = random.uniform(0, 1)

        parent_num = 0
        parent_tab = np.zeros((0, num_vars, n), dtype=int)
        for i in range(0, pop_size):
            beta = (self.adap_func(pop[i]) - self.min_adap_func(pop, pop_size))/(self.max_adap_func(pop, pop_size) - self.min_adap_func(pop, pop_size))
            if beta < alfa:
                pop_i_reshaped = pop[i].reshape(1, *pop[i].shape)
                parent_tab = np.vstack([parent_tab, pop_i_reshaped])
                parent_num = parent_num + 1

        W = np.zeros((1, parent_num), dtype=float)

        denominator = 0
        for i in range(0, parent_num):
            denominator = denominator + self.adap_func(parent_tab[i])

        for i in range(0, parent_num):
            W[0][i] = self.adap_func(parent_tab[i])/float(denominator)

        tab = np.zeros((parent_num, num_vars, n), dtype=int)
        for j in range(0, parent_num):
            for v in range(0, num_vars):
                for i in range(0, n):
                    if parent_tab[j][v][i] == 1:
                        tab[j][v][i] = 1
                    else:
                        tab[j][v][i] = -1

        f_desc = np.zeros((1, num_vars, n), dtype=int)
        for v in range(0, num_vars):
            for i in range(0, n):
                lambd = self.calc_sign(W, tab, v, i, parent_num)
                if lambd >= 0:
                    f_desc[0][v][i] = 1
                else:
                    f_desc[0][v][i] = 0

        return f_desc

    def calc_sign(self, W, tab, var_num, gen_num, parent_num):
        calc = 0
        for j in range(0, parent_num):
            calc = calc+W[0][j]*tab[j][var_num][gen_num]
        if calc >= 0:
            return 1
        else:
            return -1

    def adap_func(self, single_element):
        #print("SES: ", single_element.shape)
        val = self.binary_to_decimal(single_element, self.precision)
        return self.objective_function(val)

    def min_adap_func(self, pop, pop_size):
        minim = self.adap_func(pop[0])
        for i in range(0, pop_size):
            if self.adap_func(pop[i]) < minim:
                minim = self.adap_func(pop[i])
        return minim

    def max_adap_func(self, pop, pop_size):
        maxim = self.adap_func(pop[0])
        for i in range(0, pop_size):
            if self.adap_func(pop[i]) > maxim:
                maxim = self.adap_func(pop[i])
        return maxim

    def max_ff_parent(self, pop):
        maxim_id = 0
        maxim = -1
        # print("len of pop: ", len(pop))
        for i in range(0, len(pop)):
            if self.adap_func(pop[i]) > maxim:
                maxim_id = i
                maxim = self.adap_func(pop[i])
        return maxim_id
    
    def find_subject(self, subject):
        for i in range(len(self.population)):
            if np.array_equal(self.population[i], subject):           
                return i
        return "Błąd, nie ma takiego osobnika"
    
    def update_mask(self, subject, new_mask):
        self.population_masks[self.find_subject(subject=subject)] = new_mask
        pass

    def update_population_mask(self, parents, children):
        self.population_masks = np.vstack((self.population_masks, np.zeros((len(children), self.num_variables))))
        if len(parents) != len(children):
            raise Exception(f"Błąd, niezgodna ilość dzieci i rodziców (P:{parents} != C:{children})")
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = children[i], children[i+1]
            paretns_score = (self.evaluate_subject(parent1) + self.evaluate_subject(parent2))/2
            child_score = (self.evaluate_subject(child1) + self.evaluate_subject(child2))/2 
            if child_score > paretns_score: # W przypadku słabszego wyniku potomków wybieramy nowy maski
                self.update_mask(child1,np.random.randint(2, size=(self.num_variables)))
                self.update_mask(child2,np.random.randint(2, size=(self.num_variables)))
                self.update_mask(parent1,np.random.randint(2, size=(self.num_variables)))
                self.update_mask(parent2,np.random.randint(2, size=(self.num_variables)))
            else:
                self.update_mask(child1,self.population_masks[self.find_subject(parent1)])
                self.update_mask(child2,self.population_masks[self.find_subject(parent2)])
        pass

    def crossover(self, parents, num_elite: int):
        missing_pop = self.population_size-num_elite
        children = []
        shape = (0,) + parents.shape[1:]
        children_fwx = np.zeros(shape)
        i = 0
        while len(children) < missing_pop and children_fwx.shape[0] < missing_pop:
            parent1, parent2 = parents[i], parents[i+1]
            if random.random() < self.crossover_prob:
                if self.crossover_type == "single_point":
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                elif self.crossover_type == "two_point":
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                elif self.crossover_type == "three_point":
                    child1, child2 = self.three_point_crossover(parent1, parent2)
                elif self.crossover_type == "uniform":
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                elif self.crossover_type == "grainy":
                    child1, child2 = self.uniform_crossover(parent1, parent2)  
                elif self.crossover_type == "dominance":
                    child1, child2 = self.crossover_by_dominance(parent1, parent2, self.population_masks[self.find_subject(parent1)], self.population_masks[self.find_subject(parent1)])
                elif self.crossover_type == "Random Respectful":
                    child1, child2 = self.RRC(parent1, parent2, len(parent1))
                elif self.crossover_type == "DIS":
                    q = np.random.randint(1, len(parent1[0]))
                    child1 = self.DIS(parent1, parent2, q, len(parent1))
                    child2 = self.DIS(parent1, parent2, q, len(parent1))
                elif self.crossover_type == "adaption weighted":
                    children_fwx = np.concatenate((children_fwx, self.adaption_weighted_cross(parents)), axis=0)
                else:
                    child1, child2 = self.single_point_crossover(parent1, parent2)

                if self.crossover_type != "adaption weighted":
                    children.extend([child1, child2])
            else:
                children.extend([parent1, parent2])
            if (i+3) < (len(parents) - 1):
                i = i+2
            else:
                i = 0

        if self.crossover_type == "adaption weighted":
            children_fwx_list = children_fwx.tolist()
            for _ in range(0, len(children_fwx_list)):
                idx_to_delete = self.max_ff_parent(parents)
                parents = np.delete(parents, idx_to_delete, axis=0)
            parents_list = parents.tolist()
            parents_list.extend(children_fwx_list)
            children = parents_list

        if len(parents) % 2 != 0:
            children.append(parents[-1])

        return np.array(children)

    def get_dimensions(self, lst):
        if isinstance(lst, list):
            return [len(lst)] + self.get_dimensions(lst[0])
        else:
            return []

    def boundary_mutation(self, population):
        mutated_population = population.copy()
        for i in range(len(mutated_population)):
            for j in range(len(mutated_population[i])):
                if random.random() < self.mutation_prob:
                    mutated_population[i][j][len(mutated_population[i][j])-1] = 1 - mutated_population[i][j][len(mutated_population[i][j])-1]
        return mutated_population

    def single_point_mutation(self, population):
        mutated_population = population.copy()
        for i in range(len(mutated_population)):
            if random.random() < self.mutation_prob:
                mutation_point = random.randint(0, len(mutated_population[i]) - 1)
                selected_bit = random.randint(0, self.num_bits - 1)
                mutated_population[i][mutation_point][selected_bit] = 1 - mutated_population[i][mutation_point][selected_bit]
        return mutated_population

    def two_point_mutation(self, population):
        mutated_population = population.copy()
        for i in range(len(mutated_population)):
            if random.random() < self.mutation_prob:
                mutation_point = random.randint(0, len(mutated_population[i]) - 1)
                selected_bit = random.randint(0, self.num_bits - 1)
                selected_bit2 = random.randint(0, self.num_bits - 1)
                mutated_population[i][mutation_point][selected_bit] = 1 - mutated_population[i][mutation_point][selected_bit]
                mutated_population[i][mutation_point][selected_bit2] = 1 - mutated_population[i][mutation_point][selected_bit2]
        return mutated_population

    def mutate(self, population):
        if self.mutation_method == "boundary":
            return self.boundary_mutation(population)
        elif self.mutation_method == "single point":
            return self.single_point_mutation(population)
        elif self.mutation_method == "two point":
            return self.two_point_mutation(population)
        else:
            raise ValueError("Invalid mutation method specified")

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

            decimal_best_solution = self.binary_to_decimal(best_solution, self.precision)
            print(f"Epoch {epoch}: Best Value = {best_values[-1]}, Solution = {decimal_best_solution}")

            elite_population = self.elitism(self.population, fitness_values)

            parents = self.select_parents(fitness_values)
            children = self.crossover(parents, elite_population.shape[0])
            mutated_children = self.mutate(children)
            inversed_population = self.inversion(mutated_children)

            new_population = np.vstack((elite_population, inversed_population))
            self.population = new_population

        end_time = time.time()
        execution_time = end_time - start_time

        return best_values, decimal_best_solution, average_values, std_dev_values, execution_time


from numpy import sin
from numpy import sqrt

def keane_function(x):
    # x_decimal = np.array(x)
    # N = len(x_decimal)
    # sum_cos4 = np.sum(np.cos(x)**4)
    # prod_cos2 = np.prod(np.cos(x)**2)
    # inner_term = np.abs(sum_cos4 - prod_cos2)
    # sum_x_squared = np.sum([x[i]**2 * (i + 1) for i in range(N)])
    # result = -inner_term * sum_x_squared**(-0.5)
    # return result
    x1, x2 = x[0], x[1]
    a = -sin(x1 - x2)**2 * sin(x1 + x2)**2
    b = sqrt(x1**2 + x2**2)   
    c = a / b
    return c

def tst_function(x):
    x_decimal = np.array(x)
    result = np.sum(np.square(x_decimal))+5
    return result

def hgbat_function(x):
    x_decimal = np.array(x)
    D = x_decimal.shape[0]
    first_part = np.sqrt(np.abs(np.sum(np.square(x_decimal))**2 - np.sum(x_decimal)**2))
    second_part = (0.5*np.sum(np.square(x_decimal))+np.sum(x_decimal))/D + 0.5
    result = first_part + second_part
    return result

class GeneticAlgorithmGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Genetic Algorithm")
        self.root.geometry("600x650")

        self.num_variables_var = tk.IntVar(value=5)
        self.max_ss_var = tk.DoubleVar(value=10.0)
        self.min_ss_var = tk.DoubleVar(value=0.0)
        self.population_size_var = tk.IntVar(value=100)
        self.num_epochs_var = tk.IntVar(value=100)
        self.crossover_prob_var = tk.DoubleVar(value=0.8)
        self.mutation_prob_var = tk.DoubleVar(value=0.1)
        self.elite_percentage_var = tk.DoubleVar(value=0.6)
        self.selection_method_var = tk.StringVar(value="tournament")
        self.crossover_type_var = tk.StringVar(value="single point")
        self.precision_var = tk.IntVar(value=10)
        self.mutate_type_var = tk.StringVar(value="single point")
        self.objective_function_var = tk.StringVar(value="keane_function")

        self.create_widgets()

    def create_widgets(self):
        label1 = tk.Label(self.root, text="Number of Variables:")
        label1.pack()
        entry1 = tk.Entry(self.root, textvariable=self.num_variables_var, width=50, bg='dark gray')
        entry1.pack()

        label1a = tk.Label(self.root, text="Min Search Size:")
        label1a.pack()
        entry1a = tk.Entry(self.root, textvariable=self.min_ss_var, width=50, bg='dark gray')
        entry1a.pack()

        label1b = tk.Label(self.root, text="Max Search Size:")
        label1b.pack()
        entry1b = tk.Entry(self.root, textvariable=self.max_ss_var, width=50, bg='dark gray')
        entry1b.pack()

        label2 = tk.Label(self.root, text="Population Size:")
        label2.pack()
        entry2 = tk.Entry(self.root, textvariable=self.population_size_var, width=50, bg='dark gray')
        entry2.pack()

        label3 = tk.Label(self.root, text="Number of Epochs:")
        label3.pack()
        entry3 = tk.Entry(self.root, textvariable=self.num_epochs_var, width=50, bg='dark gray')
        entry3.pack()

        label4 = tk.Label(self.root, text="Crossover Probability:")
        label4.pack()
        entry4 = tk.Entry(self.root, textvariable=self.crossover_prob_var, width=50, bg='dark gray')
        entry4.pack()

        label5 = tk.Label(self.root, text="Mutation/Inversion Probability:")
        label5.pack()
        entry5 = tk.Entry(self.root, textvariable=self.mutation_prob_var, width=50, bg='dark gray')
        entry5.pack()

        label6 = tk.Label(self.root, text="Elite Percentage:")
        label6.pack()
        entry6 = tk.Entry(self.root, textvariable=self.elite_percentage_var, width=50, bg='dark gray')
        entry6.pack()

        label7 = tk.Label(self.root, text="Selection Method:")
        label7.pack()
        dropdown = ttk.Combobox(self.root, textvariable=self.selection_method_var,
                                values=["best", "roulette_wheel", "tournament"], width=47)
        dropdown.pack()

        label8 = tk.Label(self.root, text="Crossover Type:")
        label8.pack()
        dropdown2 = ttk.Combobox(self.root, textvariable=self.crossover_type_var,
                                 values=["single_point", "two_point", "three_point", "uniform", "grainy", "dominance",
                                         "Random Respectful", "DIS", "adaption weighted"], width=47)
        dropdown2.pack()

        label9 = tk.Label(self.root, text="Precision:")
        label9.pack()
        entry7 = tk.Entry(self.root, textvariable=self.precision_var, width=50, bg='dark gray')
        entry7.pack()

        label10 = tk.Label(self.root, text="Mutate Method:")
        label10.pack()
        dropdown3 = ttk.Combobox(self.root, textvariable=self.mutate_type_var,
                                 values=["boundary", "single point", "two point"], width=47)
        dropdown3.pack()

        label11 = tk.Label(self.root, text="Objective function:")
        label11.pack()
        dropdown4 = ttk.Combobox(self.root, textvariable=self.objective_function_var,
                                 values=["keane_function", "hgbat_function", "test"], width=47)
        dropdown4.pack()

        tk.Label(self.root, text="").pack()

        button = tk.Button(self.root, text="Run Algorithm", command=self.run_algorithm, width=42)
        button.pack(pady=10)

        self.time_label = tk.Label(self.root, text="")
        self.time_label.pack()

        self.best_solution_text = tk.Text(self.root, height=5, width=38, bg='dark gray')
        self.best_solution_text.pack()

        self.root.geometry("340x660")

    def run_algorithm(self):
        start_time = time.time()
        min_ss = self.min_ss_var.get()
        max_ss = self.max_ss_var.get()
        objective_function_var = self.objective_function_var.get()
        if(objective_function_var == "hgbat_function"):
            objective_function = hgbat_function
        elif(objective_function_var == "keane_function"):
            objective_function = keane_function
        elif (objective_function_var == "test"):
            objective_function = tst_function
        num_variables = self.num_variables_var.get()
        population_size = self.population_size_var.get()
        num_epochs = self.num_epochs_var.get()
        crossover_prob = self.crossover_prob_var.get()
        mutation_prob = self.mutation_prob_var.get()
        elite_percentage = self.elite_percentage_var.get()
        selection_method = self.selection_method_var.get()
        crossover_type = self.crossover_type_var.get()
        precision = self.precision_var.get()
        mutate = self.mutate_type_var.get()

        num_selected_as_elite = int(population_size * elite_percentage)

        ga = GeneticAlgorithm(min_ss, max_ss, objective_function, num_variables, population_size, num_epochs,
                          crossover_prob, mutation_prob, elite_percentage, num_selected_as_elite, selection_method, crossover_type=crossover_type, precision=precision, mutation_method=mutate)
    
        
        best_values, decimal_best_solution, average_values, std_dev_values, execution_time = ga.evolve()

        np.savetxt("best_values.txt", self.prepend_index_to_values(best_values), fmt="%s")
        np.savetxt("average_values.txt", self.prepend_index_to_values(average_values), fmt="%s")
        np.savetxt("std_dev_values.txt", self.prepend_index_to_values(std_dev_values), fmt="%s")
        np.savetxt("decimal_best_solution.txt", self.prepend_index_to_values(decimal_best_solution), fmt="%s")

        print(f"Best Values: {best_values}")
        print(f"Average Values: {average_values}")
        print(f"Standard Deviation Values: {std_dev_values}")
        print(f"Execution Time: {execution_time} seconds")

        best_solution_text = "Best Values:\n"
        for i, val in enumerate(decimal_best_solution):
            best_solution_text += f"Variable {i+1}: {val}\n"

        self.best_solution_text.delete('1.0', tk.END)
        self.best_solution_text.insert(tk.END, best_solution_text)

        self.display_time(execution_time)
        self.plot_graphs(best_values, average_values, std_dev_values)

    def prepend_index_to_values(self, arr):
        modified_arr = np.empty_like(arr, dtype=object)
        for i in range(len(arr)):
            modified_arr[i] = f"{i+1}: {arr[i]}"
        return modified_arr

    def plot_graphs(self, best_values, average_values, std_dev_values):
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(best_values, label='Best Values')
        plt.plot(average_values, label='Average Values')
        plt.xlabel('Epoch')
        plt.ylabel('Function Value')
        plt.title('Function Values vs. Epoch')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(std_dev_values, label='Standard Deviation')
        plt.xlabel('Epoch')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation vs. Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def display_time(self, execution_time):
        self.time_label.config(text=f"Execution Time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    #func = bf.Keane()
    #print(func._evaluate([1.3932, 0.    ]))
    gui = GeneticAlgorithmGUI()
    gui.root.mainloop()