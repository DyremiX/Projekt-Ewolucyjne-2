import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

class GeneticAlgorithm:
    def __init__(self, objective_function, num_variables, population_size=100, num_epochs=100, 
                 crossover_prob=0.8, mutation_prob=0.1, elite_percentage=0.6, num_selected=2, selection_method="tournament", crossover_type="single_point", grain_size=2, precision=2,
                 mutation_method="single point"):
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

    def initialize_population(self):
        num_bits = int(np.ceil(np.log2(10**self.precision)))
        return np.random.randint(2, size=(self.population_size, self.num_variables, num_bits))

    def binary_to_decimal(self, binary_population, precision):
        if binary_population.ndim == 3:  # Dla całej populacji
            population_size, num_variables, num_bits = binary_population.shape
            decimal_population = np.zeros((population_size, num_variables))

            for i in range(population_size):
                for v in range(num_variables):
                    decimal_value = 0
                    for j in range(num_bits):
                        decimal_value += binary_population[i][v][j] * (2 ** (num_bits - j - 1))
                    decimal_population[i][v] = decimal_value / (10 ** precision)
        elif binary_population.ndim == 2:  # Dla pojedynczego najlepszego rozwiązania
            num_variables, num_bits = binary_population.shape
            decimal_solution = np.zeros(num_variables)

            for v in range(num_variables):
                decimal_value = 0
                for j in range(num_bits):
                    decimal_value += binary_population[v][j] * (2 ** (num_bits - j - 1))
                decimal_solution[v] = decimal_value / (10 ** precision)

            decimal_population = decimal_solution
        else:
            raise ValueError("Niepoprawny wymiar tablicy binarnej.")

        return decimal_population

    def evaluate_population(self):
        decimal_population = self.binary_to_decimal(self.population, self.precision)
        return np.array([self.objective_function(individual) for individual in decimal_population])

    def select_parents(self, fitness_values):
        if self.selection_method == "best":
            return self.select_best_parents(fitness_values)
        elif self.selection_method == "roulette_wheel":
            return self.select_roulette_wheel_parents(fitness_values)
        elif self.selection_method == "tournament":
            return self.select_tournament_parents(fitness_values)

#W: select_best_parents jedynie sortuje oryginalną populacje, natomiast pozostałe select'y zmnieniają ją już na tym etapie
    def select_best_parents(self, fitness_values):
        sorted_indices = np.argsort(fitness_values)
        return self.population[sorted_indices[-self.population_size:]]

# W: select_roulette_wheel_parents zakłada, że największe wartości FF są najbardziej pożądane
    def select_roulette_wheel_parents(self, fitness_values):
        fitness_values = fitness_values - np.min(fitness_values) + 1
        total_fitness = np.sum(fitness_values)
        probabilities = fitness_values / total_fitness
        selected_indices = np.random.choice(range(len(fitness_values)), size=self.population_size, replace=True, p=probabilities)
        return self.population[selected_indices]

# W: natomiast select_tournament_parents zakłada, że najmniejsze wartości FF są najbardziej pożądane. Jeszcze nie wiem na ile to jest problem, że mamy taką różnicę, ale tak pisze na przyszłość. Popatrze na to jeszcze.
    def select_tournament_parents(self, fitness_values):
        selected_indices = []
        for _ in range(self.population_size):
            tournament_indices = np.random.choice(len(fitness_values), size=self.num_selected, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            winner_index = tournament_indices[np.argmin(tournament_fitness)]
            selected_indices.append(winner_index)
        return self.population[selected_indices]

    def single_point_crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.num_variables - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def two_point_crossover(self, parent1, parent2):
        crossover_points = sorted(random.sample(range(1, self.num_variables), 2))
        child1 = np.concatenate((parent1[:crossover_points[0]], parent2[crossover_points[0]:crossover_points[1]], parent1[crossover_points[1]:]))
        child2 = np.concatenate((parent2[:crossover_points[0]], parent1[crossover_points[0]:crossover_points[1]], parent2[crossover_points[1]:]))
        return child1, child2

# W: dodałem bo prosił w poleceniu
    def three_point_crossover(self, parent1, parent2):
        crossover_points = sorted(random.sample(range(1, self.num_variables), 3))
        child1 = np.concatenate((parent1[:crossover_points[0]],
                                 parent2[crossover_points[0]:crossover_points[1]],
                                 parent1[crossover_points[1]:crossover_points[2]],
                                 parent2[crossover_points[2]:]))
        child2 = np.concatenate((parent2[:crossover_points[0]],
                                 parent1[crossover_points[0]:crossover_points[1]],
                                 parent2[crossover_points[1]:crossover_points[2]],
                                 parent1[crossover_points[2]:]))
        return child1, child2

    def uniform_crossover(self, parent1, parent2):
        mask_shape = parent1.shape
        mask = np.random.randint(2, size=mask_shape)
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    # W: z tego co rozumiem, to grainy_crossover tworzy 1 potomka, ale tutaj proponuje implementacje dla 2
    def grainy_crossover(self, parent1, parent2): #TODO sprawdzic czy to git implementacja / W: zmieniłem na to co jest w książce Gwiazdy (tylko z tą zmianą ↑)
        child1, child2 = parent1.copy(), parent2.copy()
        for i in range(0, len(parent1), self.grain_size):
            if random.random() <= 0.5:
                child1[i]=parent1[i]
                child2[i]=parent2[i]
            else:
                child1[i]=parent2[i]
                child2[i]=parent1[i]
        return child1, child2

    def crossover_by_dominance(self, parent_A, parent_B, mask_A, mask_B):
        child_C = parent_A[:]
        child_D = parent_B[:]
        
        for i in range(len(parent_A)):
            if mask_B[i] == 1 and mask_A[i] == 0:  # dominacja B z uwagi na maskę
                child_C[i] = parent_B[i]
            if mask_B[i] == 0 and mask_A[i] == 1:  # dominacja A z uwagi na maskę
                child_D[i] = parent_A[i]
    
        return child_C, child_D

    def RRC(self, A, B, n):
        S = np.where(A == B, A, None)

        C = np.where(S != None, S, np.where(np.random.uniform(0, 1, size=(len(A), len(A[0]))) <= 0.5, 1, 0))
        D = np.where(S != None, S, np.where(np.random.uniform(0, 1, size=(len(A), len(A[0]))) <= 0.5, 1, 0))

        return D, C

    def DIS(self, ind1, ind2, q, size):
        new_ind = []
        for i in range(q):
            if ind1[i] != ind2[i]:
                new_ind.append(ind1[i])
            else:
                new_ind.append(np.random.randint(0, 2))
        for i in range(q, size):
            if ind1[i] != ind2[i]:
                new_ind.append(ind2[i])
            else:
                new_ind.append(np.random.randint(0, 2))

        return new_ind
    
    def adaption_weighted_cross(self, pop):
        n = pop[0].size
        pop_size = pop.size
        print("\n\n")
        print(pop[0])
        print(pop[0][0])
        alfa = random.uniform(0, 1)
        print(alfa)

        parent_num = 0
        parent_tab = np.zeros((0, n), dtype=int)
        num_rows, num_columns = parent_tab.shape
        print("Number of parent_tab rows:", num_rows)
        print("Number of parent_tab columns:", num_columns)
        for i in range(0,pop_size):
            beta=(self.adap_func(pop[i])-self.min_adap_func(pop,pop_size))/(self.max_adap_func(pop,pop_size)-self.min_adap_func(pop,pop_size))
            if (beta<alfa):
                parent_tab = np.vstack([parent_tab, pop[i]])
                parent_num = parent_num + 1

        print(parent_tab)

        print("parent_num: ",parent_num)
        W= np.zeros((1, parent_num), dtype=float)

        denominator = 0
        for i in range(0, parent_num):
            denominator = denominator + self.adap_func(parent_tab[i])

        for i in range(0,parent_num):
            W[0][i]=self.adap_func(parent_tab[i])/float(denominator)


        print("W: ",W)
        tab = np.zeros((parent_num, n), dtype=int)
        for j in range(0,parent_num):
            for i in range(0,n):
                if parent_tab[j][i]==1:
                    tab[j][i]=1
                else:
                    tab[j][i]=-1

        print("\nTAB:")
        print(tab)

        f_desc = np.zeros((1, n), dtype=int)
        print("calculating descendant: ")
        for i in range(0,n):
            lambd = self.calc_sign(W,tab,i,parent_num)
            if lambd>=0:
                f_desc[0][i] = 1
            else:
                f_desc[0][i] = 0

        print("Descendant: ",f_desc)
        return f_desc

    def gen_pop(self, pop_size,bit_length):
        table_size = (pop_size, bit_length)

        random_table = np.random.randint(2, size=table_size)
        return np.array(random_table)

    def calc_sign(self, W,t,gen_num,parent_num):
        calc = 0
        for j in range(0,parent_num):
            calc = calc+W[0][j]*t[j][gen_num]
        print(calc)
        if calc>0:
            return 1
        elif calc<0:
            return -1
        else:
            return 0

    def binary_row_to_decimal(self, binary_row):
        powers_of_2 = 2 ** np.arange(len(binary_row))[::-1]
        return np.sum(binary_row * powers_of_2)

    def adap_func(self, sub):
        val = self.binary_row_to_decimal(sub)
        self.objective_function(val)

    def min_adap_func(self, pop,pop_size):
        minim=self.adap_func(pop[0])
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
        for i in range(0, len(pop)):
            if self.adap_func(pop[i]) > maxim:
                maxim_id = i
                maxim = self.adap_func(pop[i])
        return maxim_id

    def crossover(self, parents):
        children = []
        children_fwx = []
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i+1]
            if random.random() < self.crossover_prob:
                if self.crossover_type == "single_point":
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                elif self.crossover_type == "two_point":
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                elif self.crossover_type == "three_point":
                    child1, child2 = self.two_point_crossover(parent1, parent2)
                elif self.crossover_type == "uniform":
                    child1, child2 = self.uniform_crossover(parent1, parent2)
                elif self.crossover_type == "grainy":
                    child1, child2 = self.uniform_crossover(parent1, parent2)  
                elif self.crossover_type == "dominance":
                    mask_A = np.random.randint(2, len(parent1)) #TODO zmienic?? nie wiem czy ma sie losowac
                    mask_B = np.random.randint(2, len(parent1)) #TODO zmienic?? nie wiem czy ma sie losowac
                    child1, child2 = self.crossover_by_dominance(parent1, parent2, mask_A, mask_B) #TODO nie dziala :(
                elif self.crossover_type == "Random Respectful":
                    child1, child2 = self.RRC(parent1, parent2, len(parent1))
                elif self.crossover_type == "DIS":
                    q = np.random.randint(1, len(parent1)) #TODO zmienic?? nie wiem czy ma sie losowac
                    child1 = self.DIS(parent1, parent2, q, len(parent1)) #TODO nie dziala :(
                    child2 = self.DIS(parent1, parent2, q, len(parent1)) #TODO zmienic??
                elif self.crossover_type == "adaption weighted":
              #TODO pomyslec jak dodac populacje / W: adaption_weighted_cross tworzy w jednym odpaleniu jednego potomka z x rodziców, więc proponuje utworzyć n potomków i zastąpić nimi najgorsze n rodziców i w ten sposób uzyskać populacje tej samej liczności po danej iteracji
              #TODO nie wiem tylko jak to ma działać dla tablic 3d, wiec WIP
                    children_fwx.append(self.adaption_weighted_cross(parents))
                else:
                    child1, child2 = self.single_point_crossover(parent1, parent2)

                if self.crossover_type != "adaption weighted":
                    children.extend([child1, child2])
            else:
                children.extend([parent1, parent2])

        # W: dodałem, żeby obsłużyć adaption_weighted_cross
        if self.crossover_type == "adaption weighted":
            for _ in range(0,len(children_fwx)):
                parents.pop(self.max_ff_parent(parents))
            children = parents.extend(children_fwx)

        if len(parents) % 2 != 0:
            children.append(parents[-1])

        return np.array(children)


    def boundary_mutation(self, population):
        mutated_population = population.copy()
        for i in range(len(mutated_population)):
            for j in range(len(mutated_population[i])):
                if random.random() < self.mutation_prob:
                    mutated_population[i][j] = random.randint(0, 1)
        return mutated_population

    def single_point_mutation(self, population):
        mutated_population = population.copy()
        for i in range(len(mutated_population)):
            if random.random() < self.mutation_prob:
                mutation_point = random.randint(0, len(mutated_population[i]) - 1)
                mutated_population[i][mutation_point] = 1 - mutated_population[i][mutation_point]
        return mutated_population

    def two_point_mutation(self, population):
        mutated_population = population.copy()
        for i in range(len(mutated_population)):
            if random.random() < self.mutation_prob:
                mutation_points = sorted(random.sample(range(len(mutated_population[i])), 2))
                mutated_population[i][mutation_points[0]:mutation_points[1]] = 1 - mutated_population[i][mutation_points[0]:mutation_points[1]]
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

        num_missing_elite = self.population_size - elite_size
        if num_missing_elite > 0:
            random_population_indices = np.random.choice(len(population), num_missing_elite, replace=False)
            random_population = population[random_population_indices]
            elite_population = np.vstack((elite_population, random_population))

        return elite_population


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

            parents = self.select_parents(fitness_values)
            children = self.crossover(parents)
            mutated_children = self.mutate(children)
            combined_population = np.vstack((self.population, mutated_children))
            self.population = combined_population 
            elite_population = self.elitism(self.population, fitness_values)
            self.population = elite_population

        end_time = time.time()
        execution_time = end_time - start_time

        return best_values, decimal_best_solution, average_values, std_dev_values, execution_time



def keane_function(x): #TODO zmienic
    x_decimal = np.array(x)
    return np.sum(np.sin(x_decimal)**2 - np.exp(-0.1*x_decimal**2))

def hgbat_function(x): #TODO zmienic
    x_decimal = np.array(x)
    return abs(np.sum(x_decimal**2)**2-np.sum(x_decimal)**2)**(1/2)+(0.5*np.sum(x_decimal**2)-np.sum(x_decimal))/len(x_decimal)+0.5

class GeneticAlgorithmGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Genetic Algorithm")
        self.root.geometry("600x650")

        self.num_variables_var = tk.IntVar(value=5)
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

        label5 = tk.Label(self.root, text="Mutation Probability:")
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
                                 values=["keane_function", "hgbat_function"], width=47)
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
        objective_function_var = self.objective_function_var.get()
        if(objective_function_var == "hgbat_function"):
            objective_function = hgbat_function
        elif(objective_function_var == "keane_function"):
            objective_function = keane_function
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

        num_selected = int(population_size * elite_percentage)

        ga = GeneticAlgorithm(objective_function, num_variables, population_size, num_epochs, 
                          crossover_prob, mutation_prob, elite_percentage, num_selected, selection_method, crossover_type=crossover_type, precision=precision, mutation_method=mutate)
    
        
        best_values, decimal_best_solution, average_values, std_dev_values, execution_time = ga.evolve()

        # W: Dodałem zapis do pliku
        np.savetxt("best_values.txt", self.prepend_index_to_values(best_values), fmt="%s")
        np.savetxt("average_values.txt", self.prepend_index_to_values(average_values), fmt="%s")
        np.savetxt("std_dev_values.txt", self.prepend_index_to_values(std_dev_values), fmt="%s")

        print(f"Best Values: {best_values}")
        print(f"Average Values: {average_values}")
        print(f"Standard Deviation Values: {std_dev_values}")
        print(f"Execution Time: {execution_time} seconds")

        best_solution_text = "Best Values:\n"
        for i, val in enumerate(decimal_best_solution):
            best_solution_text += f"Variable {i+1}: {val}\n"

        self.best_solution_text.delete('1.0', tk.END)
        self.best_solution_text.insert(tk.END, best_solution_text)

        self.display_time(execution_time) #TODO czemu wyswietla sie po zamknieciu wykresow? / W: u mnie pojawia się wraz z wynikami
        self.plot_graphs(best_values, average_values, std_dev_values)
        #TODO czemu okienko sie zmienjsza po kliknieciu run? / W: już nie (po dodaniu self.root.geometry("340x660")), albo u mnie już nie ¯\_(ツ)_/¯

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
    gui = GeneticAlgorithmGUI()
    gui.root.mainloop()