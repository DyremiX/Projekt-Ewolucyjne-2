import tkinter as tk
from tkinter import ttk
import numpy as np
import time as time
from genetic_algorithm import GeneticAlgorithm
from objective_functions import keane_function, tst_function, hgbat_function
import matplotlib.pyplot as plt

class GeneticAlgorithmGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Genetic Algorithm")
        self.root.geometry("600x650")

        self.num_variables_var = tk.IntVar(value=2)
        self.max_ss_var = tk.DoubleVar(value=10.0)
        self.min_ss_var = tk.DoubleVar(value=0.0)
        self.population_size_var = tk.IntVar(value=100)
        self.num_epochs_var = tk.IntVar(value=100)
        self.crossover_prob_var = tk.DoubleVar(value=0.8)
        self.mutation_prob_var = tk.DoubleVar(value=0.1)
        self.elite_percentage_var = tk.DoubleVar(value=0.6)
        self.selection_method_var = tk.StringVar(value="tournament")
        self.crossover_type_var = tk.StringVar(value="arithmetic")
        self.precision_var = tk.IntVar(value=4) #TODO chyba mozna wywalic??
        self.mutate_type_var = tk.StringVar(value="uniform")
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
                                 values=["arithmetic", "linear", "alpha_mixed", "alpha_beta_mixed", "average", "crossover_HX",
                                         "SX", "f1_PAX", "adaption weighted"], width=47)
        dropdown2.pack()

        label9 = tk.Label(self.root, text="Precision:")
        label9.pack()
        entry7 = tk.Entry(self.root, textvariable=self.precision_var, width=50, bg='dark gray')
        entry7.pack()

        label10 = tk.Label(self.root, text="Mutate Method:")
        label10.pack()
        dropdown3 = ttk.Combobox(self.root, textvariable=self.mutate_type_var,
                                 values=["uniform", "gaussian"], width=47)
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
    
        
        best_values, best_solution, average_values, std_dev_values, execution_time = ga.evolve()

        np.savetxt("best_values.txt", self.prepend_index_to_values(best_values), fmt="%s")
        np.savetxt("average_values.txt", self.prepend_index_to_values(average_values), fmt="%s")
        np.savetxt("std_dev_values.txt", self.prepend_index_to_values(std_dev_values), fmt="%s")
        np.savetxt("best_solution.txt", self.prepend_index_to_values(best_solution), fmt="%s")

        print(f"Best Values: {best_values}")
        print(f"Average Values: {average_values}")
        print(f"Standard Deviation Values: {std_dev_values}")
        print(f"Execution Time: {execution_time} seconds")

        best_solution_text = "Best Value: " + str(best_values[-1]) + '\n'
        for i, val in enumerate(best_solution):
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