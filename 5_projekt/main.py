#na podstawie przykładu: https://pypi.org/project/pygad/1.0.18/
import logging
import pygad
import numpy
import benchmark_functions as bf
from benchmark_functions import Keane
from opfunu.utils.operator import hgbat_func
import numpy as np
from CustomGA import CustomGA_binary
from CustomGA import CustomGA_real
import matplotlib.pyplot as plt

#Konfiguracja algorytmu genetycznego

num_genes = 2
def tst_function(x):
    x_decimal = np.array(x)
    result = np.sum(np.square(x_decimal)) + 5
    return result

func = tst_function #Keane(n_dimensions=num_genes), hgbat_func, tst_function
def fitness_func(ga_instance, solution, solution_idx):
    fitness = func(solution)
    return 1./fitness

fitness_function = fitness_func
num_generations = 100
sol_per_pop = 80
num_parents_mating = 50
#boundary = func.suggested_bounds() #możemy wziąć stąd zakresy

#dla binarnej
init_range_low = 0
init_range_high = 2
gene_type = int

# dla rzeczywistej - Keane
# init_range_low = boundary[0]
# init_range_high = boundary[1]
# gene_type = float

# dla rzeczywistej - hgbat?(todo czy da sie wyciagnac boundary?) i test
# init_range_low = -10
# init_range_high = 10
# gene_type = float

mutation_num_genes = 1
parent_selection_type = "tournament" #tournament, rws (ruletka), random

#binarne: uniform, single_point, two_points, three_point, grainy, RRC, crossover_by_dominance, DIS, adaption_weighted_cross
#rzeczywiste: arithmetic, linear, alpha_mixed, alpha_beta_mixed, average, crossover_HX, SX_version1, SX_version2, f1_PAX, fitness_weighted_cross_for_real_numbers
crossover_type = "single_point"

#random, swap
#rzeczywste: gauss
mutation_type = "random" 

#Konfiguracja logowania

level = logging.DEBUG
name = 'logfile.txt'
logger = logging.getLogger(name)
logger.setLevel(level)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

def on_generation(ga_instance):
    ga_instance.logger.info("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    ga_instance.logger.info("Best    = {fitness}".format(fitness=1./solution_fitness))
    ga_instance.logger.info("Individual    = {solution}".format(solution=repr(solution)))

    tmp = [1./x for x in ga_instance.last_generation_fitness] #ponownie odwrotność by zrobić sobie dobre statystyki

    ga_instance.logger.info("Min    = {min}".format(min=numpy.min(tmp)))
    ga_instance.logger.info("Max    = {max}".format(max=numpy.max(tmp)))
    ga_instance.logger.info("Average    = {average}".format(average=numpy.average(tmp)))
    ga_instance.logger.info("Std    = {std}".format(std=numpy.std(tmp)))
    ga_instance.logger.info("\r\n")

# def fitnessFunction(individual): 
#     ind = decodeInd(individual) 
#     # result = (ind[0] + 2* ind[1] - 7)**2 + (2* ind[0] + ind[1] -5)**2 
#     # return result
    
#     #return keane_function(ind)
#     #return hgbat_func(ind)
#     return tst_function(ind)

def decodeInd(individual, num_parts):
    part_length = len(individual) // num_parts
    
    decoded_values = []

    for i in range(num_parts):
        binary_part = individual[i * part_length: (i + 1) * part_length]
        
        value = int(binary_part, 2)
        
        #normalizacja do jakiegos zakresu, np [0, 10] ale nie wiem czy trzeba?
        max_value = 2**part_length - 1
        normalized_value = value / max_value * 10
        
        decoded_values.append(normalized_value)

        #decoded_values.append(value)
    return decoded_values

#Właściwy algorytm genetyczny
#pygad.GA - oryginalne
#CustomGA_binary
#CustomGA_real
ga_instance = CustomGA_binary(num_generations=num_generations,
          sol_per_pop=sol_per_pop,
          num_parents_mating=num_parents_mating,
          num_genes=num_genes,
          fitness_func=fitness_func,
          init_range_low=init_range_low,
          init_range_high=init_range_high,
          mutation_num_genes=mutation_num_genes,
          parent_selection_type=parent_selection_type,
          crossover_type=crossover_type,
          mutation_type=mutation_type,
          keep_elitism= 1,
          K_tournament=3,
          random_mutation_max_val=32.768,
          random_mutation_min_val=-32.768,
          logger=logger,
          on_generation=on_generation,
          parallel_processing=['thread', 4])

ga_instance.run()


best = ga_instance.best_solution()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=1./solution_fitness))


# sztuczka: odwracamy my narysował nam się oczekiwany wykres dla problemu minimalizacji
ga_instance.best_solutions_fitness = [1. / x for x in ga_instance.best_solutions_fitness]
ga_instance.plot_fitness()

generations = list(range(0, ga_instance.generations_completed + 1))
best_fitness = [fit for fit in ga_instance.best_solutions_fitness]
average_fitness = [numpy.mean([x for x in ga_instance.best_solutions_fitness[:i+1]]) for i in range(len(ga_instance.best_solutions_fitness))]
std_fitness = [numpy.std([x for x in ga_instance.best_solutions_fitness[:i+1]]) for i in range(len(ga_instance.best_solutions_fitness))]

plt.figure(figsize=(10, 5))
plt.plot(generations, best_fitness, label='Best Fitness')
plt.plot(generations, average_fitness, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title('Best and Average Fitness Over Generations')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(generations, std_fitness, label='Standard Deviation')
plt.xlabel('Generation')
plt.ylabel('Standard Deviation')
plt.legend()
plt.title('Standard Deviation of Fitness Over Generations')
plt.show()