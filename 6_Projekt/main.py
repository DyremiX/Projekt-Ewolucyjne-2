import numpy as np
from mealpy.physics_based.CDO import OriginalCDO
from mealpy import FloatVar
import benchmark_functions as bf
from opfunu.cec_based import cec2014

# Define the Keane function using the benchmark_functions library
n_dimensions1 = 2
func_1_test = bf.Keane(n_dimensions=n_dimensions1)

# Define the problem bounds and other details for Keane function
bounds1 = FloatVar(lb=[-10]*n_dimensions1, ub=[10]*n_dimensions1, name="dimension1")

# Define the problem dictionary for Keane function
problem1 = {
    "bounds": bounds1,
    "minmax": "min",   # We are minimizing the Keane function
    "obj_func": func_1_test,
    "verbose": True,
}

# Define the CEC2014 F14 function
n_dimensions2 = 10
func_2_test = cec2014.F142014(ndim=n_dimensions2)

# Define a wrapper to make the function callable
def wrapper(solution):
    return func_2_test.evaluate(solution)

# Define the problem bounds and other details for F14 function
bounds2 = FloatVar(lb=[-100]*n_dimensions2, ub=[100]*n_dimensions2, name="dimension2")

# Define the problem dictionary for F14 function
problem2 = {
    "bounds": bounds2,
    "minmax": "min",   # We are minimizing the F14 function
    "obj_func": wrapper,
    "verbose": True,
}

# Parameters of the CDO algorithm
epoch = 1000          # Number of iterations
pop_size = 50         # Population size

# Create the optimizer for Keane function
model1 = OriginalCDO(epoch=epoch, pop_size=pop_size)

# Train the optimizer to find the minimum of the Keane function
g_best1 = model1.solve(problem1)

print(f"Best position Keane: {g_best1.solution}")
print(f"Best fitness Keane: {g_best1.target.fitness}")

# Create the optimizer for F14 function
model2 = OriginalCDO(epoch=epoch, pop_size=pop_size)

# Train the optimizer to find the minimum of the F14 function
g_best2 = model2.solve(problem2)

print(f"Best position F14: {g_best2.solution}")
print(f"Best fitness F14: {g_best2.target.fitness}")