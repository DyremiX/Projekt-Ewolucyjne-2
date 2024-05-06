from math import sin, sqrt
import random

from matplotlib import pyplot as plt
import numpy as np

# Example usage:
P1 = [3,3]
P2 = [-2,-2]

def schaffer_f2(x):
    return 0.5 + ((sin(x[0]**2 + x[1]**2))**2 - 0.5) / ((1 + 0.001*(x[0]**2 + x[1]**2))**2)

def crossover_HX (P1, P2):
    C1 = []
    # alpha = random.random()
    alpha = 0.5
    if schaffer_f2(P1) >= schaffer_f2(P2):
        C1.append(alpha*(P2[0] - P1[0]) + P2[0])
        C1.append(alpha*(P2[1] - P1[1]) + P2[1])
    else:
        C1.append(alpha*(P1[0] - P2[0]) + P1[0])
        C1.append(alpha*(P1[1] - P2[1]) + P1[1])
    return C1

print(crossover_HX(P1,P2))