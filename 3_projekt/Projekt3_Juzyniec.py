import random
from math import sqrt

# Algorytm I
def SX_version1(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")
    
    n = len(parent1)
    child = []
    for i in range(n):
        child.append(sqrt((parent1[i] ** 2 + parent2[i] ** 2) / 2))
    
    return child

# Algorytm II
def SX_version2(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")
    
    n = len(parent1)
    alpha = random.uniform(0, 1)
    
    child = []
    for i in range(n):
        child.append(sqrt(alpha * parent1[i] ** 2 + (1 - alpha) * parent2[i] ** 2))
    
    return child

# Algorytm III
def SX_version3(parents):
    lengths = [len(parent) for parent in parents]
    if len(set(lengths)) != 1:
        raise ValueError("Parents must have the same length")
        
    k = len(parents)
    n = len(parents[0])
    
    alphas = [random.uniform(0, 1) for _ in range(k)]
    alpha_sum = sum(alphas)
    alphas = [alpha / alpha_sum for alpha in alphas]
    
    
    child = []
    for i in range(n):
        child.append(sqrt(sum(alphas[j] * parents[j][i] ** 2 for j in range(k))))
    
    return child

parent1 = [1, 2, 3, 4]
parent2 = [4, 3, 2, 1]
parents = [[1, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]]

print("Algorytm I:", SX_version1(parent1, parent2))
print("Algorytm II:", SX_version2(parent1, parent2))
print("Algorytm III:", SX_version3(parents))