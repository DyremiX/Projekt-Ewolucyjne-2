import random


def f1_PAX(parent1, parent2):
    if len(parent1) != len(parent2):
        raise ValueError("Parents must have the same length")

    size = len(parent1)

    cp = random.randint(0, (size - 1))

    child1 = parent1[:]
    child2 = parent2[:]

    child1[cp] = (parent1[cp] + parent2[cp]) / 2
    child2[cp] = (parent1[cp] + parent2[cp]) / 2

    return child1, child2


parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
parent2 = [5, 8, 1, 6, 3, 9, 4, 7]

child1, child2 = f1_PAX(parent1, parent2)

print("First child:", child1)
print("Second child:", child2)
