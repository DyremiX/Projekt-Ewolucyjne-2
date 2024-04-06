import numpy as np


def DIS(ind1, ind2, q, size):
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


size = 10
ind1 = np.random.randint(0, 2, size)
ind2 = np.random.randint(0, 2, size)
print("ind1:", ind1)
print("ind2:", ind2)

q = np.random.randint(1, size)
print("q:", q)

new_ind = DIS(ind1, ind2, q, size)
print("new_ind:", np.array(new_ind))
