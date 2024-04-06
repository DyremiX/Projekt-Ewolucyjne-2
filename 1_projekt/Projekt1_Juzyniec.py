import random
import numpy as np

def RRC (A, B, n):
    S = [A[i] if A[i] == B[i] else None for i in range(n)]

    C = []
    D = []

    for i in range(n):
        si = S[i]

        if si is not None:
            C.append(si)
            D.append(si)
        elif si is None:
            u = random.uniform(0, 1)

            if u <= 0.5:
                C.append(1)
            else:
                C.append(0)

            u = random.uniform(0, 1)

            if u <= 0.5:
                D.append(1)
            else:
                D.append(0)

    return D,C

n=10

A = np.random.randint(0,2,n)
B = np.random.randint(0,2,n)

D,C = RRC(A, B, n)

print( A, "\n", B, "\n", np.array(C), "\n", np.array(D))