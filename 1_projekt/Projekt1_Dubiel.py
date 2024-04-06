def crossover_by_dominance(parent_A, parent_B, mask_A, mask_B):
    child_C = parent_A[:]
    child_D = parent_B[:]
    
    for i in range(len(parent_A)):
        if mask_B[i] == 1 and mask_A[i] == 0:  # dominacja B z uwagi na maskę
            child_C[i] = parent_B[i]
        if mask_B[i] == 0 and mask_A[i] == 1:  # dominacja A z uwagi na maskę
            child_D[i] = parent_A[i]
    
    return child_C, child_D

print("Krzyżowanie przez dominacje")
print("============================")

# Przykładowe dane
A = [0, 1, 0, 1, 1, 0, 1, 1]
B = [1, 1, 0, 0, 1, 1, 1, 0]
print("Rodzice")
print("------------------------------------------------------")
print("Rodzic A :", A)
print("Rodzic B :", B)
print("\n")

A_mask = [1, 1, 1, 1, 0, 0, 0, 0]
B_mask = [0, 0, 0, 0, 1, 1, 1, 1]
print("Maski rodziców")
print("------------------------------------------------------")
print("Maska rodzica A :", A_mask)
print("Maska rodzica B :", B_mask)
print("\n")

# Wywołanie funkcji krzyżowania
child_C, child_D = crossover_by_dominance(A, B, A_mask, B_mask)



# Wyświetlenie wyników
print("Po krzyżowaniu")
print("------------------------------------------------------")
print("Potomek C :", child_C)
print("Potomek D :", child_D)
print("\n")
