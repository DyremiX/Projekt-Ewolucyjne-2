import random
import numpy as np

def adaption_weighted_cross(pop_size,n):
    pop=gen_pop(pop_size,n)
    print(pop)
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
        beta=(adap_func(pop[i])-min_adap_func(pop,pop_size))/(max_adap_func(pop,pop_size)-min_adap_func(pop,pop_size))
        if (beta<alfa):
            parent_tab = np.vstack([parent_tab, pop[i]])
            parent_num = parent_num + 1

    print(parent_tab)

    print("parent_num: ",parent_num)
    W= np.zeros((1, parent_num), dtype=float)

    denominator = 0
    for i in range(0, parent_num):
        denominator = denominator + adap_func(parent_tab[i])

    for i in range(0,parent_num):
        W[0][i]=adap_func(parent_tab[i])/float(denominator)


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
        lambd = calc_sign(W,tab,i,parent_num)
        if lambd>=0:
            f_desc[0][i] = 1
        else:
            f_desc[0][i] = 0

    print("Descendant: ",f_desc)
    pop = np.vstack((pop, f_desc))
    return pop


def gen_pop(pop_size,bit_length):
    table_size = (pop_size, bit_length)

    random_table = np.random.randint(2, size=table_size)
    return np.array(random_table)

def calc_sign(W,t,gen_num,parent_num):
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


def binary_row_to_decimal(binary_row):
    powers_of_2 = 2 ** np.arange(len(binary_row))[::-1]
    return np.sum(binary_row * powers_of_2)

def adap_func(sub):
    #Fitness Function
    val = binary_row_to_decimal(sub)
    return val * val + 5

def min_adap_func(pop,pop_size):
    minim=adap_func(pop[0])
    for i in range(0,pop_size):
        if adap_func(pop[i])<minim:
            minim=adap_func(pop[i])
    return minim

def max_adap_func(pop,pop_size):
    maxim = adap_func(pop[0])
    for i in range(0,pop_size):
        if adap_func(pop[i])>maxim:
            maxim=adap_func(pop[i])
    return maxim


adaption_weighted_cross(8,6)

## Funkcja przystosowania powinna zwracać mniejsze liczby, dla lepiej przystosowanych osobników