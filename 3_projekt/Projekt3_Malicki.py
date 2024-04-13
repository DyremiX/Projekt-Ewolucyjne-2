import random
import numpy as np
class adaption_weighted_cross_for_real_numbers:

    def __init__(self, min_ss, max_ss, pop_num, var_num):
        self.pop = self.initialize_pop(min_ss, max_ss, pop_num, var_num)

    def initialize_pop(self, min_ss, max_ss, pop_num, var_num):
        random_array = np.random.uniform(low=min_ss, high=max_ss, size=(pop_num, var_num))
        return random_array

    def cross(self):
        num_vars = self.pop.shape[1]
        pop_size = self.pop.shape[0]
        alfa = random.uniform(0, 1)

        parent_num = 0
        parent_tab = np.zeros((0, num_vars), dtype=float)
        for i in range(0, pop_size):
            beta = (self.adap_func(self.pop[i]) - self.min_adap_func(self.pop, pop_size)) / (
                        self.max_adap_func(self.pop, pop_size) - self.min_adap_func(self.pop, pop_size))
            if beta < alfa:
                pop_i_reshaped = self.pop[i].reshape(1, *self.pop[i].shape)
                parent_tab = np.vstack([parent_tab, pop_i_reshaped])
                parent_num = parent_num + 1

        W = np.zeros((1, parent_num), dtype=float)

        denominator = 0
        for i in range(0, parent_num):
            denominator = denominator + self.adap_func(parent_tab[i])

        for i in range(0, parent_num):
            W[0][i] = self.adap_func(parent_tab[i]) / float(denominator)

        f_desc = np.zeros((1, num_vars), dtype=float)
        for v in range(0, num_vars):
            meter = 0
            denominator = 0
            for p in range(0, parent_num):
                meter = meter + W[0][p]*parent_tab[p][v]
                denominator = denominator + W[0][p]
            f_desc[0][v] = meter/denominator

        return f_desc

    def adap_func(self, single_element):
        result = 0
        for i in range(0, single_element.shape[0]):
            result = result + single_element[i]**2
        result = result+5
        return result

    def min_adap_func(self, pop, pop_size):
        minim = self.adap_func(pop[0])
        for i in range(0, pop_size):
            if self.adap_func(pop[i]) < minim:
                minim = self.adap_func(pop[i])
        return minim

    def max_adap_func(self, pop, pop_size):
        maxim = self.adap_func(pop[0])
        for i in range(0, pop_size):
            if self.adap_func(pop[i]) > maxim:
                maxim = self.adap_func(pop[i])
        return maxim


fwx = adaption_weighted_cross_for_real_numbers(0.0, 100.0, 10, 2)
print(fwx.pop)
print(fwx.cross())


