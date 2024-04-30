#import benchmark_functions as bf
#import opfunu as opf
#TODO zmienic na funkcje z tego co on chciał (gotowe biblioteki) i na nich uruchamiac
import numpy as np
from numpy import sin
from numpy import sqrt

def keane_function(x):
    # x_decimal = np.array(x)
    # N = len(x_decimal)
    # sum_cos4 = np.sum(np.cos(x)**4)
    # prod_cos2 = np.prod(np.cos(x)**2)
    # inner_term = np.abs(sum_cos4 - prod_cos2)
    # sum_x_squared = np.sum([x[i]**2 * (i + 1) for i in range(N)])
    # result = -inner_term * sum_x_squared**(-0.5)
    # return result
    x1, x2 = x[0], x[1]
    a = -sin(x1 - x2)**2 * sin(x1 + x2)**2
    b = sqrt(x1**2 + x2**2)   
    c = a / b
    return c

def tst_function(x):
    x_decimal = np.array(x)
    result = np.sum(np.square(x_decimal))+5
    return result

def hgbat_function(x):
    x_decimal = np.array(x)
    D = x_decimal.shape[0]
    first_part = np.sqrt(np.abs(np.sum(np.square(x_decimal))**2 - np.sum(x_decimal)**2))
    second_part = (0.5*np.sum(np.square(x_decimal))+np.sum(x_decimal))/D + 0.5
    result = first_part + second_part
    return result