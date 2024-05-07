# TODO zmienic na funkcje z tego co on chciał (gotowe biblioteki) i na nich uruchamiac - zrobione ale jakby ktoś jeszcze zweryfikował, to byłoby git
import numpy as np
from benchmark_functions import Keane
from opfunu.utils.operator import hgbat_func
from numpy import sin


def keane_function(x):
    return Keane(len(x))._evaluate(x)


def tst_function(x):
    x_decimal = np.array(x)
    result = np.sum(np.square(x_decimal)) + 5
    return result


def hgbat_function(x):
    return hgbat_func(x)

#TODO to jakos?
def schaffer_f2(self, x):
    return 0.5 + ((sin(x[0]**2 + x[1]**2))**2 - 0.5) / ((1 + 0.001*(x[0]**2 + x[1]**2))**2)
