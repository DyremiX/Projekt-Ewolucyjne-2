# TODO zmienic na funkcje z tego co on chciał (gotowe biblioteki) i na nich uruchamiac - zrobione ale jakby ktoś jeszcze zweryfikował, to byłoby git
import numpy as np
from benchmark_functions import Keane
from opfunu.utils.operator import hgbat_func


def keane_function(x):
    return Keane(len(x))._evaluate(x)


def tst_function(x):
    x_decimal = np.array(x)
    result = np.sum(np.square(x_decimal)) + 5
    return result


def hgbat_function(x):
    return hgbat_func(x)
