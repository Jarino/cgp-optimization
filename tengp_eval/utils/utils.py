import numpy as np

from tengp_eval.utils.symreg import get_benchmark_keijzer

def get_data(generator, random, bench_id):
    """
    get data from given generator and add constant column to inputs
    """
    x_train, y_train, x_test, y_test = generator(random, bench_id)
    x_train = np.c_[np.ones(len(x_train)), x_train]
    x_test = np.c_[np.ones(len(x_test)), x_test]
    return x_train, y_train, x_test, y_test


def get_keijzer_data(random, bench_id):
    return get_data(get_benchmark_keijzer, random, bench_id)

