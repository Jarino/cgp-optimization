import pytest
from configparser import ConfigParser
import numpy as np


from tengp import FunctionSet
from ccgp_sh import run_benchmark

def test_single_benchmark_run():
    cp = ConfigParser()
    cp['DEFAULT'] = {
        'trials':'2',
        'n_nodes':'50',
        'max_back':'20',
        'temperature':'1',
        'max_evals':'5',
        'mut_prob':'0.1'
    }

    x_train = np.array([[1, 2], [2, 3], [3, 4]])
    y_train = np.array([1, 2, 3])
    funset = FunctionSet()
    funset.add(np.add, 2)
    funset.add(np.multiply, 2)
    funset.add(np.subtract, 2)

    log = run_benchmark(cp, x_train, y_train, funset)

    assert True #"run without errors"
    assert len(log) == 5
