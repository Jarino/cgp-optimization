from configparser import ConfigParser
import numpy as np
from sklearn.metrics import mean_squared_error
from tengp import simple_es, Parameters, FunctionSet
from tengp.individual import NPIndividual, IndividualBuilder

def define_cgp_system(n_nodes, n_inputs, n_outputs, funset, max_back):
    """
    define CCGP system

    Return:
        IndividualBuilder object
        Parameters
        bounds (tuple)
    """
    params = Parameters(n_inputs, n_outputs, 1, n_nodes, funset, real_valued=False, max_back=max_back)
    ib = IndividualBuilder(params)
    bounds = ib.create().bounds
    return ib, params, bounds

if __name__ == '__main__':

    import sys
    import random
    import pickle
    import os
    from time import time
    from tengp_eval.utils import get_keijzer_data
    from tengp_eval.utils.function_sets import keijzer_fixed_funset

    output_dir = sys.argv[1]
    configs_dir = sys.argv[2]


    np.warnings.filterwarnings('ignore')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time()

    for bench_id in  range(1,16):
        x_train, y_train, x_test, y_test = get_keijzer_data(random, bench_id)


        cp = ConfigParser()
        cp.read(os.path.join(configs_dir, f'baseline.ini'))

        max_trials = 1

        ib, params, bounds = define_cgp_system(
                cp.getint('DEFAULT', 'n_nodes'),
                x_train.shape[1] if len(x_train.shape) > 1 else 1,
                y_train.shape[1] if len(y_train.shape) > 1 else 1,
                keijzer_fixed_funset,
                cp.getint('DEFAULT', 'max_back'))

        all_logs = []
        random.seed(42)

        for i in range(max_trials):
            print(f'keijzer{bench_id}-{i}')
            log = []
            simple_es(x_test, y_test, mean_squared_error, params,
                evaluations=cp.getint('DEFAULT', 'evaluations'), mutation=cp['DEFAULT']['mutation'], log=log)
            all_logs.append(log)

        with open(os.path.join(output_dir, f'baseline{bench_id}.log'), 'wb') as f:
            pickle.dump(all_logs, f)

    print(f'finised, wall time: {time() - start}')
