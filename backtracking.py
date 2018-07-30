import tengp
import numpy as np
from gpbenchmarks import get_data
from sklearn.metrics import mean_squared_error

def pdivide(x, y):
    return np.divide(x, y, out=np.copy(x), where=x!=0)

def plog(x):
    return np.log(x, out=np.copy(x), where=x>0)

def pow2(x):
    return x**2

def pow3(x):
    return x**3


funset = tengp.FunctionSet()
funset.add(np.add, 2)
funset.add(np.subtract, 2)
funset.add(np.multiply, 2)
funset.add(pdivide, 2)
funset.add(pow2, 1)
funset.add(pow3, 1)


X, y = get_data('nguyenf4', 20, -1, 1)
X = np.c_[np.ones(len(X)), X]

params = tengp.Parameters(2, 1, 1, 10, funset)

builder = tengp.individual.IndividualBuilder(params)

individual = builder.create()

output = individual.transform(X)
individual.fitness = mean_squared_error(y, output)
for node in individual.nodes:
    print(node.value)
