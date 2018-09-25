import numpy as np

import tengp

from gpbenchmarks import get_data

def pdivide(x, y):
    return np.divide(x, y, out=np.copy(x), where=x!=0)

def plog(x, y):
    return np.log(x, out=np.copy(x), where=x>0)

def psin(x, y):
    return np.sin(x)

def pcos(x, y):
    return np.cos(x)

def pow(x, y):
    return np.power(x, y)

def pow2(x, y):
    return x**2

def pow3(x, y):
    return x**3

def ptan(x, y):
    return np.tan(x)

def ptanh(x, y):
    return np.tanh(x)

def psqrt(x, y):
    return  np.sqrt(x)

def pexp(x, y):
    return np.exp(x)

def pow_minus1(x, y):
    return x**(-1)

def constant(x, y):
    return x

def constant1(x, y):
    return np.tile(1, x.shape)

nguyen7_funset = tengp.FunctionSet()
nguyen7_funset.add(np.add, 2)
nguyen7_funset.add(np.subtract, 2)
nguyen7_funset.add(np.multiply, 2)
nguyen7_funset.add(pdivide, 2)
nguyen7_funset.add(plog, 2)
nguyen7_funset.add(psin, 2)
nguyen7_funset.add(pcos, 2)


c1nguyen7_funset = tengp.FunctionSet()
c1nguyen7_funset.add(np.add, 2)
c1nguyen7_funset.add(np.subtract, 2)
c1nguyen7_funset.add(np.multiply, 2)
c1nguyen7_funset.add(pdivide, 2)
c1nguyen7_funset.add(plog, 2)
c1nguyen7_funset.add(psin, 2)
c1nguyen7_funset.add(pcos, 2)
c1nguyen7_funset.add(constant1, 2)


cnguyen7_funset = tengp.FunctionSet()
cnguyen7_funset.add(constant, 2)
cnguyen7_funset.add(np.add, 2)
cnguyen7_funset.add(np.subtract, 2)
cnguyen7_funset.add(np.multiply, 2)
cnguyen7_funset.add(pdivide, 2)
cnguyen7_funset.add(plog, 2)
cnguyen7_funset.add(psin, 2)
cnguyen7_funset.add(pcos, 2)


#     function set: +, -, *, /, sin, cos, tan, tanh, sqrt, exp, log, **2, **3
korns12_funset = tengp.FunctionSet()
korns12_funset.add(np.add, 2)
korns12_funset.add(np.subtract, 2)
korns12_funset.add(np.multiply, 2)
korns12_funset.add(pdivide, 2)
korns12_funset.add(psin, 2)
korns12_funset.add(pcos, 2)
korns12_funset.add(ptan, 2)
korns12_funset.add(ptanh, 2)
korns12_funset.add(psqrt, 2)
korns12_funset.add(pexp, 2)
korns12_funset.add(plog, 2)
korns12_funset.add(pow2, 2)
korns12_funset.add(pow3, 2)

ckorns12_funset = tengp.FunctionSet()
ckorns12_funset.add(constant, 2)
ckorns12_funset.add(np.add, 2)
ckorns12_funset.add(np.subtract, 2)
ckorns12_funset.add(np.multiply, 2)
ckorns12_funset.add(pdivide, 2)
ckorns12_funset.add(psin, 2)
ckorns12_funset.add(pcos, 2)
ckorns12_funset.add(ptan, 2)
ckorns12_funset.add(ptanh, 2)
ckorns12_funset.add(psqrt, 2)
ckorns12_funset.add(pexp, 2)
ckorns12_funset.add(plog, 2)
ckorns12_funset.add(pow2, 2)
ckorns12_funset.add(pow3, 2)



vlad_funset = tengp.FunctionSet()
vlad_funset.add(np.add, 2)
vlad_funset.add(np.subtract, 2)
vlad_funset.add(np.multiply, 2)
vlad_funset.add(pdivide, 2)
vlad_funset.add(np.power, 2)
vlad_funset.add(psin, 2)
vlad_funset.add(pcos, 2)
vlad_funset.add(psqrt, 2)
vlad_funset.add(pexp, 2)
vlad_funset.add(plog, 2)
vlad_funset.add(pow_minus1, 2)

pagie_funset = tengp.FunctionSet()
pagie_funset.add(np.add, 2)
pagie_funset.add(np.subtract, 2)
pagie_funset.add(np.multiply, 2)
pagie_funset.add(pdivide, 2)
pagie_funset.add(np.power, 2)
pagie_funset.add(psqrt, 2)
pagie_funset.add(plog, 2)
pagie_funset.add(pow_minus1, 2)

keijzer_funset = tengp.FunctionSet()
keijzer_funset.add(np.add, 2)
keijzer_funset.add(np.multiply, 2)
keijzer_funset.add(psin, 2)
keijzer_funset.add(pcos, 2)
keijzer_funset.add(psqrt, 2)
keijzer_funset.add(plog, 2)
keijzer_funset.add(pow_minus1, 2)
