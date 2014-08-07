from __future__ import division
import os, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pystata import _retrieve_data
from filtering import filter_sample, pfilter_sample
from paras import ParameterSpace

cnlsy = _retrieve_data(os.path.join(os.environ['erc'], 'data', 'cnlsy-base-data', 'CNLSY_Data.dta'))['data']

np.random.seed(1234)

# Setting Up the Parameters
p = ParameterSpace()
V = np.array([[1., 0.],
              [0., 1.]])
p.add_parameter(V, 'V')
p['V'].set_free([[True,  False],
                 [False, True]])
p['V'][0,0].set_bounds(0, np.inf)
p['V'][1,1].set_bounds(0, np.inf)
C = np.array([[ 1.,  0.],
              [ 1.,  0.],
              [ 1.,  0.],
              [ 0.,  1.],
              [ 0.,  1.],
              [ 0.,  1.]])
p.add_parameter(C, 'C')
p['C'].set_free([[False,  False],
                 [True,   False],
                 [True,   False],
                 [False,  False],
                 [False,  True],
                 [False,  True]])
A = np.array([[ 1.,  0.3],
              [ 0.3, 1. ]])
p.add_parameter(A, 'A')
W = np.array([[1., 0., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0., 0.],
              [0., 0., 1., 0., 0., 0.],
              [0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 0., 1.]])
p.add_parameter(W, 'W')
p['W'].set_free([[True,  False, False, False, False, False],
                 [False, True,  False, False, False, False],
                 [False, False, True,  False, False, False],
                 [False, False, False, True,  False, False],
                 [False, False, False, False, True,  False],
                 [False, False, False, False, False, True]])
for i in xrange(6):
    p['W'][i,i].set_bounds(0, np.inf)

p.summary()

# Delinquency -- Ages 4(2)14
# bpi4             Cheats or tells lies
# bpi6             Argues too much
# bpi9             Bullies, or is cruel/mean to others

# Focus -- Ages 4(2)14
# bpi7             Difficult concentrating
# bpi13            Impulsive, or acts without thinking
# bpi17            Restless, overly active, cannot sit still

data = cnlsy[['{}age{}'.format(s, a) for s in ['bpi4', 'bpi6', 'bpi9', 'bpi7', 'bpi13', 'bpi17'] for a in xrange(2, 15, 2)]]
data = data.ix[(data.count(axis=1)/data.shape[1]) >= .85, :]
data = data.fillna(data.mean())

data.columns = pd.MultiIndex.from_product([range(0, 13, 2), ['bpi4', 'bpi6', 'bpi9', 'bpi7', 'bpi13', 'bpi17']])

from scipy.optimize import minimize

class Minimizer(object):
    '''interactive wrapper for scipy.optimize.minimize'''

    def __init__(self, func, x0, **kwargs):
        self.func = func
        self.params = x0
        self.callback_kwarg = kwargs.pop('callback', None)
        self.kwargs = kwargs
        self.results = None

    def run(self):
        '''Run the optimizer given the current parameter values'''

        try:
            self.results = minimize(self.func, x0=self.params, 
                callback=self.callback, **self.kwargs)
        except KeyboardInterrupt:
            pass

    def callback(self, params):
        '''record parameters after each optimizer iteration

        Parameters
        ----------
        params : array
            current parameter values
        '''

        self.params = params

        if self.callback_kwarg is not None: self.callback_kwarg(params)

    def fval(self):
        '''evaluate function at the current parameter value'''

        return self.func(self.params, *self.kwargs.get('args', ()))

    def perturb(self, epsilon):
        '''perturb current parameter value by epsilon

        Parameters
        ----------
        epsilon : array or scalar
            array or scalar to perturb parameters
        '''

        self.params += epsilon

def wrap_filter(params, p, state, var, data):

    p.update(params)

    V, C, A, W = p['V'].value, p['C'].value, p['A'].value, p['W'].value

    return pfilter_sample(V, C, A, W, state, var, data, 4)

# Initialization

state = np.array([[1.], [1.]])
var = np.array([[1., 0.,], [0., 1.]])
params0 = np.array([par.transform(direction='out') for par in p.params if par.isfree()])

# params = minimize(wrap_filter, x0=params0, 
#     args=(p, state, var, data), method='Powell', tol=1e-3)
m = Minimizer(wrap_filter, x0=params0, args=(p, state, var, data), 
    method='Powell', tol=1e-3)
m.run()

p.summary()