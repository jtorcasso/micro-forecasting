from __future__ import division
import os, sys
import numpy as np
import pandas as pd
from pystata import _retrieve_data
from filtering import filter_sample, pfilter_sample
from OptimController import ParameterSpace, SimpleMinimizer

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

# Initialization

state = np.array([[1.], [1.]])
var = np.array([[1., 0.,], [0., 1.]])

m = SimpleMinimizer(func=pfilter_sample, parameters=p, fargs=(state, var, data, 4),
                     method='Powell', tol=1e-3)
m.log('memory')
m.run()