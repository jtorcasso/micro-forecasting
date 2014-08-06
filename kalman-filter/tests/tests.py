import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
print sys.path[0]
from filtering import pfilter_sample, filter_sample
from paras import ParameterSpace

def dgp(V, C, A, W, state, var):
	N = 100
	M = 6
	T = 4
	K = 2
	data = np.zeros((N, M*T))

	for row in data:
		idata = np.ones(M*T)
		state_ = np.random.multivariate_normal(mean=state.flat, cov=var).reshape((-1, 1))
		idata[:M] = C.dot(state_).flat + np.random.multivariate_normal(mean=np.zeros(M), cov=W)
		for t in xrange(1, T):
			state_ = A.dot(state_) + np.random.multivariate_normal(mean=np.zeros(K), cov=V).reshape((-1, 1))
			idata[M*t:M*(t+1)] = C.dot(state_).flat + np.random.multivariate_normal(mean=np.zeros(M), cov=W)
		row[:] = idata

	data = pd.DataFrame(data)
	data.columns = pd.MultiIndex.from_product([range(T), range(M)])

	return data


np.random.seed(1234)

state = np.array([[1.], [1.]])
var = np.array([[1., 0.,], [0., 1.]])

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

V, C, A, W = p['V'], p['C'], p['A'], p['W']

data = dgp(V.value, C.value, A.value, W.value, state, var)

params0 = np.array([par.transform(par.value, direction='out')+np.random.randn() for par in p.params if par.isfree()])
p.summary()
def wrap_filter(params, p, state, var, data):
	
	p.update(params)

	V, C, A, W = p['V'].value, p['C'].value, p['A'].value, p['W'].value

	return filter_sample(V, C, A, W, state, var, data)

start = time.time()
params = minimize(wrap_filter, x0=params0+np.random.randn(len(params0)), 
                  args=(p, state, var, data), method='Powell', tol=1e-2)
print 'Finished in {} seconds'.format(time.time() - start)
p.summary()