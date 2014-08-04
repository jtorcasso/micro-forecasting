import os
from pystata import _retrieve_data
import numpy as np
from scipy.stats import multivariate_normal as mvnorm

cnlsy = _retrieve_data(os.path.join(os.environ['erc'], 'data', 'cnlsy-base-data', 'CNLSY_Data.dta'))['data']

# Measures in the Data

# Delinquency -- Ages 4(2)14
# bpi4             Cheats or tells lies
# bpi6             Argues too much
# bpi9             Bullies, or is cruel/mean to others

# Focus -- Ages 4(2)14
# bpi7             Difficult concentrating
# bpi13            Impulsive, or acts without thinking
# bpi17            Restless, overly active, cannot sit still

def packpars(params):
	'''packs params into matrices'''

	V = np.diag(np.exp(params[:2]))
	C = np.zeros((6, 2))
	C[1:3, 0] = params[2:4]
	C[4:6, 1] = params[4:6]
	C[0, 0] = -1.0
	C[3, 1] = -1.0
	A = params[6:10].reshape((2, 2))
	W = np.diag(np.exp(params[10:]))

	return V, C, A, W

params0 = np.array([0., 0, -1, -1, -1, -1, 1, .3, .3, 1, 0, 0, 0, 0, 0, 0])

def filter_step(params, state, var, measure):
	'''predict and update steps, calculate likelihood

	Parameters 
	----------
	params : array
		parameter values
	state : array
		current state (K x 1)
	var : array
		current variance (K x K)
	measure : array
		measurements of next state (M x 1, where M > K)

	Notes
	-----
	K is the number of state variables
	M is the number of measurements (of those state variables)
	'''

	# V(K x K), C(M x K), A(K x K), W(M x M)
	V, C, A, W = packpars(params)

	# Prediction Step
	pstate = A.dot(state)  # (K x 1)
	pstate_var = A.dot(var).dot(A.T) + V  # (K x K)
	pmeasure = C.dot(pstate)  # (M x 1)
	pmeasure_var = C.dot(pstate_var).dot(C.T) + W  # (M x M)

	# Update Step
	kalman_gain = pstate_var.dot(C.T).dot(np.linalg.pinv(pmeasure_var)) # (K x M)
	state = pstate + kalman_gain.dot(measure - pmeasure)  # (K x 1)
	var = pstate_var - kalman_gain.dot(pstate_var.dot(C.T).T)  # (K x K)

	# Step Likelihood
	llf = mvnorm.logpdf(measure.flat, mean=pmeasure.flat, cov=pmeasure_var)

	return state, var, llf

def filter_path(params, state, var, data):
	'''compute log-likelihood for an individual across time

	Parameters
	----------
	params : array
		parameter values
	state : array
		initial state (K x 1)
	var : array
		initial variance (K x K)
	data : data series
		measurements of all states across time, with hierarchical
		index with time t then measure m

	Returns
	-------
	llf : float
		log-likelihood contribution for an individual

	Notes
	-----
	Data should be a Series that looks like:

	0  a     7
	   b     8
	1  a     9
	   b    10

	'''

	llf = mvnorm.logpdf(data[0], mean=np.zeros(data[0].shape[0]), 
		cov=np.ones(data[0].shape[0]))

	for t in data.index.levels[0][1:]:

		state, var, llf_ = filter_step(params, state, var, data[t].reshape((-1, 1)))

		llf += llf_

	return llf

def filter_sample(params, state, var, data):
	'''sample likelihood for kalman filter

	Parameters
	----------
	params : array
		parameter values
	state : array
		initial state (K x 1)
	var : array
		initial variance (K x K)
	data : DataFrame
		measurements of all states across time for all individuals, 
		with hierarchical index with time t then measure m

	Returns
	-------
	llf : float
		log-likelihood contribution for the sample
	'''
	try:
		llf = sum((filter_path(params, state, var, r) for i,r in data.iterrows()))

		print -llf
		return -llf
	except:
		return np.inf

def dgp(params, state, var):
	N = 1000
	M = 6
	T = 4
	K = 2
	data = np.zeros((N, M*T))

	V, C, A, W = packpars(params)

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


import pandas as pd
import time
np.random.seed(1234)
state = np.array([[1.], [1.]])
var = np.array([[1., 0.,], [0., 1.]])
data = dgp(params0, state, var)

from scipy.optimize import minimize

start = time.time()
params = minimize(filter_sample, x0=params0+np.random.randn(len(params0)), 
                  args=(state, var, data), method='CG', tol=1e-2)
print 'Finished in {} seconds'.format(time.time() - start)
V, C, A, W = packpars(params.x)
print V
print C
print A
print W

# data = cnlsy[['{}age{}'.format(s, a) for s in ['bpi4', 'bpi6', 'bpi9', 'bpi7', 'bpi13', 'bpi17'] for a in xrange(2, 15, 2)]].dropna()

# data.columns = pd.MultiIndex.from_product([range(0, 13, 2), ['bpi4', 'bpi6', 'bpi9', 'bpi7', 'bpi13', 'bpi17']])

# print filter_sample(params0, state, var, data)

# from scipy.optimize import minimize

# params = minimize(filter_sample, params0, args=(state, var, data))