import traceback
import numpy as np
from scipy.stats import multivariate_normal as mvnorm

from multiprocessing import Pool
import functools

def filter_path(V, C, A, W, state, var, data, i):
	'''compute log-likelihood for an individual across time

	Parameters
	----------
	V : array
		(K x K) covariance parameters for states
	C : array
		(M x K) loading parameters in measurement system
	A : array
		(K x K) law of motion parameters for states
	W : array
		(M x M) covariance parameters for measurement system errors
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
	
	data = data.iloc[i]

	llf = mvnorm.logpdf(data[0], mean=np.zeros(data[0].shape[0]), 
		cov=np.ones(data[0].shape[0]))

	for t in data.index.levels[0][1:]:

		measure = data[t].reshape((-1, 1))

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
		llf += mvnorm.logpdf(measure.flat, mean=pmeasure.flat, cov=pmeasure_var)

	return llf

def filter_sample(V, C, A, W, state, var, data):
	'''sample likelihood for kalman filter

	Parameters
	----------
	V : array
		(K x K) covariance parameters for states
	C : array
		(M x K) loading parameters in measurement system
	A : array
		(K x K) law of motion parameters for states
	W : array
		(M x M) covariance parameters for measurement system errors
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

		llf = sum((filter_path(V, C, A, W, state, var, data, i) for i in xrange(len(data))))

		print -llf

		return -llf

	except:

		print traceback.format_exc()
		print "Error: returning very large number"
		return np.inf

def pfilter_sample(V, C, A, W, state, var, data, threads=4):
	'''sample likelihood for kalman filter

	Parameters
	----------
	V : array
		(K x K) covariance parameters for states
	C : array
		(M x K) loading parameters in measurement system
	A : array
		(K x K) law of motion parameters for states
	W : array
		(M x M) covariance parameters for measurement system errors
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
		pool = Pool(threads)

		llf = sum(pool.map(functools.partial(filter_path, V, C, A, W, state, var, data), 
			xrange(len(data))))

		pool.close()
		pool.join()

		print -llf

		return -llf

	except:

		print traceback.format_exc()
		print "Error: returning very large number"
		return np.inf