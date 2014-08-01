import os
from pystata import _retrieve_data
import numpy as np
from scipy.stats import multivariate_normal as mvnorm

data = _retrieve_data(os.path.join(os.environ['erc'], 'data', 'cnlsy-base-data', 'CNLSY_Data.dta'))['data']

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

def increment(params, state, var, measure):
	'''predict and update steps, calculate likelihood

	Parameters 
	----------
	params : array
		parameter values
	state : array
		current state
	var : array
		current variance
	measure : array
		measurements of next state
	'''

	V, C, A, W = packpars(params)

	# Prediction Step
	print A.shape, state.shape
	pstate = A.dot(state)
	print pstate.shape
	pstate_var = A.dot(var).dot(A.T) + V
	pmeasure = C.dot(pstate)
	print pmeasure.shape, C.shape
	pmeasure_var = C.dot(pstate_var).dot(C.T) + W

	# Update Step
	kalman_gain = pstate_var.dot(C.T).dot(np.linalg.pinv(pmeasure_var))
	print kalman_gain.shape, pstate.shape, measure.shape, pmeasure.shape
	state = pstate + kalman_gain.dot(measure - pmeasure)
	var = pstate_var - kalman_gain.dot(pstate_var.dot(C.T).T)

	# Step Likelihood
	llf = mvnorm.logpdf(measure, mean=pmeasure, cov=pmeasure_var)

	return state, var, llf

state = np.array([[1., 1.]])
var = np.array([[1., 0.,], [0., 1.]])
measure = np.array([[1.3, 0.7],[0.9, 0.4],[1.1, .8],[0.8, 0.9],[1.9, 0.8],[1., .7]])

print increment(params0, state, var, measure)