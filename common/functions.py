import numpy as np

def step_function(x):
	if x > 0:
		return 1
	else:
		return 0
	
def step_function_for_nparray(x):
	y = x > 0
	return y.astype(int)

def relu(x):
	return np.maximum(x, 0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def identify_function(x):
	return(x)

# overflow 발생하지 않는 softmax
def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a

	return y