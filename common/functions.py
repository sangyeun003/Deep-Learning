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

def	sum_squares_error(y, t):
	return 0.5 * np.sum((y - t) ** 2)

# def cross_entropy_error(y, t):
# 	delta = 1e-7		# 아주 작은 수 (0.00000001)
# 	return -np.sum(t * np.log(y + delta))	# log 함수에 0 입력하면 -무한대 -> 아주 작은 수 더해서 -무한대인 경우 없앰

def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	
	# 훈련 데이터가 one-hot 벡터라면 정답 레이블의 index로 반환
	if t.size == y.size:
		t = t.argmax(axis = 1)
	
	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size