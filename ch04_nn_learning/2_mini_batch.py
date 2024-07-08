import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]	# 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)	# 0~59999 범위에서 10개 난수 추출

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# t가 one-hot-encoding인 경우
def cross_entropy_error(y, t):
	if y.ndim == 1:	# 데이터가 1개라면
		t = t.reshape(1, t.size)	# reshape하면, shape = (size,) -> shape = (1, size)
		y = y.reshape(1, y.size)
	
	batch_size = y.shape[0]
	return -np.sum(t * np.log(y + 1e-7)) / batch_size

# t가 one-hot-encoding이 아닌 경우
def cross_entropy_error(y, t):
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	
	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
	# ex. n = 5, t = [2, 7, 0, 9, 4] 
	# np.arange(5) -> 0 ~ 4까지 배열 생성: [0, 1, 2, 3, 4]
	# y[np.arange(5), t] -> [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]]
	# one-hot-encoding과 달리, t에 정답이 바로 들어있음