import sys, os
sys.path.append(os.pardir)

from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
	# input_size: 입력층의 뉴런 수
	# hidden_size: 은닉층의 뉴런 수
	# output_size: 출력층의 뉴런 수
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		# 가중치 초기화
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	def predict(self, x):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']

		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y
	
	# x: 입력 데이터, t: 정답 레이블
	def loss(self, x, t):
		y = self.predict(x)

		return cross_entropy_error(y, t)
	
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)	# 각 행에서의 최대값의 index
		t = np.argmax(t, axis=1)	# 각 행에서의 최대값의 index

		accuracy = np.sum(y == t) / float(x.shape[0])	# x.shape[0]: x의 행 수
		return accuracy
	
	# x: 입력 데이터, t: 정답 레이블
	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads

################################################################
# Example
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

print(net.params['W1'].shape)	# (784, 100)
print(net.params['b1'].shape)	# (100,)
print(net.params['W2'].shape)	# (100, 10)
print(net.params['b2'].shape)	# (10,)

x = np.random.rand(100, 784)	# 100장 input
t = np.random.rand(100, 10)		# 100장 input에 대한 정답 label
y = net.predict(x)

grads = net.numerical_gradient(x, t)	# 기울기 계산

print(grads['W1'].shape)	# (784, 100)
print(grads['b1'].shape)	# (100,)
print(grads['W2'].shape)	# (100, 10)
print(grads['b2'].shape)	# (10,)