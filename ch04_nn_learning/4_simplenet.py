import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
	def __init__(self):
		self.W = np.random.randn(2, 3)	# 정규 분포로 초기화
	
	def predict(self, x):
		return np.dot(x, self.W)
	
	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		# print(y)
		loss = cross_entropy_error(y, t)

		return loss
	
net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

print("최대값의 index: ", np.argmax(p))	# 최대값의 index

t = np.array([0, 0, 1])	# 정답 label
print("Loss: ", net.loss(x, t))

# 손실 함수
# 이때 사용한 W는 dummy
def f(W):
	return net.loss(x, t)

dW = numerical_gradient(f, net.W)	# dL/dW
print("기울기: ", dW)

# Lambda 기법 사용
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print("기울기: ", dW)
# [[ 0.2926202   0.2725665  -0.5651867 ]
# [ 0.4389303   0.40884975 -0.84778005]]