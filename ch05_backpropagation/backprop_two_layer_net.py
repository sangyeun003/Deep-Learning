# 7_two_layer_net.py와 같은 파일
# 7_two_layer_net.py에서 import하기 위해 만든 파일
# 파일 이름이 숫자로 시작하면 import 불가

import sys, os
sys.path.append(os.pardir)

import numpy as np
from collections import OrderedDict

from common.layers import *
from common.gradient import numerical_gradient

class TwoLayerNet:
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		# 가중치 초기화
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)	# (input_size * hidden_size) 행렬 만듦
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

		# 계층 생성
		self.layers = OrderedDict()		# 순서가 있는 딕셔너리 -> 딕셔너리에 추가한 순서를 기억!
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

		self.lastLayer = SoftmaxWithLoss()
	
	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		
		return x
	
	# x: 입력 데이터, t: 정답 레이블
	def loss(self, x, t):
		y = self.predict(x)
		
		return self.lastLayer.forward(y, t)
	
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis = 1)
		if t.ndim != 1:
			t = np.argmax(t, axis = 1)
		
		accuracy = np.sum(y == t) / float(x.shape[0])
		
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
	
	def gradient(self, x, t):
		# 순전파
		self.loss(x, t)

		# 역전파
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		
		# 결과 저장
		grads = {}
		grads['W1'] = self.layers['Affine1'].dW
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db

		return grads