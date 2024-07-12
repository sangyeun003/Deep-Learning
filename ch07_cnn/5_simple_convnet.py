import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.layers import Convolution, Pooling, Relu, Affine, SoftmaxWithLoss
from collections import OrderedDict

class SimpleConvNet:
	def __init__(self, input_dim=(1, 28, 28), 
			  conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, 
			  hidden_size=100, output_size=10, weight_init_std=0.01):
		# 하이퍼파라미터를 딕셔너리에서 가져오기
		filter_num = conv_param['filter_num']
		filter_size = conv_param['filter_size']
		filter_pad = conv_param['pad']
		filter_stride = conv_param['stride']
		input_size = input_dim[1]
		conv_output_size = (input_size + 2 * filter_pad - filter_size) / filter_stride + 1
		pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

		# 가중치 매개변수 초기화
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
		self.params['b1'] = np.zeros(filter_num)
		self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
		self.params['b2'] = np.zeros(hidden_size)
		self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b3'] = np.zeros(output_size)

		# CNN을 구성하는 계층
		self.layers = OrderedDict()
		self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
		self.layers['Relu1'] = Relu()
		self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

		self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
		self.layers['Relu2'] = Relu()
	
		self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
		self.last_layer = SoftmaxWithLoss()

		# 추론
		def predict(self, x):
			for layer in self.layers.values():
				x = layer.forward(x)
			
			return x
		
		# 손실함수
		def loss(self, x, t):
			y = self.predict(x)

			return self.last_layer.forward(y, t)
	
		# 오차역전파법으로 기울기 계산
		def gradient(self, x, t):
			# 순전파
			self.loss(x, t)

			# 역전파
			dout = 1
			dout = self.last_layer.backward(dout)

			layers = list(self.layers.values())
			layers.reverse()
			for layer in layers:
				dout = layer.backward(dout)
			
			# 결과 저장
			grads = {}
			grads['W1'] = self.layers['Conv1'].dW
			grads['b1'] = self.layers['Conv1'].db
			grads['W2'] = self.layers['Affine1'].dW
			grads['b2'] = self.layers['Affine1'].db
			grads['W3'] = self.layers['Affine2'].dW
			grads['b3'] = self.layers['Affine2'].db

			return grads