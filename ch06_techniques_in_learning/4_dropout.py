import numpy as np

class Dropout:
	def __init__(self, dropout_ratio = 0.5):
		self.dropout_ratio = dropout_ratio
		self.mask = None
	
	def forward(self, x, train_flag = True):
		if train_flag:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio	# x와 같은 모양의 행렬. 삭제할 뉴런 위치에 False 저장
			return x * self.mask
		else:
			return x * (1.0 - self.dropout_ratio)
	
	def backward(self, dout):	# ReLU와 동작 같음. 순전파 때 신호를 통과시키는 뉴런은 역전파 때도 신호 그대로 통과. 나머지는 차단
		return dout * self.mask