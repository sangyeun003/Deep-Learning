import numpy as np

class Sigmoid:
	def __init__(self):
		self.out = None		# 순전파의 출력을 out에 저장했다가, 역전파 계산에 사용
	
	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out

		return out
	
	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out		# dL/dx

		return dx