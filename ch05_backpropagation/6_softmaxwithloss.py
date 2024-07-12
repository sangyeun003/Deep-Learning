import sys, os
sys.path.append(os.pardir)

from common.functions import softmax, cross_entropy_error

class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None	# 오차
		self.y = None		# Softmax의 출력
		self.t = None		# 정답 레이블(One-hot 벡터)
	
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss
	
	def backward(self, dout = 1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size

		return dx