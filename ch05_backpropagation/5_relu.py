class Relu:
	def __init__(self):
		self.mask = None
	
	def forward(self, x):
		self.mask = (x <= 0)
		out = x.copy()
		out[self.mask] = 0		# x<=0인 곳의 값을 0으로 만듦

		return out
	
	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout

		return dx

# import numpy as np
# r = Relu()
# print(r.forward(np.array([-1])))