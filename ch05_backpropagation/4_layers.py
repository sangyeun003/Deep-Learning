# MulLayer & AddLayer

class MulLayer:
	def __init__(self):
		self.x = None
		self.y = None
	
	def forward(self, x, y):
		self.x = x
		self.y = y
		out = x * y

		return out
	
	def backward(self, dout):
		dx = dout * self.y		# dL/dx
		dy = dout * self.x		# dL/dy

		return dx, dy			# dL/dx, dL/dy

class AddLayer:
	# 초기화 필요 x
	def __init__(self):
		pass	# 아무 것도 하지 마라
	
	def forward(self, x, y):
		out = x + y

		return out
	
	def backward(self, dout):
		dx = dout * 1		# dL/dx
		dy = dout * 1		# dL/dy

		return dx, dy		# dL/dx, dL/dy