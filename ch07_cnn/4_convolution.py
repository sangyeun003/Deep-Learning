import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.util import im2col, col2im

# Convolution 계층
class Convolution:
	def __init__(self, W, b, stride=1, pad=0):
		self.W = W
		self.b = b
		self.stride = stride
		self.pad = pad
	
	def forward(self, x):
		FN, C, FH, FW = self.w.shape	# Filter의 형상
		N, C, H, W = x.shape			# input data의 형상
		out_h = int((H + 2 * self.pad - FH) / self.stride + 1)
		out_w = int((W + 2 * self.pad - FW) / self.stride + 1)
		
		col = im2col(x, FH, FW, self.stride, self.pad)
		col_W = self.W.reshape(FN, -1).T	# 행렬 전치
		# 필터 전개. -1을 사용하면 앞 인자(FN)에 맞춰 알아서 열 수 맞춰줌(총 원소 수 / FN)

		out = np.dot(col, col_W) + self.b

		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
		# 축의 순서를 바꿔줌. -1 써서 채널 수 신경 안써도 자동으로 해줌

		return out
	
	def backward(self, dout):
		FN, C, FH, FW = self.W.shape
		dout = dout.transpose(0,2,3,1).reshape(-1, FN)

		self.db = np.sum(dout, axis=0)
		self.dW = np.dot(self.col.T, dout)
		self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

		dcol = np.dot(dout, self.col_W.T)
		dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

		return dx