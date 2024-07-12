import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.util import im2col, col2im

class Pooling:
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad
	
	def forward(self, x):
		N, C, H, W = x.shape
		out_h = int((H - self.pool_h) / self.stride + 1)
		out_w = int((W - self.pool_w) / self.stride + 1)

		# 1. 전개
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h * self.pool_w)

		# 2. 최대값 구하기
		out = np.max(col, axis = 1)

		# 3. 성형
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

		return out
	
	def backward(self, dout):
		dout = dout.transpose(0, 2, 3, 1)

		pool_size = self.pool_h * self.pool_w
		dmax = np.zeros((dout.size, pool_size))
		dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
		dmax = dmax.reshape(dout.shape + (pool_size,)) 

		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

		return dx