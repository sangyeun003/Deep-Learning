import numpy as np

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

class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.original_x_shape = None
		self.dW = None		# dL/dW
		self.db = None		# dL/db
	
	def forward(self, x):
		# 텐서 대응
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)
		self.x = x

		out = np.dot(x, self.W) + self.b

		return out
	
	def backward(self, dout):
		dx = np.dot(dout, self.W.T)		# dL/dx
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis = 0)

		dx = dx.reshape(*self.original_x_shape)	# 입력 데이터 모양 변경(텐서 대응)
		return dx

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
	
	# def backward(self, dout = 1):
	# 	batch_size = self.t.shape[0]
	# 	dx = (self.y - self.t) / batch_size

	# 	return dx
	def backward(self, dout = 1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size:	# 정답 레이블이 one-hot encoding 형태일 때
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1
			dx = dx / batch_size
		
		return dx

class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx

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

		# 중간 데이터（backward 시 사용）
		self.x = None   
		self.col = None
		self.col_W = None

		# 가중치와 편향 매개변수의 기울기
		self.dW = None
		self.db = None
	
	def forward(self, x):
		FN, C, FH, FW = self.W.shape	# Filter의 형상
		N, C, H, W = x.shape			# input data의 형상
		out_h = int((H + 2 * self.pad - FH) / self.stride + 1)
		out_w = int((W + 2 * self.pad - FW) / self.stride + 1)
		
		col = im2col(x, FH, FW, self.stride, self.pad)
		col_W = self.W.reshape(FN, -1).T	# 행렬 전치
		# 필터 전개. -1을 사용하면 앞 인자(FN)에 맞춰 알아서 열 수 맞춰줌(총 원소 수 / FN)

		out = np.dot(col, col_W) + self.b

		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
		# 축의 순서를 바꿔줌. -1 써서 채널 수 신경 안써도 자동으로 해줌

		self.x = x
		self.col = col
		self.col_W = col_W
		
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

		self.x = None
		self.arg_max = None
	
	def forward(self, x):
		N, C, H, W = x.shape
		out_h = int((H - self.pool_h) / self.stride + 1)
		out_w = int((W - self.pool_w) / self.stride + 1)

		# 1. 전개
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h * self.pool_w)

		# 2. 최대값 구하기
		arg_max = np.argmax(col, axis=1)
		out = np.max(col, axis = 1)

		# 3. 성형
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

		self.x = x
		self.arg_max = arg_max
	
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