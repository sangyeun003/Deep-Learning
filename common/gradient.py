import numpy as np

def numerical_gradient_1d(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	for idx in range(x.size):
		tmp_val = x[idx]

		# f(x+h)
		x[idx] = float(tmp_val) + h
		fxh1 = f(x)
		# f(x-h)
		x[idx] = float(tmp_val) - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2 * h)

		# 값 복원
		x[idx] = tmp_val
	return grad

def numerical_gradient_2d(f, X):
	if X.ndim == 1:
		return numerical_gradient_1d(f, X)
	else:
		grad = np.zeros_like(X)

		for idx, x in enumerate(X):		# enumerate: index와 값을 동시에 담아줌 -> x = X[idx]
			grad[idx] = numerical_gradient_1d(f, x)
		
		return grad

def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	# np.nditer(): numpy의 다차원 배열을 순회하기 위한 iterator 생성
	# flags=['multi_index']: 다차원 배열 index 사용
	# op_flags=['readwrite']: 요소를 읽고 쓸 수 있게 해줌
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

	while not it.finished:				# 배열 모든 요소 순회할 때까지 반복
		idx = it.multi_index			# 현재 요소의 다차원 index 가져옴
		tmp_val = x[idx]
		x[idx] = float(tmp_val) + h
		fxh1 = f(x)	# f(x + h)

		x[idx] = float(tmp_val) - h
		fxh2 = f(x)	# f(x - h)

		grad[idx] = (fxh1 - fxh2) / (2 * h)

		x[idx] = tmp_val	# 값 복원
		it.iternext()

# f: 최적화하려는 함수, init_x: 초기값(시작점), lr: learning rate, step_num: 경사법 반복 횟수
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
	x = init_x

	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x -= lr * grad
	return x