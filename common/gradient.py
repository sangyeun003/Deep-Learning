import numpy as np

def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)

	for idx in range(x.size):
		tmp_val = x[idx]

		# f(x+h)
		x[idx] = tmp_val + h
		fxh1 = f(x)
		# f(x-h)
		x[idx] = tmp_val - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2 * h)

		# 값 복원
		x[idx] = tmp_val
	return grad

# f: 최적화하려는 함수, init_x: 초기값(시작점), lr: learning rate, step_num: 경사법 반복 횟수
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
	x = init_x

	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x -= lr * grad
	return x