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

# Example. x0^2 + x1^2의 최소값 구하기
def function_2(x):
	return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])	# [3, 4]로 하면 error
print(gradient_descent(function_2, init_x, 0.1, 100))	# [-6.11110793e-10  8.14814391e-10] ~= [0, 0]

# 학습률이 너무 큰 경우: lr = 10.0
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, 10.0, 100))	# [-2.58983747e+13 -1.29524862e+12]

# 학습률이 너무 작은 경우: lr = 1e-10
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, 1e-10, 100))	# [-2.99999994  3.99999992]