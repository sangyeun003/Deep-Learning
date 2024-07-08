import numpy as np

def function_2(x):
	# return np.sum(x ** 2)
	return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)		# x와 모양이 같은 0으로 찬 배열 생성

	for idx in range(x.size):	# 방향 별로 기울기 계산
		tmp_val = x[idx]
		# f(x+h) 계산
		x[idx] = tmp_val + h
		fxh1 = f(x)

		# f(x-h) 계산
		x[idx] = tmp_val - h
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val	# 값 복원
	return grad

# [6. 8.]
print(numerical_gradient(function_2, np.array([3.0, 4.0])))		# x0=3, x1=4일 때 gradient
# [0. 4.]
print(numerical_gradient(function_2, np.array([0.0, 2.0])))		# x0=3, x1=4일 때 gradient
# [6. 0.]
print(numerical_gradient(function_2, np.array([3.0, 0.0])))		# x0=3, x1=4일 때 gradient

# 위와 결과 다름
# h가 매우 작은 실수 값 -> 정수형 배열로 하면 반올림 오차 발생 -> 부정확. 엉뚱한 값
# [25000 35000]
print(numerical_gradient(function_2, np.array([3, 4])))
# [0 15000]
print(numerical_gradient(function_2, np.array([0, 2])))
# [25000 0]
print(numerical_gradient(function_2, np.array([3, 0])))