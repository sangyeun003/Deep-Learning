# 나쁜 구현의 예
# 1. 1e-50은 반올림 오차 발생 -> 컴퓨터가 계산할 때 1e-50은 0으로 계산됨
# 2. 기울기에서 h를 0으로 보내지 못해서 발생하는 오차
# # def numerical_diff(f, x):
# # 	h = 1e-50
# 	return (f(x + h) - f(x)) / h

def numerical_diff(f, x):
	h = 1e-4
	return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
	return 0.01 * (x ** 2) + 0.1 * x	# 0.01 * x ** 2 + 0.1 * x과 동일. 제곱이 연산 순서 우선

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
dy = numerical_diff(function_1, x)

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

plt.title("Numerical Differentiation")
plt.xlabel("x")
plt.ylabel("df(x), f(x)")
plt.plot(x, y)
plt.plot(x, dy)
plt.show()