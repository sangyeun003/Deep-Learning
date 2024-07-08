# 오차 제곱합 (Sum of Squares for Error, SSE)
import numpy as np

def	sum_squares_error(y, t):
	return 0.5 * np.sum((y - t) ** 2)

t = np.array([0, 0, 1, 0, 0 ,0, 0, 0, 0, 0])

# ex1
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(sum_squares_error(y, t))		# 0.09750000000000003

# ex2
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(sum_squares_error(y, t))		# 0.5975