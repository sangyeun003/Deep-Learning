import numpy as np

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7

print(w * x)				# 행렬 크기 같으므로 각 원소끼리 곱함
print(np.sum(w * x))
print(np.sum(w * x) + b)

# [0.  0.5]
# 0.5
# -0.19999999999999996