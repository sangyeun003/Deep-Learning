# 3차원 그래프
def function_2(x):
	return x[0]**2 + x[1]**2
	# 또는 return np.sum(x**2)

import numpy as np
import matplotlib.pylab as plt

# x = np.array([np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1)])
# print(x[0], x[1])		# 잘 안됨
x, y = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
print(x, y)
z = function_2([x,y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap = 'viridis')

plt.show()

####################################################################

def numerical_diff(f, x):
	h = 1e-4
	return (f(x + h) - f(x - h)) / (2 * h)

def function_tmp1(x0):		# x0=3, x1=4 일 때 x0에 대한 편미분 구하기
	return x0**2 + 4**2
print(numerical_diff(function_tmp1, 3))

def function_tmp2(x1):		# x0=3, x1=4 일 때 x1에 대한 편미분 구하기
	return 3**2 + x1**2
print(numerical_diff(function_tmp2, 4))