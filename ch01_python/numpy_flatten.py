import numpy as np

x = np.array([[51, 55], [14, 19], [0, 4]])
x = x.flatten()					# 평탄화
print(x)

print(x[np.array([0, 2, 4])])	# 특정 원소만 추출

print(x > 15)					# dtype = bool
print(x[x > 15])				# 조건을 만족하는 원소만 추출