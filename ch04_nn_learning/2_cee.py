# 교차 엔트로피 오차 (Cross Entropy Error, CEE)
import numpy as np

def cross_entropy_error(y, t):
	delta = 1e-7		# 아주 작은 수 (0.00000001)
	return -np.sum(t * np.log(y + delta))	# log 함수에 0 입력하면 -무한대 -> 아주 작은 수 더해서 -무한대인 경우 없앰

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

# ex1
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(cross_entropy_error(y, t))	# 0.510825457099338

# ex2
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(cross_entropy_error(y, t))	# 2.302584092994546