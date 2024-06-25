import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))	# 차원 확인
print(A.shape)		# 배열 모양 확인(튜플로 리턴) -> shape: 인스턴스 변수
print(A.shape[0])	# 배열 행 수 확인

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
print(B.shape[0])