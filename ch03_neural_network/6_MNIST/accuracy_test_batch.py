import neuralnet_mnist as nn
import numpy as np

x, t = nn.get_data()
network = nn.init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
	x_batch = x[i : i + batch_size]			# 100 * 784
	y_batch = nn.predict(network, x_batch)	# 100 * 10
	p = np.argmax(y_batch, axis = 1)		# 100 * 1, 각 행마다 최대값의 index를 뽑아서 numpy 배열 만듦
	accuracy_cnt += np.sum(p == t[i : i + batch_size])	# i ~ i+batch_size-1까지 중 정답인 것의 개수. p==t로 bool형 배열 만듦

# print(p)
# print(p == t[i : i + batch_size])
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
