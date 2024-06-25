from neuralnet_mnist import get_data, init_network, predict
import numpy as np

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y)	# 확률이 제일 높은 곳의 index 찾기
	if p == t[i]:		# 정답이랑 같으면
		accuracy_cnt += 1
print("Accuracy: " + str(float(accuracy_cnt) / len(x)))