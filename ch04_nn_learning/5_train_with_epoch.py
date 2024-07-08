import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# Hyper Parameter
iters_num = 10000	# 반복 횟수 적절히 설정
train_size = x_train.shape[0]
batch_size = 100	# 미니 배치 크기
learning_rate = 0.1

train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

# 1epoch당 반복 횟수
# 60000 / 100 = 600
# train_size / batch_size가 1보다 작은 경우, 1로 설정
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
	# 미니배치 획득
	batch_mask = np.random.choice(train_size, batch_size)	# 0~train_size-1 중에서 batch_size만큼 랜덤 선택
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# 기울기 계산
	grad = network.numerical_gradient(x_batch, t_batch)
	# grad = network.gradient(x_batch, t_batch)		# 오차 역전파법
	
	# 매개변수 갱신
	for key in ('W1', 'b1', 'W2', 'b1'):
		network.params[key] -= learning_rate * grad[key]
	
	# 학습 경과 기록
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	# 1 epoch 당 정확도 계산
	if i % iter_per_epoch == 0:
		train_accuracy = network.accuracy(x_train, t_train)
		test_accuracy = network.accuracy(x_test, t_test)
		train_accuracy_list.append(train_accuracy)
		test_accuracy_list.append(test_accuracy)
		print("Train accuracy, Test accuracy : " + str(train_accuracy) + ", " + str(test_accuracy))