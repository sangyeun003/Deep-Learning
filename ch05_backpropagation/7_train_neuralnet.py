import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from backprop_two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# 오차 역전파법으로 기울기를 구함
	grad = network.gradient(x_batch, t_batch)

	# 갱신
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	if i % iter_per_epoch == 0:
		train_accuracy = network.accuracy(x_train, t_train)
		test_accuracy = network.accuracy(x_test, t_test)
		
		train_accuracy_list.append(train_accuracy)
		test_accuracy_list.append(test_accuracy)
		print(train_accuracy, test_accuracy)