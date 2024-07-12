import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

def __train(weight_init_std):
	bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, weight_init_std=weight_init_std, use_batchnorm=True)
	network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, weight_init_std=weight_init_std)
	optimizer = SGD(lr = learning_rate)

	train_accuracy_list = []
	bn_train_accuracy_list = []

	iter_per_epoch = max(train_size / batch_size, 1)
	epoch_cnt = 0

	for i in range(1000000000):
		batch_mask = np.random.choice(train_size, batch_size)
		x_batch = x_train[batch_mask]
		t_batch = t_train[batch_mask]

		for _network in (bn_network, network):
			grads = _network.gradient(x_batch, t_batch)
			optimizer.update(_network.params, grads)

		if i % iter_per_epoch == 0:
			train_accuracy = network.accuracy(x_train, t_train)
			bn_train_accuracy = bn_network.accuracy(x_train, t_train)
			train_accuracy_list.append(train_accuracy)
			bn_train_accuracy_list.append(bn_train_accuracy)

			print("epoch: " + str(epoch_cnt) + " | " + str(train_accuracy) + " - " + str(bn_train_accuracy))

			epoch_cnt += 1
			if epoch_cnt >= max_epochs:
				break
	
	return train_accuracy_list, bn_train_accuracy_list

# 그래프 그리기
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
	print("================= " + str(i + 1) + "/16" + " =====================")
	train_accuracy_list, bn_train_accuracy_list = __train(w)

	plt.subplot(4, 4, i + 1)
	plt.title("W: " + str(w))
	if i == 15:
		plt.plot(x, bn_train_accuracy_list, label = 'Batch Normalization', markevery=2)
		plt.plot(x, train_accuracy_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
	else:
		plt.plot(x, bn_train_accuracy_list, markevery=2)
		plt.plot(x, train_accuracy_list, linestyle='--', markevery=2)
	
	plt.ylim(0, 1.0)
	if i % 4:
		plt.yticks([])
	else:
		plt.ylabel("Accuracy")
	
	if i < 12:
		plt.xticks([])
	else:
		plt.xlabel("Epochs")
	
	plt.legend(loc='lower right')

plt.show()