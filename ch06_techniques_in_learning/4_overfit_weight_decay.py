import os, sys
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay(가중치 감쇠) 설정
weight_decay_lambda = 0		# weight decay를 사용하지 않을 경우
# weight_decay_lambda = 0.1

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10, weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr = 0.01)	# 학습률이 0.01인 SGD로 매개변수 갱신

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	grads = network.gradient(x_batch, t_batch)
	optimizer.update(network.params, grads)

	if i % iter_per_epoch == 0:
		train_accuracy = network.accuracy(x_train, t_train)
		test_accuracy = network.accuracy(x_test, t_test)
		train_accuracy_list.append(train_accuracy)
		test_accuracy_list.append(test_accuracy)

		print("Epoch: " + str(epoch_cnt) + ", train accuracy: " + str(train_accuracy) + ", test accuracy: " + str(test_accuracy))

		epoch_cnt += 1
		if epoch_cnt >= max_epochs:
			break

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_accuracy_list, marker = 'o', label='train', markevery=10)
plt.plot(x, test_accuracy_list, marker='s', label = 'test', markevery=10)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()