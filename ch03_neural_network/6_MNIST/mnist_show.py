import sys, os
sys.path.append(os.pardir)
# 현재 파일이 위치한 dir의 부모 dir을 sys.path에 추가 -> 부모 dir에 있는 모듈을 import할 수 있게 됨

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))	# np.uint8(): 정수를 8bit(0~255) 정수로 만듦
	pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]	# 첫 번째 사진의 1*784 1차원 배열
label = t_train[0]	# img에 대한 정답
print(label)

print(img.shape)
img = img.reshape(28, 28)	# 1*784(1차원) -> 28*28(2차원)
# print(img)					# (28*28)

img_show(img)