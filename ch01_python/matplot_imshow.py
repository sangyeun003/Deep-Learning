import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("90011_169155.jpeg")	# 현재 디렉토리 기준(상대 경로) or 절대 경로

plt.imshow(img)
plt.show()