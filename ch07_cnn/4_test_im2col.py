import sys, os
sys.path.append(os.pardir)

import numpy as np
from common.util import im2col

x1 = np.random.rand(1, 3, 7, 7)		# (데이터 수, 채널 수, 높이, 너비)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)	# (데이터 수, 채널 수, 높이, 너비)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)