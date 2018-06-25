from D import NoizyData

x,_ = NoizyData(0.5,0.5,0.3).train()

x = x[len(x) // 2:] #noisy part
import numpy as np

nr = 4
nc = 10

c = np.ones((28 * nr,28 * nc,1)) * 0.5

for a in range(nr):
	for b in range(nc):
		c[28 * a:28 * a + 28,28 * b:28 * b + 28] = x[np.random.choice(len(x))]

c*=255

import cv2

cv2.imwrite('nsample.png',c)