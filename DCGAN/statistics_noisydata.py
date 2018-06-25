from D import NoizyData
import numpy as np

x,_ = NoizyData(0.5,2.0,0.3).train()

from keras.datasets import mnist
(_, y), _ = mnist.load_data()
y = np.concatenate([y,y])

c = np.zeros((28 * 3,28 * 11,1))

for w in range(10):
	target = x[y == w]
	c[0:28, w * 28:w * 28 + 28] = np.max(target, axis=0)
	c[28:28 * 2, w * 28:w * 28 + 28] = np.mean(target, axis=0)
	c[28 * 2:, w * 28:w * 28 + 28] = np.min(target, axis=0)

c[0:28,280:] = np.max(x,axis=0)
c[28:28 * 2,280:] = np.mean(x,axis=0)
c[28 * 2:,280:] = np.min(x,axis=0)

import cv2
cv2.imwrite('nst.png',c*255)

x = x[len(x) // 2:] #noisy part
nr = 4
nc = 10

c = np.ones((28 * nr,28 * nc,1)) * 0.5

for a in range(nr):
	for b in range(nc):
		c[28 * a:28 * a + 28,28 * b:28 * b + 28] = x[np.random.choice(len(x))]

c*=255

import cv2

cv2.imwrite('nsample.png',c)