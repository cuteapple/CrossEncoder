import numpy as np
import cv2

from keras.datasets import mnist
(x, y), (tx, ty) = mnist.load_data()

z = np.zeros((10,28,28))

for n in range(10):
	z[n] = np.max(x[y==n],axis=0)

for n in range(10):
	cv2.imwrite('{}.png'.format(n),z[n])


cv2.imwrite('all.png',np.amax(z,axis=0))