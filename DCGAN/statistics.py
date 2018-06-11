import numpy as np
import cv2

from keras.datasets import mnist
(x, y), (tx, ty) = mnist.load_data()

f = lambda a: np.max(a,axis=0)
name = 'max'

z = np.zeros((10,28,28))

for n in range(10):
	z[n] = f(x[y==n])
	
for n in range(10):
	cv2.imwrite('{}.{}.png'.format(name,n),z[n])

cv2.imwrite('{}.all.png'.format(name),f(z))