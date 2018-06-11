import numpy as np
import cv2

from keras.datasets import mnist
(x, y), (tx, ty) = mnist.load_data()

xn = [np.zeros((28,28))]*10
for n in range(10):
	xn[n] = x[y==n]

def S(f,name):
	z = np.zeros((10,28,28))

	for n in range(10):
		z[n] = f(xn[n])
	
	for n in range(10):
		cv2.imwrite('statistic/{}.{}.png'.format(name,n),z[n])

	cv2.imwrite('statistic/{}.all.png'.format(name),f(z))

S(lambda a: np.max(a,axis=0),'max')
S(lambda a: np.min(a,axis=0),'min')
S(lambda a: np.mean(a,axis=0),'mean')