import numpy as np
import cv2

from keras.datasets import mnist
(x, y), (tx, ty) = mnist.load_data()

xn = [np.zeros((28,28))] * 10
for n in range(10):
	xn[n] = x[y == n]

def S(f,name):
	z = np.zeros((10,28,28))

	for n in range(10):
		z[n] = f(xn[n])
	
	for n in range(10):
		cv2.imwrite(f'statistic/{name}.{n}.png',z[n])

	za = f(z)
	cv2.imwrite(f'statistic/{name}.all.png',za)

	#zip
	canvas = np.zeros((28,28 *11))
	for n in range(10):
		canvas[:,28 * n:28 * n + 28]=z[n]
	canvas[:,28*10:]=za
	cv2.imwrite(f'statistic/{name}.z.png',canvas)

S(lambda a: np.max(a,axis=0),'max')
S(lambda a: np.min(a,axis=0),'min')
S(lambda a: np.mean(a,axis=0),'mean')

nr = 4
nc = 10
c = np.ones((28 * nr,28 * nc)) * 0.5
for a in range(nr):
	for b in range(nc):
		c[28 * a:28 * a + 28,28 * b:28 * b + 28] = x[np.random.choice(len(x))]
c*=255
cv2.imwrite('statistic/rsample.png',c)