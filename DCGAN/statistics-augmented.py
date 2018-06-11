import numpy as np
import cv2

import D
#(x,y) = next(D.data(40,0.6,1.0))
(x,y) = next(D.data(200000,1.0,1.0))
x*=255
x = x.reshape(-1,28,28)
y = np.argmax(y,axis=1)

xn = [np.zeros((28,28))] * 10
for n in range(10):
	xn[n] = x[y == n]

def S(f,name):
	z = np.zeros((10,28,28))

	for n in range(10):
		z[n] = f(xn[n])
	
	for n in range(10):
		cv2.imwrite('statistic/{}.{}.png'.format(name,n),z[n])

	za = f(z)
	cv2.imwrite('statistic/{}.all.png'.format(name),za)

	#zip
	canvas = np.zeros((28,28 * 11))
	for n in range(10):
		canvas[:,28 * n:28 * n + 28] = z[n]
	canvas[:,28 * 10:] = za
	cv2.imwrite('statistic/{}.z.png'.format(name),canvas)

S(lambda a: np.max(a,axis=0),'aug.max')
S(lambda a: np.min(a,axis=0),'aug.min')
S(lambda a: np.mean(a,axis=0),'aug.mean')

canvas = np.zeros((28*4,28*10))
for a in range(4):
	for b in range(10):
		canvas[a*28:(a+1)*28, b*28:(b+1)*28] = x[a*10+b]

cv2.imwrite('statistic/arg.sample.png',canvas)