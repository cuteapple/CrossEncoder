import dcgan
import numpy as np
_,G,D = dcgan.load_model(False)

def zG():
	for items in dcgan.z_generator():
		for item in items[0]:
			yield item.reshape((1,-1)), np.dot(np.arange(10),item[0:10])

zG = zG()

import cv2
for i in range(100):
	x = next(zG)
	p = G.predict(x[0])
	c = D.predict(p)
	print(x[1])
	print(c)
	p = p.reshape(28,28)*255
	cv2.imwrite('o/{}-{}.png'.format(i,x[1]),p)
