import dcgan
import numpy as np
import cv2

_,G,D = dcgan.load_model(False)

def Q():
	answer = np.eye(10)
	z = np.random.normal(size=(10,*dcgan.input_shape))
	z[:,:10] = answer
	return z

margin = 1
canvas = np.ones((28*2+3,28*5+6,1),np.uint8)*128


for n in range(10):
	res = G.predict(Q())
	for i,im in enumerate(res):
		nx = i//5
		ny = i%5
		x = nx*29+1
		y = ny*29+1
		canvas[x:x+28,y:y+28]=im*255
	cv2.imwrite('o/sample{}.png'.format(n),canvas)