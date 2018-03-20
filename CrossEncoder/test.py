import keras

from keras.datasets import mnist
from keras.models import load_model

(x, y), _ = mnist.load_data()
x=x.astype('float')/255
x=x.reshape((-1,28,28,1))
x_n = *(x[y==i] for i in range(10)),
X = x_n

import numpy as np
import cv2

M11 = load_model('M11.h5',compile=False)
M15 = load_model('M15.h5',compile=False)
M51 = load_model('M51.h5',compile=False)
M55 = load_model('M55.h5',compile=False)

pretty = lambda im: im.reshape((28,28))*255

import os
os.makedirs('test/1')
p11 = M11.predict(X[3])
p15 = M15.predict(X[3])
for i,(x1,y1,y5) in enumerate(zip(X[3],p11,p15)):
	cv2.imwrite('test/1/{}-1{}.png'.format(i,'o'),pretty(x1))
	cv2.imwrite('test/1/{}-2{}.png'.format(i,'r'),pretty(y1))
	cv2.imwrite('test/1/{}-3{}.png'.format(i,'t'),pretty(y5))

os.makedirs('test/5')
p55 = M55.predict(X[5])
p51 = M51.predict(X[5])
for i,(x1,y1,y5) in enumerate(zip(X[5],p55,p51)):
	cv2.imwrite('test/5/{}-1{}.png'.format(i,'o'),pretty(x1))
	cv2.imwrite('test/5/{}-2{}.png'.format(i,'r'),pretty(y1))
	cv2.imwrite('test/5/{}-3{}.png'.format(i,'t'),pretty(y5))
	