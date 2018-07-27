import G
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--path", default="G.h5", type=str)
parser.add_argument("-o","--output", default="o", type=str)
args = parser.parse_args()
print(args)

o = args.output
try:
	os.makedirs(o)
except:
	pass

import keras
from keras_contrib.layers.normalization import InstanceNormalization
print('loading model ...')
g = keras.models.load_model('G.h5')

z,y = next(G.z(100,20))
x = g.predict(z)

sample = [np.zeros((32,32,3))]*10

import cv2
for i,(p,y) in enumerate(zip(x,y)):
	y = int(np.dot(y,np.arange(10)))
	p = p.reshape(32,32,3)*255
	sample[y] = p
	cv2.imwrite('{}/{}-{}.png'.format(o,y,i),p)

margin = 1
canvas = np.ones((32*2+3,32*5+6,3),np.uint8)*128
for i,im in enumerate(sample):
	nx = i//5
	ny = i%5
	x = nx*33+1
	y = ny*33+1
	canvas[x:x+32,y:y+32,:]=im
	cv2.imwrite('{}/sample.png'.format(o),canvas)