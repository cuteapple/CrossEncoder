import keras
from keras_contrib import *
import numpy as np
import os
import Dataset

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

g = keras.models.load_model(args.path)
i_noise = keras.layers.Input((10,),name='i_noise')
i_class = keras.layers.Input((Dataset.nclass,),name='i_class')
input = keras.layers.concatenate([i_noise,i_class])
model = keras.models.Model([i_noise,i_class], [g(input)])

z,y = next(Dataset.ZData(100))
y=y['o_class']
p = model.predict(z)

sample = [np.zeros((28,28))]*10

import cv2
for i,(p,y) in enumerate(zip(p,y)):
	y = int(np.dot(y[:10],np.arange(10)))
	#y = 1
	p = p.reshape(28,28)*255
	sample[y] = p
	cv2.imwrite('{}/{}-{}.png'.format(o,y,i),p)

margin = 1
canvas = np.ones((28*2+3,28*5+6),np.uint8)*128
for i,im in enumerate(sample):
	nx = i//5
	ny = i%5
	x = nx*29+1
	y = ny*29+1
	canvas[x:x+28,y:y+28]=im
	cv2.imwrite('{}/sample.png'.format(o),canvas)