import keras
from keras_contrib import *
import numpy as np
import os
import Dataset
import G

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
d = keras.models.load_model('D.h5')
z,y = next(G.data(256))
p = g.predict(z)
c = d.predict(p)

with open("{}/output.txt".format(o), "w") as file:
	for i,o in zip(z,c):
		print(f'{[*i]}',file=file)
		print(f'{[*o]}',file=file)

sample = [np.zeros((28,28))]*10

import cv2
for i,(p,y) in enumerate(zip(p,y)):
	y = int(np.dot(y[:10],np.arange(10)))
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
