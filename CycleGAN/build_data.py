import cv2
import numpy as np
from collections import namedtuple
import glob
import os

test_folder = 'x2photo/test'
train_folder = 'x2photo/train'
label_name = ['cezanne','monet','photo','ukiyoi','vangogh']
output_folder = 'preprocess-data'

label_count = len(labels_name)
I = np.eye(len(labels))
label_one_hot = { I[index]:name for index,name in enumerate(labels) }

def images(fileName_iterator):
	for file in fileName_iterator:
		im = cv2.imread(file)
		if im is None:
			print('WARNNING : {{{}}} is not image'.format(file))
			continue
		else:
			yield im

def normalize(im):
	return im.astype(np.float)/255

def load_collection(root):
	D={}
	for name in label_name:
		files = glob.glob('{}/{}/*'.format(root,name))
		imgs = images(files)
		images = (normalize(im) for im in imgs)
	return D

train = load_collection(train_folder)
test = load_collection(test_folder)


