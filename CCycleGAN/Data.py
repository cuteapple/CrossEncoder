import cv2
import numpy as np
import glob

test_folder = 'x2photo/test'
train_folder = 'x2photo/train'
labels = ['cezanne','monet','photo','ukiyoi','vangogh']

I = np.eye(len(labels))
label_one_hot = { name:I[index] for index,name in enumerate(label_name) }

def one_hot(label_name):
	return label_one_hot[label_name]

def read_images(fileNames):
	for file in fileNames:
		im = cv2.imread(file)
		if im is None:
			print('warn : "{}" is not image'.format(file))
			continue
		else:
			yield im

def normalize(im):
	return im.astype(float)/255

def load_collection(root):
	D={}
	for name in label_name:
		print('reading {}'.format(name))
		files = glob.glob('{}/{}/*'.format(root,name))
		imgs = images(files)
		D[name] = list(normalize(im) for im in imgs)
	return D

train = load_collection(train_folder)
test = load_collection(test_folder)


