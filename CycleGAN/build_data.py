import cv2
import numpy as np
import glob

test_folder = 'x2photo/test'
train_folder = 'x2photo/train'
label_name = ['cezanne','monet','photo','ukiyoi','vangogh']

label_count = len(label_name)
I = np.eye(label_count)
label_one_hot = { name:I[index] for index,name in enumerate(label_name) }

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
		print('reading {}'.format(name))
		files = glob.glob('{}/{}/*'.format(root,name))
		imgs = images(files)
		D[name] = list(normalize(im) for im in imgs)
	return D

train = load_collection(train_folder)
test = load_collection(test_folder)


