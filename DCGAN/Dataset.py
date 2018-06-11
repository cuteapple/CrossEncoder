import keras
import numpy as np

nnoise = 10
nclass = 10

def load_mnist():
	'''load mnist, normalize x, categorical y, return (x,y),(tx,ty)'''
	from keras.datasets import mnist
	(x, y), (tx, ty) = mnist.load_data()
	x = x.astype('float32').reshape(-1,28,28,1) / 255
	tx = x.astype('float32').reshape(-1,28,28,1) / 255
	y = keras.utils.to_categorical(y, nclass)
	ty = keras.utils.to_categorical(ty, nclass)
	return (x,y),(tx,ty)

def add_noise(images,scalar=1,mean=0,std=0.5,area=(7,7)):
	'''add random block of gaussian noise inplace'''
	x,y,deep = images.shape[1:]
	for i in range(len(images)):
		ax,ay = area
		#ax = np.random.randint(1,x)
		#ay = np.random.randint(1,y)
		noise = np.random.normal(mean, std, size=(ax,ay,deep)) * scalar
		dx = np.random.randint(0, x - (ax - 1))
		dy = np.random.randint(0, y - (ay - 1))
		images[i, dx:dx + ax, dy:dy + ay] += noise