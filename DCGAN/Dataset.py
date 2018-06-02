import keras
import numpy as np

nclass = 10
nnoise = 10
reals = np.ones
facks = np.zeros
real_value = 1
fack_value = 0

def load_mnist():
	'''load mnist, normalize x, categorical y, return (x,y),(tx,ty)'''
	from keras.datasets import mnist
	(x, y), (tx, ty) = mnist.load_data()
	x = x.astype('float32').reshape(-1,28,28,1) / 255
	tx = x.astype('float32').reshape(-1,28,28,1) / 255
	y = keras.utils.to_categorical(y, nclass)
	ty = keras.utils.to_categorical(ty, nclass)
	return (x,y),(tx,ty)

def add_noise(images,scalar=1,mean=0,sigma=0.5):
	'''add random block of gaussian noise inplace'''
	x,y,deep = images.shape[1:]

	for i in range(len(images)):
		ax = np.random.randint(1,x)
		ay = np.random.randint(1,y)
		noise = np.random.normal(mean, sigma, size=(ax,ay,deep)) * scalar
		dx = np.random.randint(0, x - (ax - 1))
		dy = np.random.randint(0, y - (ay - 1))
		images[i, dx:dx + ax, dy:dy + ay] += noise

def ZData(batch_size):
	def g():
		c = np.eye(nclass)[np.random.choice(nclass,batch_size)]
		z = np.random.normal(size=(batch_size,nnoise))
		#r = np.random.binomial(size=batch_size, n=1, p=0.1) # assume real = 0 fack = 1
		r = reals(batch_size)
		return {'i_class':c,'i_noise':z},{'o_class':c}
	while True:
		yield g()