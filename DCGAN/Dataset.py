import keras
import numpy as np

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

nnoise = 10
real_value = 0
fack_value = 1

reals = np.zeros
facks = np.ones
#return np.ones(shape)*real_value
#return np.ones(shape)*fack_value


def ZData(batch_size):
	#r = reals(batch_size)
	def g():
		c = np.eye(nclass)[np.random.choice(nclass,batch_size)]
		z = np.random.normal(size=(batch_size,nnoise))
		r = np.random.binomial(size=batch_size, n=1, p=0.1) # assume real = 0 fack = 1
		z[:,0] = r
		return [c,z],[c,r]
	while True:
		yield g()

class NoizyData:
	'''noizy mnist data'''
	def __init__(self, noise_sigma=1.0, noise_scaler=0.5,noise_area=(5,5)):
		self.noise_mean = 0.0
		self.noise_area = noise_area
		self.noise_scaler = noise_scaler
		self.noise_sigma = noise_sigma

		(x,y),_ = self.load_mnist()

		self.x = x
		self.y = y
		self.r = reals(len(y))
	
	def addnoise(self,x):
		noise_mean = self.noise_mean
		noise_sigma = self.noise_sigma
		noise_scaler = self.noise_scaler
		noise_area = self.noise_area

		ax,ay = noise_area

		for i in range(len(x)):
			noise = np.random.normal(noise_mean, noise_sigma, size=(ax,ay,1)) * noise_scaler
			dx = np.random.randint(28 - (ax - 1))
			dy = np.random.randint(28 - (ay - 1))
			x[i, dx:dx + ax, dy:dy + ay] += noise

	def train_batch(self,nreal,nfake):
		choice = np.random.randint(len(self.x),size = nreal+nfake)
		x = self.x[choice]
		y = self.y[choice]
		r = facks(nreal+nfake)
		r[:nreal]=real_value
		self.addnoise(x[nreal:])
		return x,[y,r]
	
	def test(self):
		return self.x,[self.y, self.r]

	@staticmethod
	def transform(x):
		return x.astype('float32').reshape(-1,28,28,1) / 255

	@staticmethod
	def load_mnist():
		from keras.datasets import mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = NoizyData.transform(x_train)
		x_test = NoizyData.transform(x_test)
		y_train = keras.utils.to_categorical(y_train, nclass)
		y_test = keras.utils.to_categorical(y_test, nclass)
		return (x_train,y_train),(x_test,y_test)

