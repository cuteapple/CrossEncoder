import keras
import numpy as np

nclass = 10
nnoise = 10
real_value = 0
fack_value = 1

reals = np.zeros
facks = np.ones
#return np.ones(shape)*real_value
#return np.ones(shape)*fack_value


def ZData(batch_size):
	r = reals(batch_size)
	def g():
		c = np.eye(nclass)[np.random.choice(nclass,batch_size)]
		z = np.random.normal(size=(batch_size,nnoise))
		return [c,z],[c,r]
	while True:
		yield g()

class NoizyData:
	'''noizy mnist data'''
	def __init__(self, noise_sigma=1.0, noise_scaler=0.5,noise_area=(5,5)):
		noise_mean = 0.0

		ax = noise_area[0]
		ay = noise_area[1]
		(x,y),(tx,ty) = self.load_mnist()

		noisy_x = np.copy(x)

		for i in range(len(x)):
			noise = np.random.normal(noise_mean, noise_sigma, size=(ax,ay,1)) * noise_scaler
			dx = np.random.randint(28 - (ax - 1))
			dy = np.random.randint(28 - (ay - 1))
			noisy_x[i, dx:dx + ax, dy:dy + ay] += noise
					
		noisy_x = np.clip(noisy_x,0.0,1.0)

		noisy_y = np.copy(y)


		self.x = np.concatenate((x,noisy_x), axis=0)
		self.y = [np.concatenate((y,noisy_y), axis=0), np.concatenate((reals(len(y)),facks(len(noisy_y))), axis=0)]
		self.tx = tx
		self.ty = [ty,reals(len(ty))]

	def train(self):
		return self.x,self.y

	def test(self):
		return self.tx,self.ty

	@staticmethod
	def transform(x):
		return x.astype('float32').reshape(-1,28,28,1) / 255
	
	@staticmethod
	def transform_inv(x):
		return x * 255

	@staticmethod
	def load_mnist():
		from keras.datasets import mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = NoizyData.transform(x_train)
		x_test = NoizyData.transform(x_test)
		y_train = keras.utils.to_categorical(y_train, nclass)
		y_test = keras.utils.to_categorical(y_test, nclass)
		return (x_train,y_train),(x_test,y_test)

