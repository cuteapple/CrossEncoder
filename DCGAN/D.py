import keras
import numpy as np


class NoizyData:
	'''noizy mnist data'''
	def __init__(self,y_factor=1.0):
		
		(x,y),(tx,ty) = self.load_mnist()
		self.x = x
		self.y = y
		self.tx = tx
		self.ty = ty
		self.sy = y_factor
		self.noise_mean = 0.0
		self.noise_sigma = 1.0
		self.choicepool = np.arange(self.x.shape[0])

	def train_generator(self,size):
		while True:
			yield self.train_batch(size)

	def train_batch(self,size):
		choice = np.random.choice(self.choicepool,size)
		x = self.x[choice]
		y = self.y[choice]

		scale = np.random.uniform(size=size)
		
		dx = np.random.normal(size=x.shape)
		dx = dx.reshape((size,-1)) * scale[:,None]
		dx = dx.reshape(x.shape)
		
		sy = (1 - scale)[:,None] * self.sy

		x = np.clip(x + dx,0,1)
		y = y * sy

		return x,y

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
		y_train = keras.utils.to_categorical(y_train, 10)
		y_test = keras.utils.to_categorical(y_test, 10)
		return (x_train,y_train),(x_test,y_test)


class D:
	def __init__(self):
		self.model = self.new_classifier()

	@staticmethod
	def new_classifier():
		from keras.models import Sequential
		from keras.layers import Conv2D,Flatten,Dense,Dropout,Input
		model = Sequential(name='mnist_classifier',
			layers=[Conv2D(32, kernel_size=3, strides=1, activation='relu',input_shape=(28,28,1)),
				Conv2D(64, kernel_size=3, strides=2, activation='relu'),
				Dropout(0.25),
				Flatten(),
				Dense(128, activation='relu'),
				Dropout(0.5),
				Dense(10)])
		return model
	
	@classmethod
	def Load(cls,path=None,or_new=False):
		inst = cls()
		if path is None:
			path = cls.default_path
		try:
			inst.load_weights(path)
		except:
			print('load weight fail')
			if not or_new:
				raise
		return inst

	def save_weights(self,path=None):
		if path is None:
			path = D.default_path
		self.model.save_weights(path)
	def load_weights(self,path):
		self.model.load_weights(path)

	def compile(self,optimizer='adadelta', loss='mse' ,metrics=['accuracy'],*args):
		self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics,*args)

	def train(self,data,epochs=200,batch_size=128):

		self.model.fit_generator(
			data.train_generator(batch_size),
			steps_per_epoch=data.x.shape[0]//batch_size,
			epochs=epochs,
			validation_data=data.test(),
			shuffle=False # shuffle inside generator
			)

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	parser.add_argument("-ny","--noise_sy", default=1.0, type=float)
	args = parser.parse_args()
	print(args)

	print('loading weights ...')
	d = D.Load(args.path,True)
	
	print('training ...')
	d.compile()
	data = NoizyData(args.noise_sy)
	d.train(data,epochs=args.epochs,batch_size=args.batch_size)

	print('saving ...')
	d.save_weights(args.path)
