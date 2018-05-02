import keras
import numpy as np


class NoizyData:
	'''noizy mnist data'''
	def __init__(self,y_factor=1.0):
		
		(x,y),(tx,ty) = self.load_data()
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
		return x.astype('float32').reshape(-1,32,32,3) / 255
	
	@staticmethod
	def transform_inv(x):
		return x * 255

	@staticmethod
	def load_data():
		from keras.datasets import cifar10
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		x_train = NoizyData.transform(x_train)
		x_test = NoizyData.transform(x_test)
		y_train = keras.utils.to_categorical(y_train, 10)
		y_test = keras.utils.to_categorical(y_test, 10)
		return (x_train,y_train),(x_test,y_test)

def new_D():
	from keras.models import Sequential
	from keras.layers import Conv2D,Flatten,Dense,Dropout,Input
	model = Sequential(name='D-cifar10',
		layers=[Conv2D(32, kernel_size=3, padding='same', strides=1, activation='relu',input_shape=(32,32,3)),
			Conv2D(32, kernel_size=3, padding='same', strides=1, activation='relu'),
			Conv2D(48, kernel_size=3, padding='same', strides=1, activation='relu'),# as-is, expand
			Conv2D(48, kernel_size=3, padding='same', strides=1, activation='relu'),
			Conv2D(64, kernel_size=3, padding='same', strides=2, activation='relu'),# downsample, reduce
			Conv2D(64, kernel_size=3, padding='same', strides=1, activation='relu'),
			Conv2D(96, kernel_size=3, padding='same', strides=1, activation='relu'),# as-is, expand
			Conv2D(96, kernel_size=3, padding='same', strides=1, activation='relu'),
			Conv2D(128,kernel_size=3, padding='same', strides=2, activation='relu'),# downsample, reduce
			Conv2D(128,kernel_size=3, padding='same', strides=1, activation='relu'),
			Conv2D(256,kernel_size=3, padding='same', strides=2, activation='relu'),# downsample, as-is
			Flatten(),
			Dense(1024, activation='relu'),
			Dense(1024, activation='relu'),
			Dense(1024, activation='relu'),
			Dense(10)])
	return model

def train(self,data,epochs=200,batch_size=128):

	self.model.fit_generator(data.train_generator(batch_size),
		steps_per_epoch=data.x.shape[0] // batch_size,
		epochs=epochs,
		validation_data=data.test(),
		shuffle=False # shuffle inside generator
		)

if __name__ == "__main__":
	print('CCDCGAN-cifar10-v7.D.1')

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=256, type=int)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	parser.add_argument("-ny","--noise_sy", default=1.0, type=float)
	parser.add_argument("-plot", "--plot", action='store_true')
	args = parser.parse_args()
	print(args)
	
	
	D = new_D()

	if args.plot:
		from keras.utils import plot_model
		plot_model(D, to_file='D.png',show_shapes=True)
		raise SystemExit
	
	print('loading weights ... ',end='')
	try: D.load_weights(args.path)
	except: print('failed')
	else: print('success')
	

	print('loading data ... ', end='')
	data = NoizyData(args.noise_sy)
	print('finish')

	print('training ...')
	D.compile('adadelta','mse',['accuracy'])
	D.fit_generator(
		data.train_generator(args.batch_size),
		steps_per_epoch = data.x.shape[0] // args.batch_size,
		epochs = args.epochs,
		validation_data = data.test(),
		shuffle=False # shuffled inside generator
		)

	print('saving ...')
	D.save_weights(args.path)
