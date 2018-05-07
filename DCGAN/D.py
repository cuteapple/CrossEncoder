import keras
import numpy as np

shape = (28,28,1)

class NoizyData:
	'''noizy mnist data'''
	def __init__(self):
		
		(x,y),(tx,ty) = self.load_data()
		self.x = x
		self.y = y
		self.tx = tx
		self.ty = ty, np.zeros(ty.shape[0])
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
		x = np.clip(x + dx,0,1)

		r = scale #+ np.random.normal(0, .1, scale.shape)

		return x,[y,r]

	@staticmethod
	def transform(x):
		return x.astype('float32').reshape(-1,*shape) / 255
	
	@staticmethod
	def transform_inv(x):
		return x * 255

	@staticmethod
	def load_data():
		from keras.datasets import mnist
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = NoizyData.transform(x_train)
		x_test = NoizyData.transform(x_test)
		y_train = keras.utils.to_categorical(y_train, 10)
		y_test = keras.utils.to_categorical(y_test, 10)
		return (x_train,y_train),(x_test,y_test)

def new_D():
	from keras.models import Sequential,Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout,Input
	
	model = Sequential(name='d-pre',
		layers=[Conv2D(32, kernel_size=3, padding='same', strides=1, activation='relu',input_shape=shape),
			Dropout(.5),
			Conv2D(48, kernel_size=3, padding='same', strides=2, activation='relu',input_shape=shape),
			Dropout(.5),
			Conv2D(48, kernel_size=3, padding='same', strides=1, activation='relu',input_shape=shape),
			Dropout(.5),
			Conv2D(64, kernel_size=3, padding='same', strides=2, activation='relu',input_shape=shape),
			Dropout(.5),
			Flatten(),
			Dense(128, activation='relu')
			])
	
	x = Input(shape)
	s = model(x)
	classify = Dense(10,name = 'class')(s)
	real = Dense(1,name = 'real')(s)
	return Model([x],[classify,real],name='D')

if __name__ == "__main__":
	print('CCDCGAN-mnist-9')

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=256, type=int)
	parser.add_argument("-s","--step", default=0, type=int)
	parser.add_argument("-p","--path", default="D.h5", type=str)
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
	data = NoizyData()
	print('finish')

	print('training ...')
	D.compile('rmsprop','mse',loss_weights=[1,1],metrics=['accuracy'])
	D.fit_generator(data.train_generator(args.batch_size),
		steps_per_epoch = data.x.shape[0] // args.batch_size if args.step == 0 else args.step,
		epochs = args.epochs,
		shuffle=False # shuffled inside generator
		)

	print('saving ...')
	D.save_weights(args.path)
