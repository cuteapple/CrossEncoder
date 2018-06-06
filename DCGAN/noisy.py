import keras
import numpy as np
import Dataset

def data_generator(nreal,nnoisy):
	nbatch = nreal + nnoisy
	(x,_),_ = Dataset.load_mnist()
	y = Dataset.reals(nbatch)
	y[:nnoisy] = Dataset.noizy_value
	def impl():
		index = np.random.randint(len(x),size=nbatch)
		target = x[index]
		Dataset.add_noise(target[:nnoisy])
		return target
	while True:
		yield impl(),y

def new_model():
	from keras.models import Sequential,Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout,LeakyReLU,BatchNormalization
	return Sequential(name='noisy',
		layers=[Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=(28,28,1)),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.25),
			Conv2D(64, kernel_size=3, strides=2,padding='same'),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.25),
			Conv2D(128, kernel_size=3, strides=2, padding='same'),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.25),
			Conv2D(256, kernel_size=3, strides=2, padding='same'),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.25),
			Flatten(),
			Dense(1)])

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-s","--steps", default=64, type=int)
	parser.add_argument("-nr","--noisy_batch_ratio", default=0.7, type=float)
	parser.add_argument("-p","--path", default="noisy.h5", type=str)
	args = parser.parse_args()
	print('args',args)

	print('loading weights at {} ... '.format(args.path), end = '')
	try: 
		model = keras.models.load_model(args.path)
	except:
		print('fail')
		model = new_model()
		model.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
	else:
		print('success')
	
	print('prepare data ...')
	
	nb = args.batch_size
	nn = int(nb * args.noisy_batch_ratio)
	nr = nb - nn
	g = data_generator(nr,nn)

	print('training ... ')
	model.fit_generator(g,
		steps_per_epoch=args.steps,
		epochs=args.epochs)
	
	print('saving ...')
	model.save(args.path)
