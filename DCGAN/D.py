import keras
import numpy as np
import Dataset

def D():
	from keras.models import Sequential,Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout,LeakyReLU
	return Sequential(name='D',
		layers=[Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=(28,28,1)),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(64, kernel_size=3, strides=2,padding='same'),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(128, kernel_size=3, strides=2, padding='same'),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(256, kernel_size=3, strides=2, padding='same'),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(256, kernel_size=3, strides=1, padding='same'),
			LeakyReLU(),
			Flatten(),
			Dense(2048),
			LeakyReLU(),
			Dense(1024),
			LeakyReLU(),
			Dense(10)])

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-bnr","--batch_noise_ratio", default=0.6, type=float)
	parser.add_argument("-sy","--noisy_y_scalar", default=0.5, type=float)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	args = parser.parse_args()
	print('args',args)

	
	print('loading from {} ... '.format(args.path), end = '')
	try: 
		dc = keras.models.load_model(args.path)
	except:
		print('fail')
		dc = D()
		dc.compile(optimizer='adadelta', loss='mse')
	else:
		print('success')
	
	print('prepare data ...')

	(ox,oy),_ = Dataset.load_mnist()
	b = args.batch_size
	bn = int(args.batch_size*args.batch_noise_ratio)
	def data():
		choice = np.random.randint(len(ox),size=args.batch_size)
		x = ox[choice]
		y = oy[choice]
		Dataset.add_noise(x[:bn])
		y[:bn] *= args.noisy_y_scalar
		return x,y

	def data_g():
		while True:
			yield data()

	print('training ... ')
	dc.fit_generator(data_g(),steps_per_epoch=128,epochs=args.epochs)
	
	print('saving ...')
	dc.save(args.path)