import keras
import numpy as np
import Dataset

def new_G(input_length):
	
	from keras.models import Sequential,Model
	from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,LeakyReLU
	from keras_contrib.layers.normalization import InstanceNormalization
	return Sequential(name='G',
		layers=[Dense(128 * 7 * 7,input_shape=(input_length,)),
			LeakyReLU(),
			InstanceNormalization(),
			Reshape((7, 7, 128)),
			UpSampling2D(),
			Conv2D(128, kernel_size=3, padding="same"),
			LeakyReLU(),
			InstanceNormalization(),
			Conv2D(128, kernel_size=3, padding="same"),
			LeakyReLU(),
			InstanceNormalization(),
			UpSampling2D(),
			Conv2D(64, kernel_size=3, padding="same"),
			LeakyReLU(),
			InstanceNormalization(),
			Conv2D(64, kernel_size=3, padding="same"),
			LeakyReLU(),
			InstanceNormalization(),
			Conv2D(1, kernel_size=3, padding="same",activation='sigmoid')])

if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default = 200, type = int)
	parser.add_argument("-s","--steps", default = 64, type=int)
	parser.add_argument("-b","--batch-size", default = 32, type = int)
	parser.add_argument("-p","--path", default="G.h5", type=str)
	#parser.add_argument("-mp","--model-path", default="M.h5", type=str)
	args = parser.parse_args()
	print('args :',args)
	
	print('loading D ...')
	dn = keras.models.load_model('noisy.h5')
	dn.trainable = False
	#load other d ...
	i = keras.layers.Input((28,28,1))
	d = keras.models.Model(i,dn(i),name='D')
	d.trainable = False

	print('loading G ...')
	try:
		from keras_contrib import *
		g = keras.models.load_model(args.path)
	except:
		print('failed')
		g = new_G(10)
	else:
	   print('success')


	print('linking G & D ...')
	input = keras.layers.Input((10,))
	model = keras.models.Model(input, d(g(input)))
	model.compile(optimizer='adadelta',loss='mse',loss_weights=[1])

	print('training ...')
	z = Dataset.ZData(args.batch_size,10)
	model.fit_generator(z,
		steps_per_epoch = args.steps,
		epochs=args.epochs)

	print('saving ...')
	g.save(args.path)
	#model.save(args.model_path)