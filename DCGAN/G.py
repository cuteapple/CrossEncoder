import keras
import numpy as np
import D
from Dataset import ZData as z

from keras.models import Sequential,Model
from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation,Input
from keras_contrib.layers.normalization import InstanceNormalization

def new_G(input_shape):
	return Sequential(name='G',
		layers=[Dense(128 * 7 * 7, activation="relu", input_shape=input_shape),
			Reshape((7, 7, 128)),
			InstanceNormalization(),
			UpSampling2D(),
			Conv2D(128, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			Conv2D(128, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			UpSampling2D(),
			Conv2D(64, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			Conv2D(64, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			Conv2D(1, kernel_size=3, padding="same"),
			Activation("sigmoid")])

if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=100, type=int)
	parser.add_argument("-s","--steps", default=64, type=int)
	parser.add_argument("-b","--batch-size", default=128, type=int)
	parser.add_argument("-p","--path", default="G.h5", type=str)
	parser.add_argument("-dp","--discriminator-path", default="D.h5", type=str)
	args = parser.parse_args()
	print('args :',args)

	output_shape = (28,28,1)
	z_len = 20
	input_shape = (z_len,)
	
	
	print('loading D ...')
	d = D.new_D()
	d.load_weights(args.discriminator_path)
	d.trainable = False

	print('loading G ...')
	g = new_G(input_shape)
	try: g.load_weights(args.path)
	except: print('failed')
	else: print('success')


	print('linking G & D ...')
	input = Input(input_shape)
	m = Model(input,d(g(input)))
	m.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])

	print('training ...')
	m.fit_generator(z(args.batch_size,z_len),
		steps_per_epoch = args.steps,
		epochs=args.epochs)

	print('saving ...')
	g.save_weights(args.path)