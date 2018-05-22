import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation,Input
from keras_contrib.layers.normalization import InstanceNormalization 
from D import D


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


def z(batch_size,length):
	def g():
		answer = np.eye(10)[np.random.choice(10,batch_size)]
		z = np.random.normal(size=(batch_size,length))
		z[:,0:10] = answer
		z[10] = 0
		return z,answer
	while True:
		yield g()

if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=1000, type=int)
	parser.add_argument("-s","--steps", default=64, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="G.h5", type=str)
	parser.add_argument("-dp","--discriminator_path", default="D.h5", type=str)
	args = parser.parse_args()
	print(args)

	output_shape = (28,28,1)
	z_len = 20
	input_shape = (z_len,)
	
	print('loading G ...')
	g = new_G(input_shape)
	try: g.load_weights(args.path)
	except: print('load weight for G failed')
	
	print('loading D ...')
	d = D.Load(args.discriminator_path)
	d = d.model
	d.trainable = False

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