import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation,Input
from keras_contrib.layers.normalization import InstanceNormalization


def new_G(input_shape):
	return Sequential(name='G',
		layers=[Dense(128 * 4 * 4, activation="relu", input_shape=input_shape),
			Reshape((4, 4, 128)),
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
			UpSampling2D(),
			Conv2D(32, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			Conv2D(32, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			Conv2D(3, kernel_size=3, padding="same"),
			Activation("sigmoid")])


def z(batch_size,length):
	def g():
		answer = np.eye(10)[np.random.choice(10,batch_size)]
		z = np.random.normal(size=(batch_size,length))
		z[:,0:10] = answer
		return z,answer
	while True:
		yield g()

if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-s","--steps", default=64, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="G.h5", type=str)
	parser.add_argument("-dp","--discriminator_path", default="D.h5", type=str)
	args = parser.parse_args()
	print(args)

	output_shape = (32,32,3)
	z_len = 20
	input_shape = (z_len,)
	
	print(f'loading G at {args.path} ...')
	try:
		g = keras.models.load_model(args.path)
	except:
		print('fail, creating new')
		g = new_G(input_shape)
	else:
		print('success')
	
	print(f'loading D at {args.discriminator_path} ...')
	d = keras.models.load_model(args.discriminator_path)
	d.trainable = False

	print('linking G & D ...')
	input = Input(input_shape)
	m = Model(input,d(g(input)))
	m.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])

	print('training ...')
	m.fit_generator(z(args.batch_size,z_len), steps_per_epoch = args.steps, epochs=args.epochs)

	print('saving ...')
	g.save(args.path)