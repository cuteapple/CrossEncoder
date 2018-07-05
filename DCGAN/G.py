import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation,Input,Add
from keras_contrib.layers.normalization import InstanceNormalization


def new_G(input_shape):
	i = Input(input_shape)
	x = Dense(256 * 7 * 7, activation="relu", input_shape=input_shape)(i)
	x = Reshape((7, 7, 256))(x)
	x = InstanceNormalization()(x)
	x = Conv2D(256, kernel_size=3, padding="same", activation='relu')(x)
	x = InstanceNormalization()(x)
	
	x = UpSampling2D()(x)
	
	x1 = Conv2D(256, kernel_size=3, padding="same", activation='relu')(x)
	x = Add()([x,x1])
	x = Conv2D(128, kernel_size=3, padding="same", activation='relu')(x)
	x = InstanceNormalization()(x)
	
	x = UpSampling2D()(x)

	x1 = Conv2D(128, kernel_size=3, padding="same", activation='relu')(x)
	x = Add()([x,x1])
	x = Conv2D(64, kernel_size=3, padding="same", activation='relu')(x)
	x = InstanceNormalization()(x)
	
	x1 = Conv2D(64, kernel_size=3, padding="same", activation='relu')(x)
	x = Add()([x,x1])
	x = Conv2D(32, kernel_size=3, padding="same", activation='relu')(x)
	x = InstanceNormalization()(x)

	x = Conv2D(1, kernel_size=3, padding="same")(x)
	x = Activation('sigmoid')(x)

	return Model(i,x,'G')

def z(batch_size,length):
	def g():
		answer = np.eye(10)[np.random.choice(10,batch_size)]
		z = np.random.normal(size=(batch_size,length))
		z[:,0:10] = answer
		return z,answer
	while True:
		yield g()

if __name__ == '__main__':

	from keras.utils import plot_model
	plot_model(new_G((20,)), show_shapes=True, to_file='G.png')
	raise SystemExit(0)


	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-s","--steps", default=64, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="G.h5", type=str)
	parser.add_argument("-dp","--discriminator_path", default="D.h5", type=str)
	args = parser.parse_args()
	print(args)

	output_shape = (28,28,1)
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