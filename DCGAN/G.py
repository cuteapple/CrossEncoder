import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation,Input
from keras_contrib.layers.normalization import InstanceNormalization 
from D import new_D, shape

def new_G(input_shape):
	return Sequential(name='G',
		layers=[#input_shape

			Dense(128*4*4 , activation="relu", input_shape=input_shape),
			
			Reshape((4, 4, 128)),
			Conv2D(128, kernel_size=7, activation='relu', padding="same"),
			InstanceNormalization(),

			UpSampling2D(), # 8 8
			Conv2D(96, kernel_size=3, activation='relu', padding="same"),
			InstanceNormalization(),

			UpSampling2D(), # 16 16
			Conv2D(64, kernel_size=3, activation='relu', padding="same"),
			InstanceNormalization(),

			UpSampling2D(), # 32 32
			Conv2D(64, kernel_size=3, activation='relu'),
			InstanceNormalization(),
			Conv2D(64, kernel_size=3, activation='relu'),
			InstanceNormalization(),
			Conv2D(shape[-1], kernel_size=3, activation='relu', padding="same")
			])


def z(batch_size,length):
	def g():
		answer = np.eye(10)[np.random.choice(10,batch_size)]
		z = np.random.normal(size=(batch_size,length))
		z[:,0:10] = answer
		return z,[answer,np.zeros(len(answer))]
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
	parser.add_argument("-plot","--plot", action='store_true')
	args = parser.parse_args()
	print(args)

	z_len = 20
	input_shape = (z_len,)
	
	g = new_G(input_shape)
	
	if args.plot:
		from keras.utils import plot_model
		plot_model(g, to_file='G.png',show_shapes=True)
		raise SystemExit	


	print('loading G ... ',end='')
	try: g.load_weights(args.path)
	except: print('failed')
	else: print('success')
	
	print('loading D ... ')
	d = new_D()
	d.load_weights(args.discriminator_path) # this need success
	d.trainable = False

	print('linking G & D ... ')
	input = Input(input_shape)
	m = Model(input,d(g(input)))
	m.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])

	print('training ...')
	m.fit_generator(z(args.batch_size,z_len),
		steps_per_epoch = args.steps,
		epochs=args.epochs)

	print('saving ...')
	g.save_weights(args.path)