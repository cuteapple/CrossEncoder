import keras
import numpy as np
import Dataset
from Dataset import nnoise, nclass

def data(batch_size):
	x = np.zeros((batch_size,nnoise+nclass))
	def g():
		x[:,:nclass] = y = np.eye(nclass)[np.random.choice(nclass,batch_size)]
		x[:,nclass:] = np.random.normal(size=(batch_size,nnoise))
		return x,y
	while True:
		yield g()

def new_G():	
	from keras.models import Sequential,Model
	from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation
	from keras_contrib.layers.normalization import InstanceNormalization
	return Sequential(name='G',
		layers=[Dense(128 * 7 * 7,input_shape=(nnoise+nclass,)),
			Activation("relu"),
			InstanceNormalization(),
			Reshape((7, 7, 128)),
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
	parser.add_argument("-e","--epochs", default = 200, type = int)
	parser.add_argument("-s","--steps", default = 64, type=int)
	parser.add_argument("-b","--batch-size", default = 128, type = int)
	parser.add_argument("-p","--path", default="G.h5", type=str)
	args = parser.parse_args()
	print('args :',args)
	
	print('loading D ...')
	d = keras.models.load_model('D.h5')
	d.trainable = False
	
	print('loading G ...')
	try:
		from keras_contrib import *
		g = keras.models.load_model(args.path)
	except:
		print('failed')
		g = new_G()
	else:
		print('success')

	print('linking G & D ...')
	input = keras.layers.Input((Dataset.nnoise+Dataset.nclass,))
	model = keras.models.Model(input,d(g(input)))
	model.compile(optimizer='adadelta',loss='mse', metrics=['accuracy'])

	print('training ...')
	
	model.fit_generator(data(args.batch_size), steps_per_epoch = args.steps, epochs=args.epochs)
		
	print('saving ...')
	g.save(args.path)
	d.save('DG.h5')