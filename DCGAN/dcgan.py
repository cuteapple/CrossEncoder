import keras
from keras.models import Sequential
from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation
from keras_contrib.layers.normalization import InstanceNormalization 
import classifier

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-e","--epochs", default=1000, type=int)
parser.add_argument("-b","--batch_size", default=128, type=int)
args = parser.parse_args()

output_shape = (28,28,1)
input_shape = (100,)
model_path = 'mnist_generator.h5'
epochs = args.epochs
batch_size = args.batch_size

print(epochs,batch_size,model_path)

del parser
del args
del argparse


def new_generator():
	return Sequential(name='G',
		layers=[
			Dense(128 * 7 * 7, activation="relu", input_shape=input_shape),
			Reshape((7, 7, 128)),
			InstanceNormalization(),
			UpSampling2D(),
			Conv2D(128, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			UpSampling2D(),
			Conv2D(64, kernel_size=3, padding="same"),
			Activation("relu"),
			InstanceNormalization(),
			Conv2D(1, kernel_size=3, padding="same"),
			Activation("sigmoid")])

def new_model():
	''' load D, create G, return GAN,G,D'''

	D = classifier.load_model(new_on_fail=False)
	D.trainable = False
	G = new_generator()

	input = keras.layers.Input(input_shape)
	model = keras.models.Model(input,D(G(input)))
	model.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])

	return model,G,D

def load_model(new_on_fail=True):
	m,G,D = new_model()
	try:
		print('loading {}'.format(model_path))
		G.load_weights(model_path)
	except (OSError,ValueError) as e:
		if not new_on_fail:
			print(str(e))
			print('load weights failed, terminate')
			raise SystemExit()
		else:
			print(str(e))
			print('load weights failed, recreate')
	return m,G,D


def z_generator():
	import numpy as np
	def g():
		answer = np.eye(10)[np.random.choice(10,batch_size)]
		z = np.random.normal(size=(batch_size,*input_shape))
		z[:,0:10] = answer
		return z,answer
	while True:
		yield g()

def train_model(models):
	model = models[0]
	model.fit_generator(z_generator(),steps_per_epoch=100,epochs=epochs)

def save_model(models):
	G = models[1]
	print('saving {}'.format(model_path))
	G.save_weights(model_path)

def plot():
	from keras.utils import plot_model
	plot_model(model, to_file='model.png',show_shapes=True)
	plot_model(G, to_file='G.png',show_shapes=True)
	plot_model(D, to_file='D.png',show_shapes=True)

def main():
	models = load_model(new_on_fail=True)
	train_model(models)
	save_model(models)

if __name__ == '__main__':
	main()