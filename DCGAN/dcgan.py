import keras
from keras.layers.normalization import BatchNormalization
import classifier

output_shape = (28,28,1)
input_shape = (100,)
model_path = 'mnist_generator.h5'
epochs = 50
batch_size = 32


def new_generator():

	from keras.models import Sequential
	from keras.layers import Dense,Reshape,UpSampling2D,Conv2D,Activation

	return Sequential(name='generator',
		layers=[
			Dense(128 * 7 * 7, activation="relu", input_shape=input_shape),
			Reshape((7, 7, 128)),
			BatchNormalization(momentum=0.8),
			UpSampling2D(),
			Conv2D(128, kernel_size=3, padding="same"),
			Activation("relu"),
			BatchNormalization(momentum=0.8),
			UpSampling2D(),
			Conv2D(64, kernel_size=3, padding="same"),
			Activation("relu"),
			BatchNormalization(momentum=0.8),
			Conv2D(1, kernel_size=3, padding="same"),
			Activation("sigmoid")
		])

def new_model():
	'''return GAN,G,D'''
	D = classifier.load_model(new_on_fail=False)
	D.trainable=False
	G = new_generator()

	input = keras.layers.Input(input_shape)
	model = keras.models.Model(input,D(G(input)))
	model.compile(optimizer='adadelta',loss='mse',metrics=['accuracy'])

	return model,G,D

def load_model(new_on_fail=True):
	models = new_model()
	try:
		print('loading {}'.format(model_path))
		models[0].load_weights(model_path)
	except (OSError,ValueError) as e:
		if not new_on_fail:
			raise
		else:
			print(str(e))
			print('load weights failed, recreate')
	return models


def z_generator():
	import numpy as np
	def g():
		answer = np.eye(10)[np.random.choice(10,batch_size)]
		z = np.random.normal(size=(batch_size,*input_shape))
		z[:,0:10]=answer
		return z,answer
	while True:
		yield g()

def train_model(model):
	pass
	#model.fit_generator(z_generator(),steps_per_epoch=100,epochs=epochs)

def save_model(model):
	print('saving {}'.format(model_path))
	model.save_weights(model_path)


def main():
	model,G,D = load_model(new_on_fail=True)
	
	from keras.utils import plot_model
	plot_model(model, to_file='model.png',show_shapes=True)
	plot_model(G, to_file='G.png',show_shapes=True)
	plot_model(D, to_file='D.png',show_shapes=True)
	
	train_model(model)
	save_model(model)

if __name__ == '__main__':
	main()