import keras

input_shape=(28,28,1)
model_path = 'mnist_autoencoder.h5'
epochs = 50
batch_size = 128

import os
model_path = os.path.join(os.path.dirname(__file__), model_path)
del os

def new_model():
	'''return : (autoencoder, encoder, decoder) tuple'''
	from keras.models import Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout,Input,Add,UpSampling2D,ZeroPadding2D

	x = Input(shape=input_shape)
	z = x
	z = Conv2D(filters=16, kernel_size=7, strides=2, activation='relu', padding='same')(z)
	z = Conv2D(filters=32, kernel_size=5, strides=2, activation='relu', padding='same')(z)
	z = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(z)
	Encoder = Model(x,z,name='Encoder')

	z = Input(shape = Encoder.output_shape[1:])
	y = Conv2D(32, (3, 3), activation='relu', padding='same')(z)
	y = UpSampling2D((2, 2))(y)
	y = Conv2D(16, (3, 3), activation='relu', padding='same')(y)
	y = UpSampling2D((2, 2))(y)
	y = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
	#y_res = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
	#y = Add()([y, y_res])
	#y = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
	Decoder = Model(z,y,name='Decoder')

	x = Input(shape=input_shape)
	y = Decoder(Encoder(x))
	AutoEncoder = Model(x,y,name='AutoEncoder')
	AutoEncoder.compile(optimizer='adadelta', loss='mse')

	from keras.utils import plot_model
	plot_model(AutoEncoder, to_file='model.png',show_shapes=True)
	plot_model(Encoder, to_file='model_encoder.png',show_shapes=True)
	plot_model(Decoder, to_file='model_decoder.png',show_shapes=True)

	return AutoEncoder,Encoder,Decoder

loaded_mnist = False
def load_mnist():
	global loaded_mnist
	if not loaded_mnist:
		from keras.datasets import mnist
		def transform(x):
			return x.astype('float32').reshape(-1,*input_shape)/255

		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x_train = transform(x_train)
		x_test = transform(x_test)
		y_train = keras.utils.to_categorical(y_train, 10)
		y_test = keras.utils.to_categorical(y_test, 10)
		loaded_mnist = (x_train,y_train),(x_test,y_test)
	return loaded_mnist

def load_model(new_on_fail=True):
	m = new_model()
	try:
		print('loading {}'.format(model_path))
		m[0].load_weights(model_path)
	except (OSError,ValueError) as e:
		if not new_on_fail:
			raise
		else:
			print(str(e))
			print('load weights failed, recreate')
	return m

def train_model(model):
	(x_train,_),(x_test,_) = load_mnist()
	model.fit(x_train, x_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(x_test, x_test))

def save_model(model):
	print('saving {}'.format(model_path))
	model.save_weights(model_path)


def main():
	model,enc,dec = load_model(new_on_fail=True)
	train_model(model)
	save_model(model)

if __name__ == '__main__':
	main()

