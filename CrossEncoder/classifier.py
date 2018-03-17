import keras
from keras.models import Sequential,Model
from keras.layers import *

input_shape=(28,28,1)
model_path = 'mnist_classifier.h5'
epochs = 50
batch_size = 128
num_classes = 10

import os
model_path = os.path.join(os.path.dirname(__file__), model_path)
del os


def new_model():
	model = Sequential(name='mnist_classifier')
	model.add(Conv2D(32, kernel_size=3, strides=1, activation='relu',input_shape=input_shape))
	model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.compile(optimizer='adadelta', loss='categorical_crossentropy' ,metrics=['accuracy'])
	return model

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
	model = new_model()
	try:
		print('loading {}'.format(model_path))
		model.load_weights(model_path)
	except (OSError,ValueError) as e:
		if not new_on_fail:
			print('load weights failed')
			raise SystemExit()
		else:
			print('load weights failed, recreate')
	return model

def train_model(model):
	(x_train,y_train),(x_test,y_test) = load_mnist()
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(x_test, y_test))

def save_model(model):
	print('saving {}'.format(model_path))
	model.save_weights(model_path)

def main():
	model = load_model(new_on_fail=True)
	train_model(model)
	save_model(model)

if __name__ == '__main__':
	main()

