import keras
import classifier

number_of_class = 5
input_shape=(256,256,3+number_of_class)
model_path = 'xphoto_G.h5'
train_data_folder = 'x2photo/train'
test_data_folder = 'x2photo/test'
epochs = 50
steps_per_epoch = 200
batch_size = 32

import os
model_path = os.path.join(os.path.dirname(__file__), model_path)
del os

def new_G():
	from keras.models import Sequential
	from keras.layers import Conv2D,Flatten,Dense,Dropout, LeakyReLU,UpSampling2D,Reshape
	import numpy

	preprocess_layers = [
		Dense(number_of_class*10,input_shape=(number_of_class,)),
		LeakyReLU(),
		Dense(numpy.prod(input_shape),activation='relu'),
		LeakyReLU(),
		Reshape(input_shape)
		]

	layers = [
		Conv2D(32,3,strides=1,padding='same',input_shape=input_shape),
		LeakyReLU(),
		Conv2D(32,3,strides=2,padding='same'), #1/2 128
		LeakyReLU(),
		Dropout(0.25),
		Conv2D(64,3,strides=1,padding='same'),
		LeakyReLU(),
		Conv2D(64,3,strides=2,padding='same'), #1/4 64
		LeakyReLU(),
		Conv2D(128,3,strides=1,padding='same'),
		LeakyReLU(),
		Conv2D(128,3,strides=2,padding='same'), #1/8 32
		LeakyReLU(),
		
		Conv2D(128,3,strides=1,padding='same'),
		
		UpSampling2D(),
		Conv2D(64,3,strides=1,padding='same'),
		LeakyReLU(),
		UpSampling2D(),
		Conv2D(32,3,strides=1,padding='same'),
		UpSampling2D(),
		Conv2D(16,3,strides=1,padding='same',activation = 'sigmoid'),
		Conv2D(3,3,strides=1,padding='same',activation = 'sigmoid'),
		]
	model = Sequential(layers=layers, name='xphoto_G')
	return model

def new_model(compile = True):
	G = new_G()
	D = classifier.load_model(for_test=True)
	D.trainable=False
	i = keras.layers.Input(input_shape)
	GAN= keras.models.Model(i,D(G(i)))
	if compile:
		GAN.compile(optimizer='RMSProp', loss='mse' ,metrics=['accuracy'])
	return GAN,G,D

def dataGenerator(path):
	from keras.preprocessing import image
	imG = image.ImageDataGenerator(data_format = 'channels_last')
	return imG.flow_from_directory(path, class_mode='categorical',target_size = input_shape[:2], batch_size = batch_size)

def save_model(model):
	print('saving {}'.format(model_path))
	model.save_weights(model_path)

def load_model(for_test = True):
	GAN,G,D = new_model(compile = not for_test)
	try:
		print('loading {}'.format(model_path))
		G.load_weights(model_path)
	except (OSError,ValueError) as e:
		if for_test:
			raise
		else:
			print('load weights failed, recreate')
	return model

def train_model(model):
	model.fit_generator(
		generator = dataGenerator(train_data_folder),
		validation_data = dataGenerator(test_data_folder),
		epochs = epochs,
		steps_per_epoch = steps_per_epoch,
		validation_steps = steps_per_epoch/10,
		verbose=1,
	)

def main():
	GAN,G,D = load_model(for_test = False)
	train_model(GAN)
	save_model(G)

if __name__ == '__main__':
	main()

