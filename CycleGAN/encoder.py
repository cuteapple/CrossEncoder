import keras
import classifier

number_of_class = 5
input_shape=(256,256,3)
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
	from keras.models import Sequential,Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout, LeakyReLU,UpSampling2D,Reshape,Lambda,Input,Concatenate

	layers = [
		Lambda(lambda x:x/255.0,input_shape=input_shape), #temp normalize until keras fix 
		Conv2D(16,3,strides=2,padding='same'),
		LeakyReLU(0.2),
		Dropout(0.25),
		Conv2D(32,3,strides=2,padding='same'),
		LeakyReLU(0.2),
		Dropout(0.5),
		Conv2D(64,3,strides=2,padding='same'),
		LeakyReLU(0.2),
		Dropout(0.8),

		UpSampling2D(),
		Conv2D(64,3,strides=1,padding='same'),
		LeakyReLU(),
		UpSampling2D(),
		Conv2D(32,3,strides=1,padding='same'),
		LeakyReLU(),
		UpSampling2D(),
		Conv2D(16,3,strides=1,padding='same'),
		LeakyReLU(),
		Conv2D(3,3,strides=1,padding='same'),
		Lambda(lambda x:x*255.0),
		]
	model = Sequential(layers=layers)
	return model

def new_model(compile = True):
	G = new_G()
	D = classifier.load_model(for_test=True)
	D.trainable=False
	i = keras.layers.Input(input_shape)
	GAN = keras.models.Model(i,D(G(i)))

	if compile:
		GAN.compile(optimizer='RMSProp', loss='mse' ,metrics=['accuracy'])
	return GAN,G,D

def dataGenerator(path):
	from keras.preprocessing import image
	import numpy as np
	imG = image.ImageDataGenerator(data_format = 'channels_last')
	imG = imG.flow_from_directory(path, class_mode='categorical',target_size = input_shape[:2], batch_size = batch_size)
	while True:
		images, labels = next(imG) # ([batch_size,img],[batch_size,label one hot])
		#random_labels = np.eye(number_of_class)[np.random.choice(number_of_class,images.shape[0])]
		random_labels = np.array([[0,0,1,0,0]]*images.shape[0])
		yield images, random_labels

def save_model(G):
	print('saving {}'.format(model_path))
	G.save_weights(model_path)

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
	return GAN,G,D

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

