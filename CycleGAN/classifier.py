import keras

input_shape=(256,256,3)
model_path = 'xphoto_classifier.h5'
train_data_folder = 'x2photo/train'
test_data_folder = 'x2photo/test'
epochs = 50
steps_per_epoch = 200
batch_size = 32
number_of_class = 5

import os
model_path = os.path.join(os.path.dirname(__file__), model_path)
del os

def new_model(compile = True):
	from keras.models import Sequential
	from keras.layers import Conv2D,Flatten,Dense,Dropout
	layers = [
		Conv2D(32,3,strides=1,activation='relu',padding='same',input_shape=input_shape),
		Conv2D(32,3,strides=2,activation='relu',padding='same'),
		Dropout(0.25),
		Conv2D(64,3,strides=1,activation='relu',padding='same'),
		Conv2D(64,3,strides=2,activation='relu',padding='same'),
		Dropout(0.25),
		Conv2D(128,3,strides=1,activation='relu',padding='same'),
		Conv2D(128,3,strides=2,activation='relu',padding='same'),
		Dropout(0.25),
		Flatten(),
		Dense(number_of_class*30),
		Dense(number_of_class*10),
		Dense(number_of_class)
		]
	model = Sequential(layers=layers, name='xphoto_classifier')
	if compile:
		model.compile(optimizer='RMSProp', loss='mse' ,metrics=['accuracy'])
	return model

def dataGenerator(path):
	from keras.preprocessing import image
	imG = image.ImageDataGenerator(data_format = 'channels_last')#preprocessing_function not avalible
	return imG.flow_from_directory(path, class_mode='categorical',target_size = input_shape[:2], batch_size = batch_size)
	#while True:
	#	imgs , labels = next(imG)
	#	return imgs/255,labels

def save_model(model):
	print('saving {}'.format(model_path))
	model.save_weights(model_path)

def load_model(for_test = True):
	model = new_model(compile = not for_test)
	try:
		print('loading {}'.format(model_path))
		model.load_weights(model_path)
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
	model = load_model(for_test = False)
	train_model(model)
	save_model(model)

if __name__ == '__main__':
	main()

