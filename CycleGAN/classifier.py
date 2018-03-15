import keras

input_shape=(256,256)
model_path = 'xphoto_classifier.h5'
data_folder = 'x2photo/train'
epochs = 50
batch_size = 32
number_of_class = 4

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

def train_data():
	imG = keras.preprocessing.image.ImageDataGenerator()
	imG.flow_from_directory(data_folder, class_mode='categorical',batch_size = batch_size)

def load_model(new_on_fail=True):
	model = new_model()
	try:
		print('loading {}'.format(model_path))
		model.load_weights(model_path)
	except (OSError,ValueError) as e:
		if not new_on_fail:
			raise
		else:
			print(str(e))
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

def test_model(model):
	_,(x_test,y_test) = load_mnist()
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


def main():
	model = load_model(new_on_fail=True)
	train_model(model)
	save_model(model)
	#test_model(model)

if __name__ == '__main__':
	main()

