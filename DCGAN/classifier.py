import keras
import numpy as np

input_shape = (28,28,1)
model_path = 'mnist_classifier.h5'
epochs = 200
batch_size = 128
noise_factor = 0.5
noise_y_factor = 0

def new_model():
	from keras.models import Sequential
	from keras.layers import Conv2D,Flatten,Dense,Dropout,Input
	model = Sequential(name='mnist_classifier')
	model.add(Conv2D(32, kernel_size=3, strides=1, activation='relu',input_shape=input_shape))
	model.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.compile(optimizer='adadelta', loss='mse' ,metrics=['accuracy'])
	return model

loaded_mnist = False
def load_mnist():
	global loaded_mnist
	if not loaded_mnist:
		from keras.datasets import mnist
		def transform(x):
			return x.astype('float32').reshape(-1,*input_shape) / 255

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
			raise
		else:
			print(str(e))
			print('load weights failed, recreate')
	return model
	
def data():
	(x,y),(a,b) = load_mnist()
	x_noizy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
	y_noizy = y * noise_y_factor
	x_noizy = np.clip(x_noizy,0.0,1.0)
	y_noizy = np.clip(y_noizy,0.0,1.0)
	x = np.concatenate(x,x_noizy)
	y = np.concatenate(y,y_noizy)
	return (x,y),(a,b)

def train_model(model):
	(x,y),(tx,ty) = data()
	model.fit(
		x,y,
		batch_size = batch_size,
		epochs=epochs,
		validation_data=(tx,ty)
		)

def save_model(model):
	print('saving {}'.format(model_path))
	model.save_weights(model_path)

def main():
	model = load_model(new_on_fail=True)
	train_model(model)
	save_model(model)

if __name__ == '__main__':
	main()

