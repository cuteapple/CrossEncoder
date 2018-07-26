import keras
import numpy as np


class NoizyData:
	'''noizy mnist data'''
	def __init__(self, noise_sigma=1.0, noise_scaler=0.5, y_scaler=0.3):
		noise_mean = 0.0

		(x,y),(tx,ty) = self.load_datas()

		nx = x + noise_scaler * np.random.normal(noise_mean, noise_sigma, size=x.shape)
		ny = y * y_scaler

		self.x = np.concatenate((x,nx), axis=0)
		self.y = np.concatenate((y,ny), axis=0)
		self.tx = tx
		self.ty = ty

	def train(self):
		return self.x,self.y

	def test(self):
		return self.tx,self.ty

	@staticmethod
	def transform(x):
		return x.astype('float32').reshape(-1,32,32,3) / 255
	
	@staticmethod
	def transform_inv(x):
		return x * 255

	@staticmethod
	def load_datas():
		from keras.datasets import cifar10
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		x_train = NoizyData.transform(x_train)
		x_test = NoizyData.transform(x_test)
		y_train = keras.utils.to_categorical(y_train, 10)
		y_test = keras.utils.to_categorical(y_test, 10)
		return (x_train,y_train),(x_test,y_test)

def new_D():
	from keras.models import Sequential
	from keras.layers import Conv2D,Flatten,Dense,Dropout,Input
	model = Sequential(name='D',
		layers=[Conv2D(32, kernel_size=3, strides=1, activation='relu',input_shape=(32,32,3)),
			Conv2D(64, kernel_size=3, strides=2, activation='relu'),
			Dropout(0.5),
			Conv2D(64, kernel_size=3, strides=2, activation='relu'),
			Dropout(0.5),
			Flatten(),
			Dense(256, activation='relu'),
			Dropout(0.5),
			Dense(128, activation='relu'),
			Dropout(0.5),
			Dense(10)])
	model.compile('adadelta', loss='mse', metrics=['accuracy'])
	return model

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	parser.add_argument("-ny","--noise_y", default=0.3, type=float)
	parser.add_argument("-nx","--noise_sacler_x", default=0.5, type=float)
	args = parser.parse_args()

	print(f'loading model at {args.path} ...')
	try:
		d = keras.models.load_model(args.path)
	except:
		print('fail, creating new')
		d = new_D()
	else:
		print('success')
	
	print('training ...')
	x,y = NoizyData(y_scaler=args.noise_y, noise_scaler=args.noise_sacler_x).train()
	d.fit(x,y,epochs=args.epochs,batch_size=args.batch_size)

	print('saving ...')
	d.save(args.path)
