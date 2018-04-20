import keras
import numpy as np

input_shape = (256,256,3)
output_shape=(2,)

def model():
	from keras.models import Sequential
	from keras.layers import Conv2D,Flatten,Dense,Dropout,Lambda,MaxPooling2D,Activation
	from keras.layers import LeakyReLU
	layers = [
		Conv2D(16,(5,5),strides= (2,2),padding='same',input_shape=input_shape),
		LeakyReLU(),
		Conv2D(32,(5,5),strides= (2,2),padding='same'),
		LeakyReLU(),
		Conv2D(64,(5,5),strides= (2,2),padding='same'),
		LeakyReLU(),
		Conv2D(128,(5,5),strides= (2,2),padding='same'), #(16,16,128)
		LeakyReLU(),
		Conv2D(256,(5,5),strides= (2,2),padding='same'), #(8,8,256)
		LeakyReLU(),
		Flatten(),
		Dense(2048),
		LeakyReLU(),
		Dense(2048),
		LeakyReLU(),
		Dense(output_shape),
		]
	return Sequential(layers=layers, name='xphoto_D')

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	parser.add_argument("-ny","--noise_y", default=0.5, type=float)
	args = parser.parse_args()

	print('loading weights ...')
	d = D.Load(args.path,True)
	
	print('training ...')
	d.compile()
	data = NoizyData(y_scaler=args.noise_y)
	d.train(data,epochs=args.epochs,batch_size=args.batch_size)

	print('saving ...')
	d.save_weights(args.path)
