import keras
import numpy as np
from Dataset import NoizyData

def new_D():
	from keras.models import Sequential
	from keras.layers import Conv2D,Flatten,Dense,Dropout
	model = Sequential(name='D',
		layers=[Conv2D(32, kernel_size=3, strides=1, activation='relu',input_shape=(28,28,1)),
			Conv2D(64, kernel_size=3, strides=2, activation='relu'),
			Dropout(0.5),
			Conv2D(64, kernel_size=3, strides=2, activation='relu'),
			Dropout(0.5),
			Flatten(),
			Dense(128, activation='relu'),
			Dropout(0.5),
			Dense(128, activation='relu'),
			Dropout(0.5),
			Dense(11)])
	return model

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	args = parser.parse_args()
	print('args',args)

	d = new_D()

	print('loading weights at {} ... '.format(args.path), end = '')
	try: d.load_weights(args.path)
	except: print('fail')
	else: print('success')
	
	print('prepare data ...')
	data = NoizyData()
	
	print('training ... ')
	d.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
	
	x,y = data.train()
	d.fit(x,y, batch_size=args.batch_size, epochs=args.epochs, validation_data=data.test())

	print('saving ...')
	d.save_weights(args.path)
