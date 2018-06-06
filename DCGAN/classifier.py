import keras
import numpy as np
import Dataset

def new_model():
	from keras.models import Sequential,Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout,LeakyReLU,BatchNormalization
	return Sequential(name='class',
		layers=[Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=(28,28,1)),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(64, kernel_size=3, strides=2,padding='same'),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(128, kernel_size=3, strides=2, padding='same'),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(256, kernel_size=3, strides=2, padding='same'),
			BatchNormalization(),
			LeakyReLU(),
			Dropout(0.5),
			Conv2D(256, kernel_size=3, strides=1, padding='same'),
			BatchNormalization(),
			LeakyReLU(),
			Flatten(),
			Dense(2048),
			BatchNormalization(),
			LeakyReLU(),
			Dense(1024),
			BatchNormalization(),
			LeakyReLU(),
			BatchNormalization(),
			LeakyReLU(),
			Dense(10)])

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="classifier.h5", type=str)
	args = parser.parse_args()
	print('args',args)

	
	print('loading weights at {} ... '.format(args.path), end = '')
	try: 
		dc = keras.models.load_model(args.path)
	except:
		print('fail')
		dc = new_model()
		dc.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
	else:
		print('success')
	
	print('prepare data ...')
	(x,y),_ = Dataset.load_mnist()

	print('training ... ')
	dc.fit(x,y,batch_size=args.batch_size, epochs = args.epochs)
	
	print('saving ...')
	dc.save(args.path)