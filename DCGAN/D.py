import keras
import numpy as np
from Dataset import NoizyData

def new_D():
	from keras.models import Sequential,Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout,Input,MaxPooling2D
	modelc = Sequential(name='classifier',
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
			Dense(10)])

	modeln = Sequential(name='isnoisy',
		layers=[Conv2D(32, kernel_size=3, strides=1, activation='relu',input_shape=(28,28,1)),
			MaxPooling2D(),
			Conv2D(64, kernel_size=3, activation='relu'),
			MaxPooling2D(),
			Dropout(0.5),
			Flatten(),
			Dense(128, activation='relu'),
			Dropout(0.5),
			Dense(128, activation='relu'),
			Dropout(0.5),
			Dense(1)])
	
	i = Input((28,28,1),name='image')
	oc = modelc(i)
	on = modeln(i)

	model = Model(i,[oc,on],name='D')
	#model.summary()

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
	def g():
		nreal,nfake = args.batch_size//2, args.batch_size//2
		while True:
			yield data.train_batch(nreal,nfake)

	print('training ... ')
	d.compile(optimizer='adadelta', loss='mse', metrics=['mae','accuracy'])
	
	d.fit_generator(
		g(),
		steps_per_epoch = len(data.x) // args.batch_size,
		validation_data = data.test(),
		epochs=args.epochs
		)
	
	print('saving ...')
	d.save_weights(args.path)
