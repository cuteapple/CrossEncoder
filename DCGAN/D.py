import keras
import numpy as np
import Dataset

def data(batch_size,batch_noise_ratio,scale_y):
	(ox,oy),_ = Dataset.load_mnist()
	b = batch_size
	bn = int(b*batch_noise_ratio)
	def g():
		choice = np.random.randint(len(ox),size=batch_size)
		x = ox[choice]
		y = oy[choice]
		Dataset.add_noise(x[:bn])
		y[:bn] *= scale_y
		return x,y
	
	while True:
		yield g()

def D():
	from keras.models import Sequential,Model
	from keras.layers import Conv2D,Flatten,Dense,Dropout,LeakyReLU,Activation
	return Sequential(name='D',
		layers=[Conv2D(32, kernel_size=3, strides=1, padding='same', input_shape=(28,28,1)),
			Activation('relu'),
			Dropout(0.5),
			Conv2D(64, kernel_size=3, strides=2,padding='same'),
			Activation('relu'),
			Dropout(0.5),
			Flatten(),
			Dense(256),
			Activation('relu'),
			Dropout(0.5),
			Dense(256),
			Activation('relu'),
			Dense(10)])

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-bnr","--batch_noise_ratio", default=0.6, type=float)
	parser.add_argument("-sy","--noisy_y_scalar", default=0.5, type=float)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	args = parser.parse_args()
	print('args',args)

	
	print('loading from {} ... '.format(args.path), end = '')
	try: 
		dc = keras.models.load_model(args.path)
	except:
		print('fail')
		dc = D()
		dc.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
	else:
		print('success')
	
	print('prepare data ...')

	d = data(args.batch_size,args.batch_noise_ratio,args.noisy_y_scalar)
	print('training ... ')
	dc.fit_generator(d,steps_per_epoch=256,epochs=args.epochs)
	
	print('saving ...')
	dc.save(args.path)