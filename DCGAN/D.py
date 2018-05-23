import keras
import numpy as np
from Dataset import NoizyData

class D:
	def __init__(self):
		self.model = self.new_classifier()

	@staticmethod
	def new_classifier():
		from keras.models import Sequential
		from keras.layers import Conv2D,Flatten,Dense,Dropout,Input
		model = Sequential(name='mnist_classifier',
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
	
	@classmethod
	def Load(cls,path=None,or_new=False):
		inst = cls()
		if path is None:
			path = cls.default_path
		try:
			inst.load_weights(path)
		except:
			print('load weight fail')
			if not or_new:
				raise
		return inst

	def save_weights(self,path=None):
		if path is None:
			path = D.default_path
		self.model.save_weights(path)
	def load_weights(self,path):
		self.model.load_weights(path)

	def compile(self,optimizer='adadelta', loss='mse' ,metrics=['accuracy'],*args):
		self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics,*args)

	def train(self,data,epochs=200,batch_size=128):
		x,y = data.train()
		tx,ty = data.test()

		self.model.fit(x,y,
			batch_size = batch_size,
			epochs=epochs,
			validation_data=(tx,ty))

if __name__ == "__main__":

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-e","--epochs", default=200, type=int)
	parser.add_argument("-b","--batch_size", default=128, type=int)
	parser.add_argument("-p","--path", default="D.h5", type=str)
	parser.add_argument("-nx","--noise_sacler_x", default=0.5, type=float)
	args = parser.parse_args()

	print('loading weights ...')
	d = D.Load(args.path,True)
	
	print('training ...')
	d.compile()
	data = NoizyData(noise_scaler=args.noise_sacler_x)
	d.train(data,epochs=args.epochs,batch_size=args.batch_size)

	print('saving ...')
	d.save_weights(args.path)
