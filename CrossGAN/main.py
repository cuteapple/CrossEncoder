import keras
import argparse

class AutoEncoder():
	def __init__(self,compile=True):

		#parameters

		self.width = 128
		self.height = 128
		
		#build graph

		self.ioshape = (self.width, self.height, 3)
		self.encoder = self.newEncoder()
		self.z_shape = self.encoder.output_shape[1:]
		self.decoder = self.newDecoder()
		assert self.ioshape == self.decoder.output_shape[1:]
		
		#some easy-to-use things 
		
		self.i = keras.layers.Input(self.ioshape)
		self.z = self.encoder(self.i)
		self.o = self.decoder(self.z)

		self.autoencoder = keras.models.Model(self.i,self.o)
		if compile:
			self.autoencoder.compile('SGD',loss='mse',metrics=['accuracy'])

	def save(file):
		self.autoencoder.save_weights(file)
	
	def load(file):
		self.autoencoder.load_weights(file)

	def newEncoder(self):
		from keras.layers import Input,Conv2D
		i = Input(self.ioshape)
		layers = []

	def newDecoder(self):
		...


def main():
	a = AutoEncoder()
	b = AutoEncoder()

	a2b = b.decoder(a.z)
	b2a = a.decoder(b.z)



	y = b.autoencoder
	z = a.decoder(b.encoder)
	w = b.encoder(a.decoder)