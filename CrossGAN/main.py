import keras
import keras_contrib

class AutoEncoder():
	''' autoencoder + discriminator, all trainable'''
	def __init__(self,name):

		#parameters
		self.name = name
		self.width = 128
		self.height = 128
		# parameter for [encoder, decoder]
		self.deeps = [[32,64,128,256],[128,64,32,3]]
		self.nres = [3,3]

		#build graph

		self.ioshape = (self.width, self.height, 3)
		self.encoder = self.newEncoder()
		self.z_shape = self.encoder.output_shape[1:]
		self.decoder = self.newDecoder()
		assert self.ioshape == self.decoder.output_shape[1:]
		self.discriminator = self.newDiscriminator()

		#some easy-to-use things 
		
		self.i = keras.layers.Input(self.ioshape)
		self.z = self.encoder(self.i)
		self.o = self.decoder(self.z)
		self.d = self.discriminator(self.o)

		self.fullmodel = keras.models.Model(self.i,self.d)

	def save(self):
		''' save weights to {self.name} '''
		self.fullmodel.save_weights(self.name+'.h5')
	
	def load(self):
		''' load weights from {self.name} '''
		self.fullmodel.load_weights(self.name+'.h5')

	def newEncoder(self):
		''' brand new encoder '''
		
		input_shape = self.ioshape
		nres, deeps = self.nres[0], self.deeps[0]

		y = i = keras.layers.Input(input_shape)

		for d in deeps:
			y = self.convdown(y,d)

		for _ in range(nres):
			y = self.convres(y,deeps[-1])

		return keras.models.Model(i,y)

	def newDecoder(self):
		''' brand new decoder '''

		input_shape = self.z_shape
		nres, deeps = self.nres[1], self.deeps[1]

		y = i = keras.layers.Input(input_shape)

		for _ in range(nres):
			y = self.convres(y, input_shape[-1])

		for deep in deeps:
			y = self.convup(y,deep)

		return keras.models.Model(i,y)

	def newDiscriminator(self):
		''' TODO : parameterize '''
		from keras.layers import Flatten,Dense
		input_shape = self.ioshape
		deeps = [16,32,64,128]
		y = i = keras.layers.Input(input_shape)
		for d in deeps:
			y = self.convdown(y,d)
		y = Flatten()(y)
		y = Dense(1)(y)
		return keras.models.Model(i,y)

	@staticmethod
	def convdown(x,deep,kernal=(5,5)):
		''' conv 1/2 -> lrelu -> instanceNorm '''
		from keras.layers import Conv2D,LeakyReLU
		from keras_contrib.layers.normalization import InstanceNormalization
		x = Conv2D(deep, kernel_size=kernal, strides=2, padding='same')(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = InstanceNormalization()(x)
		return x

	@staticmethod
	def convup(x,deep,kernal=(5,5)):
		''' upsample -> conv -> lrelu -> instanceNorm '''
		from keras.layers import Conv2D,LeakyReLU,UpSampling2D
		from keras_contrib.layers.normalization import InstanceNormalization
		x = UpSampling2D()(x)
		x = Conv2D(deep, kernel_size=kernal, strides=1, padding='same')(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = InstanceNormalization()(x)
		return x

	@staticmethod
	def convres(x,deep,kernal=(3,3)):
		''' resblock '''
		from keras.layers import Conv2D,Add,LeakyReLU
		y = Conv2D(deep, kernel_size=kernal, strides=1, padding='same')(x)
		y = LeakyReLU(alpha=0.2)(y)
		y = Conv2D(deep, kernel_size=kernal, strides=1, padding='same')(y)
		y = Add()([x,y])
		y = LeakyReLU(alpha=0.2)(y)
		return y

class CrossEncoder():
	def __init__(self):
		from keras.models import Model
		from keras.layers import Input
		from DataLoader import DataLoader
		import numpy

		self.batch_size = 128
		self.ones = numpy.ones(batch_size)
		self.zeros = numpy.zeros(batch_size)

		self.a = a = AutoEncoder('ukiyoe')
		self.a.dataset = DataLoader('x2photo/train/ukiyoe',(a.width,a.height))
		self.b = b = AutoEncoder('photo')
		self.b.dataset = DataLoader('x2photo/train/photo',(b.width,b.height))

		fack_b = b.decoder(a.z)
		fack_a = a.decoder(b.z)

		a2b = b.discriminator(b.decoder(a.z))
		b2a = a.discriminator(a.decoder(b.z))

		class Models:pass
		self.models = models = Models()
		models.gab = Model(a.i, fack_b)
		#compile gab
		models.gba = Model(a.i, fack_b)
		#compile gba
		models.gab.trainable = False
		models,gba.trainable = False
		models.dab = 
		
		m_abD = Model(a.i,a2b)
		m_baD = Model(b.i,b2a)

	def train_discrimator(self):
		''' train discirminator '''
		

		za = a.encoder.predict
		half_batch = batch_size/2

		real_a = a.dataset.load(half_batch)
		
		fack_a = b.dataset.load(half_batch)
		


		a.encoder.predict(da)
		db = b.dataset.load(batch_size)
		a.encoder.predict(da)
	
	def xxx():
		from keras.utils import plot_model
		plot_model(a.encoder, to_file='e.png',show_shapes =True)
		plot_model(a.decoder, to_file='d.png',show_shapes =True)
		plot_model(a.discriminator, to_file='di.png',show_shapes =True)
		a.save()
		a.load()

	def train():
		...

if __name__ == '__main__':
	CrossEncoder().train()