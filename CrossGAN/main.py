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

		self.autoencoder = keras.models.Model(self.i,self.o)
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

		self.auto_loss = 5
		self.cross_loss = 1

		self.batch_size = 128
		self.real_flags = numpy.ones(self.batch_size)
		self.fake_flags = numpy.zeros(self.batch_size)

		self.a = a = AutoEncoder('ukiyoe')
		self.a.dataset = DataLoader('x2photo/train/ukiyoe',(a.width,a.height))
		self.b = b = AutoEncoder('photo')
		self.b.dataset = DataLoader('x2photo/train/photo',(b.width,b.height))
		
		a.discriminator.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'])
		b.discriminator.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'])

		a.autoencoder.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'],loss_weights=[self.auto_loss])
		a.autoencoder.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'],loss_weights=[self.auto_loss])

		
		fack_a = a.decoder(b.z)
		fack_b = b.decoder(a.z)

		a.discriminator.trainable = False
		b.discriminator.trainable = False

		da = a.discriminator(fack_a)
		db = b.discriminator(fack_b)

		cross_ab = Model(a.i,db)
		cross_ba = Model(b.i,da)
		cross_ab.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'], loss_weights=[self.cross_loss])
		cross_ba.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'], loss_weights=[self.cross_loss])

		# just for convenience, not train directly
		class Models:pass
		self.models = models = Models()
		models.gab = Model(a.i, fack_b)
		models.gba = Model(b.i, fack_a)
		models.cross_ab = cross_ab
		models.cross_ba = cross_ba

	def generate_a(self,batch_size):
		''' generate *batch_size* a from b '''
		data = self.b.dataset.load(batch_size)
		return self.models.gab.predict(data,batch_size=batch_size)

	def generate_b(self,batch_size):
		''' generate *batch_size* a from b '''
		data = self.a.dataset.load(batch_size)
		return self.models.gba.predict(data,batch_size=batch_size)

	def train_discrimator(self):
		''' train discirminator TODO: log loss '''
		
		half_batch = batch_size/2

		real_a = a.dataset.load(half_batch)
		fack_a = self.generate_a(half_batch)

		real_b = b.dataset.load(half_batch)
		fack_b = self.generate_b(half_batch)

		self.a.discriminator.train_on_batch(real_a, self.real_flags)
		self.a.discriminator.train_on_batch(fack_a, self.fake_flags)

		self.b.discriminator.train_on_batch(real_b, self.real_flags)
		self.b.discriminator.train_on_batch(fack_b, self.fake_flags)

	def train_autoencoder(self):
		''' TODO? : combine two model on training? '''
		real_a = self.a.dataset.load(self.batch_size)
		real_b = self.b.dataset.load(self.batch_size)

		self.a.autoencoder.train_on_batch(real_a,real_a)
		self.b.autoencoder.train_on_batch(real_b,real_b)

	def train_crossencoder(self):
		''' TODO? : combine two model on training? '''
		data = self.a.dataset.load(self.batch_size)
		self.models.cross_ab.train_on_batch(data,self.real_flags)
		data = self.b.dataset.load(self.batch_size)
		self.models.cross_ba.train_on_batch(data,self.real_flags)

	def save(self):
		self.a.save()
		self.b.save()

	def tryload(self):
		try:
			self.a.load()
		except:
			print(self.a.name, " load failed")
		try:
			self.b.load()
		except:
			print(self.b.name, " load failed")

	def train(self, epoch=30000, batch_size=128, save_interval=20, save_path='save'):
		for round in range(epoch):
			print(round)
			self.train_autoencoder()
			self.train_discrimator()
			self.train_crossencoder()

			if round % save_interval == 0:
				self.save_images(save_path,round)

	def save_images(self,path,round):
		from DataLoader import DataLoader
		a,ra,cb = self._gen_save_images(self.a,self.b)
		b,rb,ca = self._gen_save_images(self.b,self.a)

		l = [a,ra,cb,b,rb,ca]
		for i,im in enumerate(l):
			f = '{}/{}-{}.png'.format(path,round,i)
			DataLoader.save_image(im,f)
			

	@staticmethod		
	def _gen_save_images(a,b):
		im_a = a.dataset.load(1)
		z_a = a.encoder.predict(im_a)
		auto_a = a.decoder(z_a)
		cross_b = b.decoder(z_a)
		return im_a,auto_a,cross_b


if __name__ == '__main__':
	E = CrossEncoder()
	E.tryload()
	try:
		E.train()
	finally:
		E.save()