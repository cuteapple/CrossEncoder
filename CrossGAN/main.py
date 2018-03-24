import keras
from FullDataLoader import DataLoader
from AutoEncoder import AutoEncoder

class CrossEncoder():
	def __init__(self):
		from keras.models import Model
		from keras.layers import Input
		import numpy

		self.auto_loss = 5
		self.cross_loss = 1

		self.a = a = AutoEncoder('ukiyoe')
		self.a.dataset = DataLoader('x2photo/train/ukiyoe',(a.width,a.height))
		self.b = b = AutoEncoder('photo')
		self.b.dataset = DataLoader('x2photo/train/photo',(b.width,b.height))
		
		a.discriminator.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'])
		b.discriminator.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'])

		a.autoencoder.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'],loss_weights=[self.auto_loss])
		b.autoencoder.compile(optimizer = 'rmsprop',loss = 'mse', metrics=['accuracy'],loss_weights=[self.auto_loss])

		
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
		
		half_batch = self.batch_size//2

		real_a = self.a.dataset.load(half_batch)
		fack_a = self.generate_a(half_batch)

		real_b = self.b.dataset.load(half_batch)
		fack_b = self.generate_b(half_batch)

		self.a.discriminator.train_on_batch(real_a, self.real_flags[:half_batch])
		self.a.discriminator.train_on_batch(fack_a, self.fake_flags[:half_batch])

		self.b.discriminator.train_on_batch(real_b, self.real_flags[:half_batch])
		self.b.discriminator.train_on_batch(fack_b, self.fake_flags[:half_batch])

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
			pass
		try:
			self.b.load()
		except:
			print(self.b.name, " load failed")
			pass

	def train(self, epoch=30000, batch_size=128, save_interval=20, save_path='save'):

		try:
			import os
			os.makedirs(save_path)
		except:
			pass

		
		import numpy
		self.batch_size = batch_size
		self.real_flags = numpy.ones(self.batch_size)
		self.fake_flags = numpy.zeros(self.batch_size)

		from datetime import timedelta
		from time import time as now
		start = now()
		
		for round in range(epoch):
			print(round,end=' ', flush = True)
			print('auto',end=' ', flush = True)
			self.train_autoencoder()
			#print('dis',end=' ', flush = True)
			#self.train_discrimator()
			#print('cross',end=' ', flush = True)
			#self.train_crossencoder()
			print('end -- ' , str(timedelta(seconds=now()-start)), flush = True)

			if round % save_interval == 0:
				self.save_images(save_path,round)

	def save_images(self,path,round):
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
		auto_a = a.decoder.predict(z_a)
		cross_b = b.decoder.predict(z_a)
		return im_a,auto_a,cross_b


if __name__ == '__main__':
	E = CrossEncoder()
	E.tryload()
	try:
		E.train(10000)
	finally:
		E.save()