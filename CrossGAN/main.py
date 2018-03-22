import keras

class AutoEncoder():
	def __init__(self):
		self.width = 128
		self.height = 128
		
		#build graph
		self.ioshape = (self.width, self.height, 3)
		self.encoder = self.newEncoder()
		self.z_shape = self.encoder.output_shape[1:]
		self.decoder = self.newDecoder()
		assert self.ioshape == self.decoder.output_shape[1:]
		#self.discriminator = self.newDiscriminator()
		
		#some easy-to-use things 
		self.i = keras.layers.Input(self.ioshape)
		self.z = self.encoder(self.i)
		self.o = self.decoder(self.z)

		self.autoencoder = keras.models.Model(self.i,self.o)
		self.autoencoder.compile('SGD','')

	def save(file):
		self.encoder.save_weights(file)
		self.decoder.save_weights(file)
	
	def load(file):
		self.encoder.load_weights(file)
		self.decoder.load_weights(file)

	def newEncoder(self):
		...

	def newDecoder(self):
		...

	def link_and_compile(self):
		enc = self.encoder
		dec = self.decoder
		#D = self.discriminator
		from keras.layers import Input
		i = Input(shape = (self.width, self.height, 3))
		o = dec(enc(i))
		return keras.models.Model(i,o)


a = AutoEncoder()
b = AutoEncoder()

ia = Input(a.ioshape)
ib = Input(b.ioshape)


x = a.decoder(a.encoder(ia))
y = b.autoencoder
z = a.decoder(b.encoder)
w = b.encoder(a.decoder)