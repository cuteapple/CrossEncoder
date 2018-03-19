import keras

import classifier
classifier = classifier.load()

from model import Encoder, Decoder, input_shape
from keras.models import Model
from keras.layers import Input

z_dim = 100
from keras.datasets import mnist
import numpy as np
(x, y), _ = mnist.load_data()
#x1 ,y1 = x[y==1], y[y==1] # y is full of 1 :P
x=x.astype('float')/255
x=x.reshape((-1,*input_shape))

x_n = *(x[y==i] for i in range(10)),

X = x_n

e1 = Encoder()
d1 = Decoder()

e5 = Encoder()
d5 = Decoder()

i1 = Input(input_shape)
i5 = Input(input_shape)

auto1 = d1(e1(i1))
auto5 = d5(e5(i5))

cross51 = d1(e5(i5))
cross15 = d5(e1(i1))

D15 = classifier(cross15)
D51 = classifier(cross51)

model_1 = Model([i1],[auto1,D15])
model_1.compile(optimizer='RMSProp',loss=['mse','categorical_crossentropy'],loss_weights=[1,2],metrics=['accuracy'])

model_5 = Model([i5],[auto5,D51])
model_5.compile(optimizer='RMSProp',loss=['mse','categorical_crossentropy'],loss_weights=[1,2],metrics=['accuracy'])

model_all = Model([i1,i5],[auto1,auto5,D51,D15])
model_all.compile(optimizer='RMSProp',
				loss=['mse','mse','categorical_crossentropy','categorical_crossentropy']
				,loss_weights=[1,1,1,1],metrics=['accuracy'])

try: # cannot plot model on google colab
	from keras.utils import plot_model
	plot_model(model_all, to_file='model_all.png',show_shapes=True)
except:
	pass


yab = lambda a,b : np.tile(np.eye(10)[b],len(X[a])).reshape(-1,10)
y15 = yab(1,5)
y51 = yab(5,1)

for i in range(100):
	print('epoch {} '.format(i))
	model_1.fit(
		x = [X[1]],
		y = [X[1],y15],
		batch_size = 128,
		epochs=2,
		verbose = 1
		)
	model_5.fit(
		x = [X[5]],
		y = [X[5],y51],
		batch_size = 128,
		epochs = 2,
		verbose = 1
	)

model_1.save('model_1.h5')
model_5.save('model_5.h5')