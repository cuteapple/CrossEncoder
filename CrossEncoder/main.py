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

x_n = [x[y==i] for i in range(10)]

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

M11 = Model(i1,auto1)
M15 = Model(i1,cross15)
M55 = Model(i5,auto5)
M51 = Model(i5,cross51)

M11.compile(optimizer='RMSProp',loss='mse',metrics=['accuracy'])
M55.compile(optimizer='RMSProp',loss='mse',metrics=['accuracy'])
M15.compile(optimizer='RMSProp',loss='categorical_crossentropy',metrics=['accuracy'])
M51.compile(optimizer='RMSProp',loss='categorical_crossentropy',metrics=['accuracy'])


yab = lambda a,b : np.tile(np.eye(10)[b],len(X[a])).reshape(-1,10)

#from keras.preprocessing.image import ImageDataGenerator

#imG = ImageDataGenerator(width_shift_range=20,height_shift_range=20,fill_mode='constant',cval=0)

x1 = X[3]
x5 = X[5]
y15 = yab(3,5)
y51 = yab(5,3)

#G11 = img.flow(x1,x1,batch_size=256)
#G55 = img.flow(x5,x5,batch_size=256)
#G15 = img.flow(x1,y15,batch_size=256)
#G51 = img.flow(x1,y15,batch_size=256)

for i in range(100):
	print('epoch {} '.format(i))
	M11.fit(x1,x1,batch_size=128,epochs=10,verbose=0)
	M55.fit(x5,x5,batch_size=128,epochs=10,verbose=0)
	M15.fit(x1,y15,batch_size=128,epochs=30,verbose=0)
	M51.fit(x5,y51,batch_size=128,epochs=30,verbose=0)

Model(i1,auto1).save('M11.h5')
Model(i1,cross15).save('M15.h5')
Model(i5,auto5).save('M55.h5')
Model(i5,cross51).save('M51.h5')