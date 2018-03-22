import keras
from keras.models import Sequential,Model
from keras.layers import *
import numpy
#from keras_contrib.layers.normalization import InstanceNormalization

input_shape = (28,28,1)
z_shape = (10,)

def Encoder():
	return Sequential([
		Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=input_shape),
		MaxPooling2D((2, 2), padding='same'),
		Conv2D(8, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same'),
		Conv2D(8, (3, 3), activation='relu', padding='same'),
		Flatten(),
		Dense(z_shape[0], activation='relu')
		])

def Decoder():
	return Sequential([
		Dense(7*7*8, activation='relu',input_shape=z_shape),
		Reshape((7,7,8)),
		Conv2D(8, (3, 3), activation='relu', padding='same',input_shape=input_shape),
		UpSampling2D((2, 2)),
		Conv2D(16, (3, 3), activation='relu', padding='same'),
		UpSampling2D((2, 2)),
		Conv2D(1, (3, 3), activation='sigmoid', padding='same')
		])