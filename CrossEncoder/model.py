import keras
from keras.models import Sequential,Model
from keras.layers import *

def Encoder(input_shape=(28,28,1)):
	return Sequential([
		Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=input_shape),
		MaxPooling2D((2, 2), padding='same'),
		Conv2D(8, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same'),
		Conv2D(8, (3, 3), activation='relu', padding='same'),
		MaxPooling2D((2, 2), padding='same')
		])

def Decoder(input_shape):
	return Sequential([
		Conv2D(8, (3, 3), activation='relu', padding='same',input_shape=input_shape),
		UpSampling2D((2, 2)),
		Conv2D(8, (3, 3), activation='relu', padding='same'),
		UpSampling2D((2, 2)),
		Conv2D(16, (3, 3), activation='relu'),
		UpSampling2D((2, 2)),
		Conv2D(1, (3, 3), activation='sigmoid', padding='same')
		])

