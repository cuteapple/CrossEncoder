import keras

from keras.datasets import mnist
from keras.models import load_model

(x, _), _ = mnist.load_data()
x=x.astype('float')/255
x=x.reshape((-1,*input_shape))
x_n = *(x[y==i] for i in range(10)),
X = x_n

import numpy as np
import cv2

M11 = load_model('M11.h5')
M15 = load_model('M15.h5')
M51 = load_model('M51.h5')
M55 = load_model('M11.h5')