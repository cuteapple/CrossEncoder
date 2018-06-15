import keras

m1 = keras.models.load_model('D04.h5')
m2 = keras.models.load_model('D59.h5')

i = keras.models.Input((28,28,1))
o = keras.layers.concatenate([m1(i),m2(i)])

m = keras.models.Model(i,o)
m.save('D.h5')