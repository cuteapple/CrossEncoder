import encoder

_,G,D = encoder.load_model(True)

zG = encoder.dataGenerator(encoder.test_data_folder)

z,l = next(zG)
for i,(z,p,l) in enumerate(zip(z,D.predict(z),l)):
	print(p)
	print(l)

'''
z = next(zG)[0]
p = G.predict(z)
d = D.predict(p)


import cv2
for i,(x,y,dd) in enumerate(zip(z,p,d)):
	cv2.imwrite('{}_x.png'.format(i),x[:,:,[2,1,0]])
	cv2.imwrite('{}_y.png'.format(i),y)
	print(dd)
	'''