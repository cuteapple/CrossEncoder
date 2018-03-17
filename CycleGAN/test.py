import encoder

_,G,D = encoder.load_model(True)

z = next(encoder.dataGenerator(encoder.test_data_folder))[0]
p = G.predict(z)

import cv2
for i,(x,y) in enumerate(zip(z,p)):
	cv2.imwrite('{}_x.png'.format(i),x[:,:,[2,1,0]])
	cv2.imwrite('{}_y.png'.format(i),y)