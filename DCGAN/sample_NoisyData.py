from D import NoizyData

n = NoizyData()

train_data = n.train_batch(100)

a,b = n.test()
test_data = [a[:100],b[:100]]

import cv2
import numpy as np

for i,(im,label) in enumerate(zip(*train_data)):
	cv2.imwrite('results/od/train-'+str(i)+'.png',(im*255).astype(np.uint8))

for i,(im,label) in enumerate(zip(*test_data)):
	cv2.imwrite('results/od/test-'+str(i)+'.png',(im*255).astype(np.uint8))