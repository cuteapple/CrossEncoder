from D import NoizyData

n = NoizyData()

data = n.train_batch(100)

import cv2
import numpy as np

for i,(im,label) in enumerate(zip(*data)):
	cv2.imwrite('results/od/'+str(i)+'.png',(im*255).astype(np.uint8))
