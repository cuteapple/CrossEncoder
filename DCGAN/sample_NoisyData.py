from D import NoizyData

n = NoizyData(noise_scaler=0.5)

a,b = n.train()
train_data = [a[:100],b[:100]]

a,b = n.test()
test_data = [a[:100],b[:100]]

import cv2
import numpy as np

for i,(im,label) in enumerate(zip(*train_data)):
	cv2.imwrite('results/od/train-' + str(i) + '.png',(im * 255))

for i,(im,label) in enumerate(zip(*test_data)):
	cv2.imwrite('results/od/test-' + str(i) + '.png',(im * 255))