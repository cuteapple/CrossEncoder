from D import NoizyData

x,_ = NoizyData(0.5,0.5,0.3).train()

x = x[len(x)//2:] #noisy part

import numpy as np

c = np.ones((28*4,28*5,1))*0.5

for a in range(4):
	for b in range(5):
		c[28*a:28*a+28,28*b:28*b+28]=x[np.random.choice(len(x))]

c*=255

import cv2

cv2.imwrite('nsample.png',c)