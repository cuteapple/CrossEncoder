from Dataset import NoizyData
import cv2
import numpy as np

window = 'window'
cv2.namedWindow(window)

dataset = NoizyData()
canvas = np.zeros((280+100,280,1))

def samples():
	while True:
		im,(c,n) = dataset.train()
		a = np.arange(len(im))
		np.random.shuffle(a)
		for i in a:
			yield im[i],c[i],n[i]

samples = samples()

while cv2.getWindowProperty(window, 0) >= 0:
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break
	if k == ord(' '):
		im,c,n = next(samples)
		canvas[:280,:280] = cv2.resize(im,(280,280),interpolation=cv2.INTER_NEAREST).reshape(280,280,-1)
		print(n, c)
		cv2.imshow(window,canvas)
			

cv2.destroyAllWindows()
