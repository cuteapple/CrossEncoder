import D
import cv2
import numpy as np

window = 'window'
cv2.namedWindow(window)

canvas = np.zeros((280*2,280*5,1))

def samples():
	data = D.data(10,0.6,0.5)
	while True:
		x,y = next(data)
		for x,y in zip(x,y):
			yield x,y
samples = samples()
while cv2.getWindowProperty(window, 0) >= 0:
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break
	if k == ord(' '):
		x,y = next(samples)
		canvas = cv2.resize(x,(280,280),interpolation=cv2.INTER_NEAREST).reshape(280,280,-1)
		print(y)
		cv2.imshow(window,canvas)

cv2.destroyAllWindows()
