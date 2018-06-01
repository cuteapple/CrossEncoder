from noisy import data_generator
import cv2
import numpy as np

def image():
	w = h = 28 * 5
	canvas = np.zeros((h * 2,w * 5,1))
	g = data_generator(5,5)
	while True:
		data,label = next(g)
		for i,im in enumerate(data[:5]):
			canvas[:h,i * w:i * w + w] = cv2.resize(im,(h,w),interpolation=cv2.INTER_NEAREST).reshape(h,w,1)
		for i,im in enumerate(data[5:10]):
			canvas[h:,i * w:i * w + w] = cv2.resize(im,(h,w),interpolation=cv2.INTER_NEAREST).reshape(h,w,1)
		yield canvas

images = image()

window = 'noisy'
cv2.namedWindow(window)
cv2.imshow(window,next(images))
while cv2.getWindowProperty(window, 0) >= 0:
	k = cv2.waitKey(33) & 0xFF
	if k == ord(' '):
		cv2.imshow(window,next(images))

cv2.destroyAllWindows()