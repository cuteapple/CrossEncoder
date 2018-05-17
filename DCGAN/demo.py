import cv2
import numpy as np

def nothing(x):
    pass

update = True
z_shape = (20,)
z = np.random.normal(size=z_shape)

# Create a black image, a window
#img = np.zeros((300,512,3), np.uint8)
Wcontrols = 'control'
cv2.namedWindow(Wcontrols,cv2.WINDOW_NORMAL)

Wimg = 'result'
cv2.namedWindow(Wimg,cv2.WINDOW_AUTOSIZE)


# create trackbars for color change
for i in range(10):
	def update_i(x,i=i):
		z[i] = x / 100
		global update
		update = True
		print('set {} to {}'.format(i,z[i]))
	cv2.createTrackbar(str(i),Wcontrols,0,100, update_i)

print('loading model ...')
import G
g = G.new_G(z_shape)
g.load_weights('G.h5')

def predict():
	if not update:
		return
	
	global im
	im = g.predict(z.reshape(1,*z_shape)).reshape(28,28,1)

while cv2.getWindowProperty(Wcontrols, 0) >= 0:
	predict()

	cv2.imshow(Wimg,im)
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()