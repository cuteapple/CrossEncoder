import cv2
import numpy as np


print('initializing ...')
import G


print('loading model ...')
z_shape = (20,)
g = G.new_G(z_shape)
g.load_weights('G.h5')

z = np.random.normal(size=z_shape)
z[:10] = 0

update = True
def predict():
	global im
	global update

	if not update:
		return
	update = False
	print('predict', z)
	im = g.predict(z.reshape(1,*z_shape)).reshape(28,28,1)
	im = cv2.resize(im,(280,280))

Wcontrols = 'control'
Wimg = 'result'
cv2.namedWindow(Wcontrols,cv2.WINDOW_NORMAL)
cv2.namedWindow(Wimg,cv2.WINDOW_AUTOSIZE)



# create trackbars
for i in range(10):
	steps = 100 # step of trackbar
	def update_i(x,i=i):
		z[i] = x / steps
		global update
		update = True
		print('set {} to {}'.format(i,z[i]))
	cv2.createTrackbar(str(i),Wcontrols,int(z[i] * steps),steps, update_i)
	
def random_z():
	z[10:] = np.random.normal(size=10)
cv2.createButton('Random Z',random_z)

while cv2.getWindowProperty(Wcontrols, 0) >= 0:
	predict()

	cv2.imshow(Wimg,im)
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()