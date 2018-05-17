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


import colorsys
import random
hcolors = [colorsys.hsv_to_rgb(r / 10,1,1) for r in range(20)]
frame = np.zeros((280 + 5 + 100,280,3))

im_canvas = frame[:280]
z_canvas = frame[285:]

def draw_img():
	im_canvas[:,:,0] = im
	im_canvas[:,:,1] = im
	im_canvas[:,:,2] = im

def draw_z_img():
	z_canvas[:] = 1
	h,w,_ = z_canvas.shape
	delta = w / len(z)
	for pos,color,value in zip(range(len(z)),hcolors,z):
		x = int(pos * delta)
		height = int(h * value)
		z_canvas[h - height:h,x + 1:int(x + delta)] = color
		z_canvas[0:h - height,x + 1:int(x + delta)] = 0

while cv2.getWindowProperty(Wcontrols, 0) >= 0:
	predict()
	draw_img()
	draw_z_img()
	cv2.imshow(Wimg,frame)
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()