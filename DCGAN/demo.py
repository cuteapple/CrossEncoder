import cv2
import numpy as np

CONTROL_ONLY = True

z_shape = (20,)

if not CONTROL_ONLY:
	print('initializing ...')
	import G
	print('loading model ...')
	g = G.new_G(z_shape)
	g.load_weights('G.h5')

	import D
	d = D.D().model
	d.load_weights('D.h5')


z = np.random.normal(size=z_shape)
z[:10] = 0
outG = np.zeros((280,280))
outD = np.zeros(10)

update = True
def predict():
	global outG
	global outD
	global update
	print('predict', z)
	outG = g.predict(z.reshape(1,*z_shape))[0]
	outD = d.predict(outG.reshape(1,28,28,1))[0]
	print('D',outD)

if CONTROL_ONLY:
	def predict():
		global outD
		outD = z
		print('pass predict')

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
		#print('set {} to {}'.format(i,z[i]))
	cv2.createTrackbar(str(i),Wcontrols,int(z[i] * steps),steps, update_i)


import colorsys
import random
hcolors = np.array([colorsys.hsv_to_rgb(r / 10,1,1) for r in range(20)])
frame = np.zeros((280 + 100,280,3))

im_canvas = frame[:280]
z_canvas = frame[280:]

def draw_img():
	im = cv2.resize(outG,(im_canvas.shape[0],im_canvas.shape[1]),interpolation=cv2.INTER_NEAREST)
	im_canvas[:,:,0] = im
	im_canvas[:,:,1] = im
	im_canvas[:,:,2] = im

def draw_z_img():
	z_canvas[:] = (.2,.2,.2)
	h,w,_ = z_canvas.shape
	delta = w / len(z)

	#draw input z
	for pos,color,value in zip(range(len(z)),hcolors,z):
		x1 = int(pos * delta)
		x2 = x1 + int(delta)
		x1 += 1
		x2 -= 1
		height = int(h * value)
		z_canvas[h - height:h, x1:x2] = color

	#draw out D (D(G(z)))
	for pos,color,value in zip(range(len(outD)),hcolors,outD):
		x = int(pos * delta)
		height = int(h * value)
		padding = int(delta / 4)
		z_canvas[h - height:h, x + padding:int(x + delta - padding)] = color * .5

	
	#draw text
	for i in range(10):
		x1 = int(i * delta)
		x2 = x1 + int(delta)
		x1 += 1
		#print(x1)
		cv2.putText(z_canvas,str(i),(x1,h-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(1,1,1),1,cv2.FILLED,False)
		
while cv2.getWindowProperty(Wcontrols, 0) >= 0:
	if update:
		update = False
		predict()
		draw_img()
		draw_z_img()
	cv2.imshow(Wimg,frame)
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break
	if k == ord(' '):
		z[10:]=np.random.normal(0,1,10)
		update = True

cv2.destroyAllWindows()