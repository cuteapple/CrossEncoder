import cv2
import numpy as np

CONTROL_ONLY = False

z_shape = (20,)

z = np.random.normal(size=z_shape)
z_class = z[:10]
z_noise = z[10:]
z_class[:] = 0
z_noise = np.random.normal(size = z_noise.shape)
z_update = True

outG = np.zeros((28,28))
outD = np.zeros(10)


if not CONTROL_ONLY:
	def init_models():
		print('initializing ...')
		import G
		import D

		global g
		global d

		print('loading model ...')
		g = G.new_G(z_shape)
		g.load_weights('G.h5')

		d = D.D().model
		d.load_weights('D.h5')

	def predict():
		global outG
		global outD
		global z_update
		print('predict', z)
		outG = g.predict(z.reshape(1,*z_shape))[0]
		outD = d.predict(outG.reshape(1,28,28,1))[0]
		print('D',outD)
else:
	def init_models():
		print('pass init models')
	def predict():
		global outD
		outD = z_class
		print('no model loaded')


Wcontrols = 'control'
Wimg = 'result'
cv2.namedWindow(Wcontrols,cv2.WINDOW_NORMAL)
cv2.namedWindow(Wimg,cv2.WINDOW_AUTOSIZE)



# create trackbars
for i in range(10):
	steps = 100 # step of trackbar
	def update_i(x,i=i):
		z[i] = x / steps
		global z_update
		z_update = True
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
		cv2.putText(z_canvas,str(i),(x1,h - 5),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(1,1,1),1,cv2.FILLED,False)


#
# graph selection
#
def gss():

	w = 32
	h = 32

	canvas = np.zeros((w,h,3))
	zs = np.zeros((w,h,10))

	dx3 = w / 4
	dx4 = w / 8
	dy = h / 4
	poss = [[dx3,dy],[dx3 * 2,dy],[dx3 * 3,dy],
		[dx4,dy * 2],[dx4 * 3,dy * 2],[dx4 * 5,dy * 2],[dx4 * 7,dy * 2],
		[dx3,dy * 3],[dx3 * 2,dy * 3],[dx3 * 3,dy * 3]]
	#poss = np.random.uniform(0,32,(10,2))

	for x in range(w):
		for y in range(h):
			pos = np.array([x,y])
			color = np.zeros(3)
			z = np.zeros(10)
			weight = 0
			for i in range(10):
				dist = np.sqrt(np.sum(np.square(pos - poss[i])))
				w = 1 / (dist + 0.1)
				weight += w
				color += w * hcolors[i]
				z[i] = w
			color /= weight
			z /= weight
			zs[y,x] = z 
			canvas[y,x] = color

	canvas = cv2.resize(canvas,(0,0),fx=12,fy=12,interpolation=cv2.INTER_CUBIC)
	zs = cv2.resize(zs,(0,0),fx=12,fy=12,interpolation=cv2.INTER_CUBIC)
	cv2.namedWindow('im-selection')
	cv2.imshow('im-selection',canvas)

	hold = False
	ix,iy = 0,0
	def onmouse(event,x,y,flags,param):
		nonlocal hold,ix,iy
		if event == cv2.EVENT_LBUTTONDOWN:
			hold = True
			ix,iy = x,y

		elif event == cv2.EVENT_MOUSEMOVE:
			if not hold:
				return
			ix,iy = x,y

		elif event == cv2.EVENT_LBUTTONUP:
			hold = False

		#print(ix,iy)
		#print(zs[iy,ix])

		z = zs[iy,ix]

		for i in range(10):
			cv2.setTrackbarPos(str(i),Wcontrols,int(z[i] * 100))

	cv2.setMouseCallback('im-selection',onmouse)

gss()		



while cv2.getWindowProperty(Wcontrols, 0) >= 0:
	if z_update:
		z_update = False
		predict()
		draw_img()
		draw_z_img()
	cv2.imshow(Wimg,frame)
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break
	if k == ord(' '):
		z[10:] = np.random.normal(0,1,10)
		z_update = True

cv2.destroyAllWindows()