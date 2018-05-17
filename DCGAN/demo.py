import cv2
import numpy as np

CONTROL_ONLY = True

z = np.random.normal(size=(20,))
z_class = z[:10]
z_noise = z[10:]
z_class[:] = 0
z_noise = np.random.normal(size = z_noise.shape)
z_update = True

def setz(i,value):
	'''set z and update z_update'''
	z_class[i] = value
	global z_update
	z_update = True

outG = np.zeros((28,28))
outD = np.zeros(10)




if not CONTROL_ONLY:
	def init_models():
		print('initializing ...')
		import G
		import D

		print('loading model ...')
		g = G.new_G(z.shape)
		g.load_weights('G.h5')

		d = D.D().model
		d.load_weights('D.h5')

		global predict
		def predict():
			global outG
			global outD
			global z_update
			print('predict', z)
			outG = g.predict(z.reshape(1,*z.shape))[0]
			outD = d.predict(outG.reshape(1,28,28,1))[0]
			print('D',outD)
else:
	def init_models():
		print('no model mode')
		global predict
		def predict():
			global outD
			outD = z_class

init_models()

class window_trackbars:
	def __init__(self,name='bars'):
		self.name = name
		cv2.namedWindow(name,cv2.WINDOW_NORMAL) # autosize not work (too small)
		steps = 100

		for i in range(len(z_class)):
			cv2.createTrackbar(str(i), self.name, int(z_class[i] * steps), steps, lambda x,i=i: setz(i,x / steps))

	def update(self):
		for i in range(len(z_class)):
			cv2.setTrackbarPos(str(i),self.name,z_class[i])


class window_result:
	def __init__(self, name='result'):
		self.name = name
		cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)

		import colorsys
		import random
		
		self.zcolors = np.array([colorsys.hsv_to_rgb(r / 10,1,1) for r in range(20)])
		self.frame = np.zeros((280 + 100,280,3))

		self.canvas_img = self.frame[:280]
		self.canvas_z = self.frame[280:]

	def render(self):
		self.draw_img()
		self.draw_z_img()
		cv2.imshow(self.name,self.frame)

	def alive(self):
		return cv2.getWindowProperty(self.name, 0) >= 0

	def draw_img(self):
		im = cv2.resize(outG,(self.canvas_img.shape[0],self.canvas_img.shape[1]),interpolation=cv2.INTER_NEAREST)
		self.canvas_img[:,:,0] = im
		self.canvas_img[:,:,1] = im
		self.canvas_img[:,:,2] = im

	def draw_z_img(self):
		self.canvas_z[:] = (.2,.2,.2)
		h,w,_ = self.canvas_z.shape
		delta = w / len(z)

		#value : [0,1]
		def drawbar(color,left,right,value,padding=0):
			dh = int(value * h)
			left = left + padding
			right = right - padding
			self.canvas_z[h - dh:h,left:right] = color
		def impl():
			for i,(color,z,zd) in enumerate(zip(self.zcolors,z_class,outD)):
				x1 = int(i * delta)
				x2 = x1 + int(delta)
			
				drawbar(color,x1,x2,z,1)
				drawbar(color * 0.5,x1,x2,zd,int(delta / 4))

				#label
				cv2.putText(self.canvas_z,str(i),(x1,h - 5),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(1,1,1),1,cv2.FILLED,False)
		impl()


wbar = window_trackbars()
wresult = window_result()


#
# graph selection
#
class window_graph_selection():
	def __init__(self,name='im-selection'):
		self.name = name

		w,h = 32,32

		canvas = np.zeros((w,h,3))
		z_map = np.zeros((w,h,10))

		dx3 = w / 4
		dx4 = w / 8
		dy = h / 4
		center = [[dx3,dy],[dx3 * 2,dy],[dx3 * 3,dy],
			[dx4,dy * 2],[dx4 * 3,dy * 2],[dx4 * 5,dy * 2],[dx4 * 7,dy * 2],
			[dx3,dy * 3],[dx3 * 2,dy * 3],[dx3 * 3,dy * 3]]
		#center = np.random.uniform(0,32,(10,2))

		for x in range(w):
			for y in range(h):
				pos = np.array([x,y])
				color = np.zeros(3)
				z = np.zeros(10)
				total_weight = 0
				for i in range(10):
					dist = np.sqrt(np.sum(np.square(pos - center[i])))
					w = 1 / (dist + 0.1)
					total_weight += w
					color += w * wresult.zcolors[i]
					z[i] = w
				color /= total_weight
				z /= total_weight
				z_map[y,x] = z 
				canvas[y,x] = color

		canvas = cv2.resize(canvas,(0,0),fx=12,fy=12,interpolation=cv2.INTER_CUBIC)
		z_map = cv2.resize(z_map,(0,0),fx=12,fy=12,interpolation=cv2.INTER_CUBIC)
		cv2.namedWindow(self.name)
		cv2.imshow(self.name,canvas)

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

			z_class[:] = z_map[iy,ix]
			global z_update
			z_update = True
			
		cv2.setMouseCallback('im-selection',onmouse)

wselection = window_graph_selection()

while wresult.alive():
	if z_update:
		z_update = False
		predict()
		wresult.render()
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break
	if k == ord(' '):
		z_noise[:] = np.random.normal(size=z_noise.shape)
		z_update = True

cv2.destroyAllWindows()