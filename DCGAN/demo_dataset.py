from Dataset import NoizyData, ZData
import cv2
import numpy as np

showD = True
showG = False

window = 'window'
cv2.namedWindow(window)

canvas = np.zeros((280+100,280,1))

def samples():
	dataset = NoizyData()
	while True:
		im,(c,n) = dataset.train_batch(1,1)
		a = np.arange(len(im))
		#np.random.shuffle(a)
		for i in a:
			yield im[i],c[i],n[i]

def zsamples():
	while True:
		[c,z],[c2,r] = next(ZData(1))
		yield c[0],z[0],c2[0],r[0]

if showD:	
	samples = samples()
if showG:
	zsamples = zsamples()

while cv2.getWindowProperty(window, 0) >= 0:
	k = cv2.waitKey(33) & 0xFF
	if k == 27:
		break
	if k == ord(' '):
		if showD:
			im,c,n = next(samples)
			canvas[:280,:280] = cv2.resize(im,(280,280),interpolation=cv2.INTER_NEAREST).reshape(280,280,-1)
			print(n, c)
			cv2.imshow(window,canvas)
		if showG:
			print('zg',*next(zsamples),sep='\n')

cv2.destroyAllWindows()
