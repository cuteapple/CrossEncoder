import cv2
import numpy as np

def nothing(x):
    pass

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
		print('set {} to {}'.format(i,z[i]))
	cv2.createTrackbar(str(i),Wcontrols,0,100, update_i)

#import G
#g = G.new_G(z_shape)
while(1):
    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()