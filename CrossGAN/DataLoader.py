import cv2
import glob
import numpy

class DataLoader:
	''' load data '''
	def __init__(self, dir, size = None):
		self.dir = dir
		self.size = size
		self.files = glob.glob(dir + '/*')

	def load(self,batch_size):
		''' return a batch of data '''
		ch = numpy.random.choice(self.files, batch_size)
		return numpy.stack([self.imread(f) for f in ch])

	def imread(self,file):
		''' read and preprocess images from file '''
		im = cv2.imread(file)
		if im is None:
			raise RuntimeError('{} is not image'.format(file))
		if self.size:
			im = cv2.resize(im,dsize = self.size)
		im = im.astype('float')/255
		return im
	
	@staticmethod
	def save_image(img,file):
		''' reverse preprocess (no resize) and save to *file* '''
		img*=255
		cv2.imwrite(file,img)

def test():
	l = DataLoader('testim',(64,64))
	data = l.load(20)
	for i,im in enumerate(data):
		DataLoader.save_image(im,file = 'testimo/{}.png'.format(i))

if __name__ == '__main__':
	test()