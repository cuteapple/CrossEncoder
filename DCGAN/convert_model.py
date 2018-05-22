from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument('-G',metavar='path',help='path to g weights',default = 'G.h5')
ap.add_argument('-D',metavar='path',help='path to d weights',default = 'D.h5')
args = ap.parse_args()

def convertG():
	import G
	G.new_G()
