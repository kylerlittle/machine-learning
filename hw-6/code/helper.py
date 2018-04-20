import numpy as np 
import math
import matplotlib.pyplot as plt
import scipy.io as scio

def load_data(dataloc):
	data = scio.loadmat(dataloc)
	return data['A']


def show_images(data, p, i):
	fig = plt.figure()
	data = data
	plt.imshow(data[i,:].reshape((16,16)))
	fig.savefig('Reconstructed images%d with %d component'%(i,p))

