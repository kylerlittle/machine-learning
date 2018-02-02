from matplotlib import pyplot as plt
import numpy as np 
import math
import matplotlib.pyplot as plt

def load_data(dataloc):
	data = np.loadtxt(dataloc, unpack='true')
	data = np.transpose(data, (1,0))
	return data	


def extract_feature(image):
	image = np.reshape(image, (16, 16))
	flip_image = np.flip(image, 1)
	diff = abs(image-flip_image)
	sys  = -sum(sum(diff))/256
	intense = sum(sum(image))/256
	return sys, intense


def load_features(dataloc):
	data = load_data(dataloc)
	n, _ = data.shape
	data_set = []
	for i in range(n):
		label = 1 if data[i, 0]==1 else -1
		image = data[i, 1:]
		sys, intense = extract_feature(image)
		data_set.append([label, 1, sys, intense])
	return np.array(data_set)[:,1:], np.array(data_set)[:,0]


