from helper import load_data, show_images
from solution import pca, reconstruction, reconstruct_error
import numpy as np

def test_pca():
	dataloc = "../data/USPS.mat"
	A = load_data(dataloc)
	ps = [10, 50, 100, 200]
	for p in ps:
		U_p, A1 = pca(A, p)
		Re_A = reconstruction(U_p, A1)
		error = reconstruct_error(A, Re_A)
		print(error)
		show_images(Re_A, p, 1)
		show_images(Re_A, p, 2)
	
if __name__ == '__main__':
	test_pca()
