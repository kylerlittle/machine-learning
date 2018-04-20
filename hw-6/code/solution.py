import numpy as np
'''
Homework6: Principal Component Analysis

Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a certain shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix. You may use it for the reconstruct_error function.

'''
def pca(A, p):
	'''
	Principal component analysis function.

	Args:
	A: the data with shape (3000, 256). 3000 is the total number of samples and 256 is the total features/values of each sample.
	p: the number of principal components. A scatter number.

	Returns:
	U_p: 'p' principal components with shape (256, p).
	A1: The reduced data matrix after PCA with shape (p, 3000).
	'''
	pass


def reconstruction(U, A1):
	'''
	Reconstruct data function.

	Args:
	U: 'p' principal components with shape (256, p).
	A1: The reduced data matrix after PCA with shape (p, 3000).

	Return:
	Re_A: The reconstructed matrix with shape (3000, 256)
	'''
	pass


def reconstruct_error(A, B):
	'''
	reconstruction error function.

	Args: 
	A & B: Two matrices needed to be compared with shape (3000, 256).

	Return: 
	error: the Frobenius norm's square of the matrix A-B. A scatter number.
	'''
	pass

