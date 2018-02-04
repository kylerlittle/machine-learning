import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''

'''
Auxilary Functions
'''
def sigmoid(x):
        return 1./(1.+np.exp(-x))

def linear_signal(w, x):
        return np.dot(x,w)

def compute_grad_cross_entropy(x, y, w):
        grad_cross_entropy = np.zeros((x.shape[1], 1))
        for i in np.arange(len(x)):
                np.transpose(grad_cross_entropy)[0] += (y[i] * x[i]) / (1+np.exp(y[i] * linear_signal(w,x[i]))[0])
        grad_cross_entropy = -1./len(x) * grad_cross_entropy
        return grad_cross_entropy

def logistic_regression(data, label, max_iter, learning_rate):
        # Initialize variables.
        w = np.zeros((data.shape[1], 1)); allowableError = 1.0e-6; t = 0
        gradient_cross_entropy = np.ones((data.shape[1], 1))   # initialize 'gradient_cross_entropy' to something nonzero
        # Run the algorithm
        while t < max_iter and abs(np.linalg.norm(gradient_cross_entropy)) > allowableError:
                gradient_cross_entropy = compute_grad_cross_entropy(data, label, w)
                w = w - (learning_rate * gradient_cross_entropy)
                t +=1
        return w

def thirdorder(data):
	pass

def accuracy(x, y, w):
        threshold = 0.5; correctlyClassified = 0
        for index in range(len(x)):
                predictedVal = 1. if (sigmoid(linear_signal(w,x[index]))) > threshold else -1.
                if predictedVal == y[index]:
                        correctlyClassified += 1
        return float(correctlyClassified) / len(x)
