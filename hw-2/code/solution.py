import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
        w = np.zeros((data.shape[1], 1))
        pass


def thirdorder(data):
	pass


def sigmoid(x):
        return 1./(1.+np.exp(-x))

def accuracy(x, y, w):
        threshold = 0.5; correctlyClassified = 0
        for index in range(x):
                predictedVal = 1. if (sigmoid(np.dot(np.transpose(w),x[index]))) > threshold else -1.
                if predictedVal == y[index]:
                        correctlyClassified += 1
        return float(correctlyClassified) / len(x)


