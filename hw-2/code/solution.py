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

def compute_transformation(x):
        third_transform = np.zeros(10)
        third_transform[0] = 1.0; third_transform[1] = x[0]
        third_transform[2] = x[1]; third_transform[3] = (x[0])**2.0
        third_transform[4] = x[0]*x[1]; third_transform[5] = (x[1])**2.0
        third_transform[6] = (x[0])**3.0; third_transform[7] = ((x[0])**2.0)*x[1]
        third_transform[8] = ((x[1])**2.0)*x[0]; third_transform[9] = (x[1])**3.0
        return third_transform

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
        transformed_data = np.zeros((len(data), 10))
        for index in np.arange(len(data)):
                transformed_data[index] = compute_transformation(data[index])
        return transformed_data

def accuracy(x, y, w):
        threshold = 0.5; correctlyClassified = 0
        for index in range(len(x)):
                predictedVal = 1. if (sigmoid(linear_signal(w,x[index]))) > threshold else -1.
                if predictedVal == y[index]:
                        correctlyClassified += 1
        return float(correctlyClassified) / len(x)
