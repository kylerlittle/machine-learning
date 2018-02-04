import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''

'''
Auxilary Functions
'''
def sigmoid(x):
        '''
        The probabilistic function (a standard sigmoid curve)
        '''
        return np.exp(x)/(1.+np.exp(x))

def linear_signal(w, x):
        '''
        Calculates the linear signal (i.e. the dot product between w and x)
        '''
        return np.dot(x,w)

def compute_grad_cross_entropy(x, y, w):
        '''
        Computes the gradient of the cross entropy function 
        '''
        grad_cross_entropy = np.zeros((x.shape[1], 1))
        for i in np.arange(len(x)):
                np.transpose(grad_cross_entropy)[0] += (y[i] * x[i]) / (1+np.exp(y[i] * linear_signal(w,x[i]))[0])
        grad_cross_entropy = -1./len(x) * grad_cross_entropy
        return grad_cross_entropy

def compute_transformation(x):
        '''
        Calculates the third-order polynomial transform of 2-dimensional input vector [1.0, x1, x2]
        In this case, we've passed in x as just [x1, x2], so I adjust indices accordingly
        '''
        third_transform = np.zeros(10)
        third_transform[0] = 1.0; third_transform[1] = x[0]
        third_transform[2] = x[1]; third_transform[3] = (x[0])**2.0
        third_transform[4] = x[0]*x[1]; third_transform[5] = (x[1])**2.0
        third_transform[6] = (x[0])**3.0; third_transform[7] = ((x[0])**2.0)*x[1]
        third_transform[8] = ((x[1])**2.0)*x[0]; third_transform[9] = (x[1])**3.0
        return third_transform

def logistic_regression(data, label, max_iter, learning_rate):
        '''
        The logistic regression classifier function. 
        Args:
        data: train data with shape (1561, 3), which means 1561 samples and each sample has 3 features.(1, symmetry, average internsity)
        label: train data's label with shape (1561,1). 1 for digit number 1 and -1 for digit number 5.
        max_iter: max iteration numbers
        learning_rate: learning rate for weight update
        
        Returns:
        w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
        '''
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
	'''
	This function is used for a 3rd order polynomial transform of the data.
	
	Args:
	data: input data with shape (:, 3) the first dimension represents total samples (training: 1561; testing: 424) and the second dimesion represents total features.
	
	Return:
	result: A numpy array format new data with shape (:,10), which using a 3rd order polynomial transformation to extend the feature numbers from 3 to 10. 
	The first dimension represents total samples (training: 1561; testing: 424) and the second dimesion represents total features
	'''
	transformed_data = np.zeros((len(data), 10))

	for index in np.arange(len(data)):
	    transformed_data[index] = compute_transformation(data[index])
	return transformed_data
	    


def accuracy(x, y, w):
        '''
        This function is used to compute accuracy of a logsitic regression model.
        
        Args:
        x: input data with shape (n, d), where n represents total data samples and d represents total feature numbers of a certain data sample.
        y: corresponding label of x with shape(n, 1), where n represents total data samples.
        w: the seperator learnt from logistic regression function with shape (d, 1), where d represents total feature numbers of a certain data sample.
        
        Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5, which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
        '''
        threshold = 0.5; correctlyClassified = 0
        for index in range(len(x)):
                predictedVal = 1. if (sigmoid(linear_signal(w,x[index]))) > threshold else -1.
                if predictedVal == y[index]:
                        correctlyClassified += 1
        return float(correctlyClassified) / len(x)
