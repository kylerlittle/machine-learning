import numpy as np 
from helper import *

'''
Homework1: perceptron classifier
'''
def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):
        '''
	This function is used for plot image and save it.

	Args:
	data: Two images from train data with shape (2, 16, 16). The shape represents total 2
	images and each image has size 16 by 16. 

	Returns:
	Do not return any arguments, just save the images you plot for your report.
	'''
        for index in range(len(data)):
                imgplot = plt.imshow(data[index])
                plt.show(imgplot)                  # After each figure pops up, I save it for the report


                
def assign_sym_col(label):
        '''
        This function assigns a red color & '*' symbol to labels '+1' (i.e. digit number 1) and a blue color & '+'
        symbol to labels '-1' (i.e. digit number 5).

        Args:
        label: train data's label with shape (1561,1). 
	1 for digit number 1 and -1 for digit number 5.

        Returns:
        A 2-tuple of two lists, each of length 1561
        '''
        symbols = []; colors = []
        for val in label:
                if val == 1:
                        sym = '*'; col = 'red'
                else:
                        sym = '+'; col = 'blue'
                symbols.append(sym); colors.append(col)
        return (symbols, colors)


def show_features(data, label):
        '''
        This function is used for plot a 2-D scatter plot of the features and save it.
        
        Args:
        data: train features with shape (1561, 2). The shape represents total 1561 samples and 
        each sample has 2 features.
        label: train data's label with shape (1561,1). 
        +1 for digit number 1 and -1 for digit number 5.

        Returns:
        Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
        '''
        fig, ax = plt.subplots()
        (symbols, colors) = assign_sym_col(label)
        for _s, c, _x, _y in zip(symbols, colors, data[:,0], data[:,1]):
                ax.scatter(_x, _y, s=50, marker=_s, c=c)
        plt.title('Training Data'); plt.xlabel('Symmetry Classifier'); plt.ylabel('Average Intensity')
        plt.show()

        

def perceptron(data, label, max_iter, learning_rate):
        '''
        The perceptron classifier function.
        
        Args:
        data: train data with shape (1561, 3), which means 1561 samples and 
        each sample has 3 features.(1, symmetry, average internsity)
        label: train data's label with shape (1561,1).
        1 for digit number 1 and -1 for digit number 5.
        max_iter: max iteration numbers
        learning_rate: learning rate for weight update

        Returns:
        the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
        '''
        t = 0; w = np.zeros((1,len(data[0])))
        while t < max_iter:
                for data_index, data_val in enumerate(data):
                        if sign(np.dot(w[0], data_val)) != label[data_index]:   # Misclassified Item
                                w += learning_rate * label[data_index] * data_val  # Update weight
                t += 1
        return w


def show_result(data, label, w):
        '''
        This function is used for plot the test data with the separators and save it.
        
        Args:
        data: test features with shape (424, 2). The shape represents total 424 samples and
        each sample has 2 features.
        label: test data's label with shape (424,1).
        +1 for digit number 1 and -1 for digit number 5.

        Returns:
        Do not return any arguments, just save the image you plot for your report.
        '''
        # Set up plot
        fig, ax = plt.subplots()

        # Overlay the linear separator
        X = data[:,0]
        Y = np.zeros(len(X))
        line_slope = -w[0][1] / w[0][2]; y_intercept = -w[0][0] / w[0][2]
        for index in range(len(Y)):
                Y[index] = (line_slope * X[index]) + y_intercept
        ax.plot(X, Y, color='k', linewidth=2)

        # Add scatter plot
        (symbols, colors) = assign_sym_col(label)
        for _s, c, _x, _y in zip(symbols, colors, data[:,0], data[:,1]):
                ax.scatter(_x, _y, s=50, marker=_s, c=c)

        # Configure scaling, title, and xy labels
        ax.set_xlim([X.min(), X.max()]); ax.set_ylim([-1., 0.2])
        plt.title('Test Data'); plt.xlabel('Symmetry Classifier'); plt.ylabel('Average Intensity')
        plt.show()


#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc


