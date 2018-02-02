from helper import *
from solution import *


#Use for testing the training and testing processes of a model
def train_test_a_model(modelname, train_data, train_label, test_data, test_label, max_iter, learning_rate):
	print(modelname+" testing...")
	# max iteration test cases 
	for i, m_iter in enumerate(max_iter):
		w = logistic_regression(train_data, train_label, m_iter, learning_rate[1])
		Ain, Aout = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
		print("max iteration testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))
	# learning rate test cases
	for i, l_rate in enumerate(learning_rate):
		w = logistic_regression(train_data, train_label, max_iter[3], l_rate)
		Ain, Aout = accuracy(train_data, train_label, w), accuracy(test_data, test_label, w)
		print("learning rate testcase%d: Train accuracy: %f, Test accuracy: %f"%(i, Ain, Aout))
	print(modelname+" test done.")	


def test_logistic_regression():
	max_iter = [100, 200, 500,1000]
	learning_rate = [0.1, 0.2, 0.5]
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	try:
		train_test_a_model("logistic regression", train_data, train_label, test_data, \
							test_label, max_iter, learning_rate)
	except:
		print("Please finish logistic_regression() and cross_entropy_error() functions \n\
				before you run the test_logistic_regression() function.\n")


def test_thirdorder_logistic_regression():
	max_iter = [100, 200, 500,1000]
	learning_rate = [0.1, 0.2, 0.5]
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	try:
		new_train_data = thirdorder(train_data[:,1:3])
		new_test_data = thirdorder(test_data[:,1:3])
		train_test_a_model("3rd order logistic regression", new_train_data, train_label, \
						new_test_data, test_label, max_iter, learning_rate)
	except:
		print("Please finish thirdorder() function before you run\n\
				the test_thirdorder_logistic_regression() function.\n")


if __name__ == '__main__':
	test_logistic_regression()
	test_thirdorder_logistic_regression()
