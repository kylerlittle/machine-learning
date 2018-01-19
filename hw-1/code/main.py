from helper import *
from solution import *

def play_with_data():
	# show the data
	traindataloc = "../data/train.txt"
	nums = 2
	data = load_data(traindataloc)[0:nums,1:]
	[n,d]=data.shape
	w= math.floor(math.sqrt(d))
	data = np.reshape(data, (nums, w, w))
	show_images(data)


def play_with_features():
	#get data
	traindataloc, testdataloc = "../data/train.txt", "../data/test.txt"
	train_data, train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	show_features(train_data,train_label)


def test_accuracy():
	max_iter = [1, 3, 5, 10, 20]
	result_mi = [(0.973,0.948), (0.971, 0.950), (0.980, 0.962), (0.981, 0.959), (0.977, 0.948)]
	learning_rate = [0.1, 0.2, 0.3, 0.5, 1.0]
	result_lr = [(0.981, 0.962), (0.981, 0.962), (0.981, 0.962), (0.981, 0.962), (0.981, 0.962)]
	for i, m_iter in enumerate(max_iter):
		train_acc, test_acc = test_perceptron(m_iter, learning_rate[0], 1)
		if equal(train_acc, result_mi[i][0]) and equal(test_acc, result_mi[i][1]):
			print("testcase%d passed"% (i+1))
		else:
			print("testcase%d failed"% (i+1))
	for i, l_rate in enumerate(learning_rate):
		train_acc, test_acc = test_perceptron(max_iter[2], l_rate, 2)
		if equal(train_acc, result_lr[i][0]) and equal(test_acc, result_lr[i][1]):
			print("testcase%d passed"% (i+len(max_iter)+1))
		else:
			print("testcase%d failed"% (i+len(max_iter)+1))


def play_with_result():
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	max_iter = 10
	learning_rate = 0.5
	w = perceptron(train_data, train_label, max_iter, learning_rate)	
	show_result(test_data[:,1:3], test_label, w)


if __name__ == '__main__':
	play_with_data()
	play_with_features()
	test_accuracy()
	play_with_result()
