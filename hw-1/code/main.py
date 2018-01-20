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
	print("play with data done!")


def play_with_features():
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	show_features(train_data[:,1:3],train_label)
	print("play with features done!")


def test_accuracy():
	max_iter = [10, 30, 50, 100, 200]
	result_mi = [(0.973,0.948), (0.971, 0.950), (0.980, 0.962), (0.981, 0.959), (0.977, 0.948)]
	learning_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
	result_lr = [(0.981, 0.962), (0.981, 0.962), (0.981, 0.962), (0.981, 0.962), (0.981, 0.962)]
	for i, m_iter in enumerate(max_iter):
		_, train_acc, test_acc = test_perceptron(m_iter, learning_rate[0])
		print("Case %d train accuracy:%f  test accuracy: %f"%(i+1, train_acc, test_acc))
	for i, l_rate in enumerate(learning_rate):
		_, train_acc, test_acc = test_perceptron(max_iter[4], l_rate)
		print("Case %d train accuracy:%f  test accuracy: %f"%(i+6,train_acc, test_acc))
	print("accuracy test done!")


def play_with_result():
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	max_iter = 10
	learning_rate = 0.5
	w = perceptron(train_data, train_label, max_iter, learning_rate)	
	show_result(test_data[:,1:3], test_label, w)
	print("play with result done!")


if __name__ == '__main__':
	## test question (a)
	play_with_data()
	## test question (b)
	play_with_features()
	## test question (c)
	test_accuracy()
	# test question (d)
	play_with_result()