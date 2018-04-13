Instruction of programming assignments for CptS 437: Introduction to Machine Learning

We will use the Python programming language for all assignments in this course. Specifically, we will use a few popular libraries (numpy, matplotlib, math) for scientific computing.

We expect that many of you already have some basic experience with Python and Numpy. We also provide basic guide to learn Python and Numpy.


Setup
-----
Download and install Anaconda with Python3.6 version:
- Download at the website: https://www.anaconda.com/download/
- Install Python3.6 version(not Python 2.7)
Anaconda will include all the Python libraries we need. 

Start programming:
Open Anaconda and choose Spyder to start your programming exercise.


Python & Numpy Tutorial
-----------------------
- Official Python tutorial: https://docs.python.org/3/tutorial/
- Official Numpy tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
- Good tutorial sources: http://cs231n.github.io/python-numpy-tutorial/ 


Dataset Descriptions
--------------------
We will use part of MNIST image dataset for all the assignments. All the data are in 'data' folder namely 'train.txt' and 'test.txt'. Here are the details of the dataset information:
		     samples   image size   labels 
Training Dataset      1561      16*16      1 or 5
Testing Dataset       424       16*16      1 or 5

We already extracted two features discussed in class, so you can directly use these features as your input. 
feature1: symmetry
feature2: average intensity
Here are the details of the feature information:
		    samples   feature numbers   labels 
Training Dataset     1561           2           1 or 5
Testing Dataset       424           2           1 or 5


Assignment Descriptions
-----------------------
There are total three Python files including 'main.py', 'solution.py' and 'helper.py'. In this assignment, you only need to add your solution in 'solution.py' file following the given instruction. However, you might need to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like load data, extract features, etc. The 'main.py' is used to test your solution. 

Notes: Do not change anything in 'main.py', 'helper,py' and any other files. Only try to add your code to 'solution.py' file and keep function names and parameters unchanged.  


LIBSVM tutorial
---------------
the libsvm library has been put into the code folder and already been imported to the 'solution.py' file. You can directly call the corresponding functions in 'solution.py' file. In this homework, you will need two functions name: 'svm_train' and 'svm_predict' to start your work. Following is how it works.

1. library setup
In the command line, change to location './libsvm/' and run 'make' command. Then you can use the certain functions in the the libsvm package.

2. Train a svm classifier:

Command:
model = svm_train(train_label, train_data, libsvm_options)

- train_label: an m by 1 list of training labels. m represents total training data samples.
- train_data: an m by n two dimension list. m represents total training data samples and n represents number of features for each data sample.
- libsvm_options: a string format of training options. You will using following options in your homework:
-c cost: set the parameter C of C-SVC, epsilon-SVR and nu-SVR (default 1)
-t kernel: set type of kernel function (default 3). 0: linear kernel; 1: polynomial kernel; 2: radial basis function kernel.

Here is an example to use svm_train:

suppose you have train_data and train_label, then set up libsvm options as:
libsvm_options = '-c 2 -t 1' and code:
model = svm_train(train_label, train_data, libsvm_options)

You will get a svm model with polynomial kernel saved on 'model' and it will also print following information:
**********************************
optimization finished, #iter = 254
nu = 0.322870
obj = -34.870094, rho = -0.069467
nSV = 504, nBSV = 504
Total nSV = 504
**********************************************

Where #iter means total iterations used to find the optimal solution; Total nSV means total number of support vectors in this model.

3. Predict label on test data

Command:
predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)

- test_label: an m by 1 list of prediction labels
- test_data: an m by n two dimension list. m represents total testing data samples and n represents number of features for each data sample.
- model: the output of svm_train function.
This function will return three values: predicted label, test accuracy and decision values with classify accuracy printing out.

Following is an example printing output after call this function:
**********************************************
Accuracy = 96.2264% (408/424) (classification)
**********************************************


Submitting requirements
-----------------------
Please submit a write-up to briefly report your results and the ‘solution.py’ file to the Blackboard.


Free feel to email Yongjun Chen for any assistant.
Email address: yongjun.chen@wsu.edu
