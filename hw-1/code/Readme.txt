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
There are total three Python files including 'main.py', 'solution.py' and 'helper.py'. In all assignments, you only need to add your solution in 'solution.py' file following the given instruction. However, you might need to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like load data, extract features, etc. The 'main.py' is used to test your solution. 

Notes: Do not change anything in 'main.py' and 'helper,py' files. Only try to add your code to 'solution.py' file and keep function names and parameters unchanged.  


Submitting requirements
-----------------------
Please submit a write-up to briefly report your results and the ‘solution.py’ file to the Blackboard.


Free feel to email Yongjun Chen for any assistant.
Email address: yongjun.chen@wsu.edu




