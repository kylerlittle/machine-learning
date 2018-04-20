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
We will use USPS dataset for this assignments. The USPS dataset is in the “data” folder: USPS.mat. The whole data has already been loaded into the matrix A. The matrix A contains all the images of size 16 × 16. Each of the 3000 rows in A corresponds to the image of one handwritten digit (between 0 and 9). 


Assignment Descriptions
-----------------------
There are total three Python files including 'main.py', 'solution.py' and 'helper.py'. In this assignment, you only need to add your solution in 'solution.py' file following the given instruction. However, you might need to read all the files to fully understand the requirement. 

The 'helper.py' includes all the helper functions for the assignments, like load data, show images, etc. The 'main.py' is used to test your solution. 

Notes: Do not change anything in 'main.py' and 'helper,py' files. Only try to add your code to 'solution.py' file and keep function names and parameters unchanged.  


Helper functions
----------------
In this assignment, you may want to use following helper functions:
- np.linalg.svd(): compute the singular value decomposition on a given matrix. 
- np.dot(): matrices multiplication operation.
- np.mean(): compute the mean value on a given matrix.
- np.ones(): generate a all '1' matrix with a certain shape.
- np.transpose(): matrix transpose operation.
- np.linalg.norm(): compute the norm value of a matrix. You may use it for the reconstruct_error function.



Submitting requirements
-----------------------
Please submit a write-up to briefly report your results and the ‘solution.py’ file to the Blackboard.

Feel free to email Yongjun Chen for any assistant.
Email address: yongjun.chen@wsu.edu




