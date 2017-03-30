阿哲


```python

#coding=utf-8
import numpy as np
from numpy import *

#第一個範例
#矩陣相加錯誤示範
x = [10,20,30]
y = [40,50,60]
print x+y

#正確示範
a = np.array([10, 20, 30])
b = np.array([20, 40, 60])
print a+b

#第二個範例
a=np.array([2,4,5])
b=np.array([-3,2,-1])

la=np.sqrt(a.dot(a))
lb=np.sqrt(b.dot(b))
print("----計算向量長度---")
print (la,lb)

cos_angle=a.dot(b)/(la*lb)

print("----計算cos ----")
print (cos_angle)

angle=np.arccos(cos_angle)

print("----計算夾角(單位為π)----")
print (angle)

angle2=angle*360/2/np.pi
print("----轉換單位為角度----")
print (angle2)

a=np.array([[2, 5], [3, 2]])
b=np.array([[2, 3], [2, 5]])
c=np.mat([[2, 4], [2, 3]])
d=np.mat([[1, 2], [3, 4]])
e=np.dot(a,b)
f=np.dot(c,d)
print("----乘法運算----")
print (a*b)
print (c*d)
print("----矩陣相乘----")
print (e)
print (f)

a=np.random.randint(1, 10, (3, 5))
#a=np.random.randint(1, 10, 8)
print (a)

a = mat([[1, 3, -1], [2, 0, 1], [3, 2, 1]])

print linalg.det(a)

from matplotlib import pyplot

x = np.arange(0,10,0.1)
y = np.sin(x)
pyplot.plot(x,y)
pyplot.show()


```





































Data Science from Scratch
=========================

Here's all the code and examples from my book __[Data Science from Scratch](http://joelgrus.com/2015/04/26/data-science-from-scratch-first-principles-with-python/)__. The `code` directory contains Python 2.7 versions, and the `code-python3` direction contains the Python 3 equivalents. (I tested them in 3.5, but they should work in any 3.x.)

Each can be imported as a module, for example (after you cd into the /code directory):

```python
from linear_algebra import distance, vector_mean
v = [1, 2, 3]
w = [4, 5, 6]
print distance(v, w)
print vector_mean([v, w])
```
  
Or can be run from the command line to get a demo of what it does (and to execute the examples from the book):

```bat
python recommender_systems.py
```  

Additionally, I've collected all the [links](https://github.com/joelgrus/data-science-from-scratch/blob/master/links.md) from the book.

And, by popular demand, I made an index of functions defined in the book, by chapter and page number. 
The data is in a [spreadsheet](https://docs.google.com/spreadsheets/d/1mjGp94ehfxWOEaAFJsPiHqIeOioPH1vN1PdOE6v1az8/edit?usp=sharing), or I also made a toy (experimental) [searchable webapp](http://joelgrus.com/experiments/function-index/).

## Table of Contents

1. Introduction
2. A Crash Course in Python
3. [Visualizing Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/visualizing_data.py)
4. [Linear Algebra](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/linear_algebra.py)
5. [Statistics](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/statistics.py)
6. [Probability](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/probability.py)
7. [Hypothesis and Inference](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/hypothesis_and_inference.py)
8. [Gradient Descent](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/gradient_descent.py)
9. [Getting Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/getting_data.py)
10. [Working With Data](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/working_with_data.py)
11. [Machine Learning](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/machine_learning.py)
12. [k-Nearest Neighbors](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/nearest_neighbors.py)
13. [Naive Bayes](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/naive_bayes.py)
14. [Simple Linear Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/simple_linear_regression.py)
15. [Multiple Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/multiple_regression.py)
16. [Logistic Regression](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/logistic_regression.py)
17. [Decision Trees](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/decision_trees.py)
18. [Neural Networks](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/neural_networks.py)
19. [Clustering](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/clustering.py)
20. [Natural Language Processing](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/natural_language_processing.py)
21. [Network Analysis](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/network_analysis.py)
22. [Recommender Systems](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/recommender_systems.py)
23. [Databases and SQL](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/databases.py)
24. [MapReduce](https://github.com/joelgrus/data-science-from-scratch/blob/master/code/mapreduce.py)
25. Go Forth And Do Data Science
