#coding=utf-8
import numpy as np
from numpy import *

a = np.array([10, 20, 30])
b = np.array([20, 40, 60])
print a+b

#第二個範例
a=np.array([5,3,2])
b=np.array([-3,1,-5])

la=np.sqrt(a.dot(a))
lb=np.sqrt(b.dot(b))
print("----計算向量長度---")
print (la,lb)