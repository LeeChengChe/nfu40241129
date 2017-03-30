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
