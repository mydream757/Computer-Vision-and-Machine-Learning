import numpy as np
import matplotlib.pyplot as plt

#The initial values about data
num     = 201
std     = 20
#The values about original fomula
a       = 2
b       = 10
#generate noisy data
n       = np.random.rand(num)
nn      = n - np.mean(n)
x       = np.linspace(-100,100,num)
y1      = a * x + nn * std + b
#expected data from original fomula
y2      = a * x + b
#This things are used for computing least square error
sumXSquare = np.sum(x**2)
sumX = np.sum(x)
sumY = np.sum(y1)
sumXY = np.sum(x*y1)
#compute values about approximating line
pa = (sumXY - (sumX * sumY)/num)/(sumXSquare-(sumX**2)/num)
pb = -(sumX/num)*pa + sumY/num
#fomula of approximating line
y3 = pa * x + pb
#plot the data
plt.figure()
plt.plot(x, y1, 'b.', label='noisy')
plt.plot(x, y2, 'k--', label='clean')
plt.plot(x, y3, 'g--',label='approximating')
plt.legend(loc='lower right')
plt.show()

#show the difference between lines
plt.figure()
plt.plot(x, y1, 'b.', label='noisy')
plt.plot(x, y2, 'k--', label='clean')
plt.plot(x, y3, 'g--',label='approximating')
plt.legend(loc='lower right')
plt.xlim(65,85)
plt.ylim(65*a+b,85*a+b)
plt.show()
