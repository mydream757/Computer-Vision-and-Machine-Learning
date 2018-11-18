import numpy as np
import matplotlib.pyplot as plt

# x  : x-coordinate data
def fun(x):
	# f = np.sin(x) * (1 / (1 + np.exp(-x)))
	f = np.abs(x) * np.sin(x)
	return f
def polyFun(x,coEff,k):
    temp = np.ones(x.shape)
    for i in range(k):
        if i==0:
            temp *= coEff[k-1]
        temp = temp + coEff[k-i-1]*(x**i)
    return temp

#the number and standard deviation of data
num     = 1001
std     = 5
#generate data
n       = np.random.rand(num)
nn      = n - np.mean(n)
x       = np.linspace(-10,10,num)
#y1: clean data
y1      = fun(x)
#y2: noisy data
y2      = y1 + nn * std
#max p
p = 31
result = []
for i in range(p):
    coEff = np.polyfit(x,y2,i)
    #y3: polynomial approximation data
    y3 = polyFun(x,coEff,i+1)

    plt.figure()
    plt.title("p=%d Result"%i)
    plt.plot(x, y1,'b.',label='clean')
    plt.plot(x, y2,'k.',label='noisy')
    plt.plot(x, y3,'r.',label='p-approximation')
    plt.legend(loc='lower right')
    plt.show()
    result.append(np.sum((y3-y2)**2))

plt.figure()
xs=[]
for k in range(p):
    xs.append(k)
plt.title("p=%d Error"%i)
plt.plot(xs,result,'b.',label='error')
plt.legend(loc='upper right')
plt.show()
