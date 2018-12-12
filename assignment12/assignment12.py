import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg

num     = 1001
std     = 5

# x  : x-coordinate data
# y1 : (clean) y-coordinate data
# y2 : (noisy) y-coordinate data

def fun(x):

	# f = np.sin(x) * (1 / (1 + np.exp(-x)))
	f = np.abs(x) * np.sin(x)

	return f

def create_A(p, x, l):
    A = np.ones((x.shape))
    for i in range(p):
        if i != 0:
            A = np.vstack((A,pow(x,i)))
    A = A.T

    return A
def create_B(n,l,y):
    b = y.reshape((n,1))
    return b
def make_l(p):
    e = np.eye(p)
    return e

def polyFunc(c,x):
    y = np.zeros(x.shape)

    for i in range(c.size):
        y = y + c[i]*pow(x,i)
    return y

n       = np.random.rand(num)
nn      = n - np.mean(n)
x       = np.linspace(-10,10,num)
y1      = fun(x) 			# clean points
y2      = y1 + nn * std		# noisy points


for i in range(10):
    p = i+6         # p is 6 to 15
    plt.figure('asdf')
    for k in range(4):
        l = pow(2,k*50-100) #    l = pow(2,k-100) # lambda is 2^(-100) to 2^100
        A = create_A(p,x,l)
        b = create_B(num,l,y2)
        coeff = np.dot(np.dot(linalg.pinv(np.dot(A.T,A)+pow(2,k*50-100)*make_l(p)),A.T),b)
		y3 = polyFunc(coeff,x)
		r = np.dot(A,coeff)-b
        energy = r**2 + l*np.dot(coeff.T,coeff)*np.ones((num,1))
        print('P: ', p)
        plt.subplot(2,2,k+1)
        plt.plot(x,y1,'b.', label = 'clean')
        plt.plot(x,y2,'k.', label = 'noisy')
        plt.plot(x,y3,'r.', label = 'poly fit with l')
        plt.plot(x,energy,'g.', label= 'energy')

	plt.title(r'$\lambda=%d$' % l)
	plt.legend()
    plt.show()
