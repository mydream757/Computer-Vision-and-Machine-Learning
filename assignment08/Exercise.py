import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin


temp1 = np.array([[1,2,3],[4,5,6]])
temp2 = np.array([[1,0],[0,1],[1,0]])
t3 = np.dot(temp1,temp2)
print(t3)

t4 = lin.pinv(temp1)
print(t4)
