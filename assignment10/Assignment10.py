import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
from matplotlib.image import imread

file_data_train = "mnist_train.csv"
file_data_test  = "mnist_test.csv"

h_data_train    = open(file_data_train, "r")
h_data_test     = open(file_data_test, "r")

data_train      = h_data_train.readlines()
data_test       = h_data_test.readlines()

h_data_train.close()
h_data_test.close()

size_row    = 28    # height of the image
size_col    = 28    # width of the image

num_train   = len(data_train)   # number of training images
num_test    = len(data_test)    # number of testing images

# normalize the values of the input data to be [0, 1]
def normalize(data):
    data_normalized = (data - min(data)) / (max(data) - min(data))
    return(data_normalized)

# example of distance function between two vectors x and y
def distance(x, y):

    d = (x - y) ** 2
    s = np.sum(d)
    # r = np.sqrt(s)
    return(s)

# return the feature function vectors upto Max P
def featureFun(maxP):
    #feature function: numpy.random.normal
    f = np.empty((size_col*size_row,maxP),dtype=float)
    for i in range(maxP):
        f[:,i]=np.random.normal(0,1,size_col*size_row)
    #result = np.dot(f.T,x.T)
    return f
#make a matrix each column of which represents an images
list_image_train    = np.empty((num_train, size_row * size_col), dtype=float)
list_label_train    = np.empty(num_train, dtype=int)

list_image_test     = np.empty((num_test, size_row * size_col), dtype=float)
list_label_test     = np.empty(num_test, dtype=int)

#parse the data sets
count = 0
for line in data_train:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label_train[count]    = label
    list_image_train[count,:]  = im_vector

    count += 1

count = 0
for line in data_test:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label_test[count]    = label
    list_image_test[count,:]  = im_vector

    count += 1

im_average  = np.zeros((10, size_col*size_row), dtype=float)
im_count    = np.zeros(10, dtype=int)

for i in range(num_train):
    im_average[list_label_train[i],:] += list_image_train[i,:]
    im_count[list_label_train[i]] += 1

for i in range(10):
    im_average[i,:] /= im_count[i]

#Ready for test
#P can't be over MaxP
maxP = pow(2, 14)
print("MaxP: ",maxP)
#compute feature function of MaxP
f = featureFun(maxP)
#compute coefficient using average image vectors and feature function
bestP = 0
bestScore = 0
for i in range(13):

    #set different parameter, P
    p = pow(2,i+1)
    print(i+1,"iteration P: ",p)


    im_label = np.zeros((10,10), dtype=float)

    for j in range(10):
        fx = np.dot(f[:,:p-1].T,im_average.T)
        inverse = lin.pinv(fx.T)
        for r in range(10):

            #label = j : 1, others : -1 for 0,1....9 digits
            if r==j:
                im_label[r,j] = 1
            else:
                im_label[r,j] = -1

    #classifiers of the digits 0,1,2....9
    coefficient = np.dot(inverse, im_label)

    result = np.dot(np.dot(f[:,:p-1].T, list_image_test.T).T, coefficient)
    #experiment result
    indexOfMax = np.argmax(result,1)

    num = 0

    confusionMatrix = np.zeros((10,10),dtype=int)
    confusionMatrixTable = np.zeros((11,11),dtype=int)
    for b in range(10):
        confusionMatrixTable[0][b+1] = b
        confusionMatrixTable[b+1][0] = b

    for a in range(indexOfMax.size):
        if indexOfMax[a] == list_label_test[a]:
            confusionMatrixTable[indexOfMax[a]+1][indexOfMax[a]+1] += 1
            confusionMatrix[indexOfMax[a]][indexOfMax[a]] += 1
        elif indexOfMax[a] != list_label_test[a]:
            confusionMatrixTable[list_label_test[a]+1][indexOfMax[a]+1] += 1
            confusionMatrix[list_label_test[a]][indexOfMax[a]] += 1
    print(confusionMatrixTable)

    precision = np.zeros(10,dtype=float)
    recall = np.zeros(10, dtype=float)

    for c in range(10):
        if np.sum(confusionMatrix, axis=1)[c]!=0:
            precision[c] = confusionMatrix[c][c]/np.sum(confusionMatrix, axis=1)[c]
        if np.sum(confusionMatrix, axis=0)[c]!=0:
            recall[c] = confusionMatrix[c][c]/np.sum(confusionMatrix, axis=0)[c]
    precision = np.sum(precision)
    recall = np.sum(recall)
    F1score = 2 * (precision * recall)/(precision + recall)
    if bestScore<F1score:
        bestP = p
        bestScore = F1score
    print("F1 score :", F1score)

print("Best P: ",bestP)
print("Best score: ",bestScore)
