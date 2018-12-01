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

im_label = np.zeros((10,1), dtype=float)
for i in range(10):
    im_average[i,:] /= im_count[i]
    #label = 0 : 1, others : -1
    if i==0:
        im_label[0,0] = 1
    else:
        im_label[i,0] = -1

#Ready for test
experiment_label_test = np.empty(num_test, dtype=int)
experiment_average_test = np.zeros((4,size_row * size_col), dtype=float)
experiment_count_test = np.zeros(4, dtype=int)
result_average_test = np.zeros((4,size_row * size_col), dtype=float)
result_count_test = np.zeros(4, dtype=int)
Final = 0
FinalP = 0
#P can't be over MaxP
maxP = pow(2, 14)
print("MaxP: ",maxP)
#compute feature function of MaxP
f = featureFun(maxP)
#compute coefficient using average image vectors and feature function
for i in range(13):
    score = 0
    #initialize containers
    experiment_label_test = np.empty(num_test, dtype=int)
    experiment_average_test = np.zeros((4,size_row * size_col), dtype=float)
    experiment_count_test = np.zeros(4, dtype=int)

    #set different parameter, P
    p = pow(2,i)
    print(i,"iteration P: ",p)
    fx = np.dot(f[:,:p-1].T,im_average.T)
    inverse = lin.pinv(fx.T)
    coefficient = np.dot(inverse, im_label)

    #Evaluate my classifier using MNIST test set
    result = np.dot(np.dot(f[:,:p-1].T, list_image_test.T).T, coefficient)
    num = 0
    for k in range(result.size):
        #sign(f(x))
        if result[k] >= 0:
            experiment_label_test[k] = 1
        else:
            experiment_label_test[k] = -1
        #check TP, FP, TN, FN
        if experiment_label_test[k] == 1 and list_label_test[k] == 0:
            num = 0 #then TP
        elif experiment_label_test[k] == 1 and list_label_test[k] !=0:
            num = 1 #then FP
        elif experiment_label_test[k] == -1 and list_label_test[k] == 0:
            num = 2 #then FN
        elif experiment_label_test[k] == -1 and list_label_test[k] !=0:
            num = 3 #then TN
        experiment_average_test[num,:] += list_image_test[k,:]
        experiment_count_test[num] += 1

    for j in range(4):
        if experiment_count_test[j]!=0:
            experiment_average_test[j, :] /= experiment_count_test[j]

    precision = experiment_count_test[0]*100/(experiment_count_test[0]+experiment_count_test[1])
    recall = experiment_count_test[0]*100/(experiment_count_test[0]+experiment_count_test[2])
    #get the score
    score = (2*precision*recall)/(precision+recall)
    print(i,"iteration score: ",score)
    if score > Final:
        FinalP = p
        Final = score
        result_average_test = experiment_average_test
        result_count_test = experiment_count_test

    print(i,"iteration best score: ",Final)

#varing p with standard deviation = 1
k=0
incremental = 1
p = FinalP
stopChecker = 0
for r in range(100):
    print(r," iteration")
    #set different parameter, P
    p = p + incremental
    score = 0
    #initialize containers
    experiment_label_test = np.empty(num_test, dtype=int)
    experiment_average_test = np.zeros((4,size_row * size_col), dtype=float)
    experiment_count_test = np.zeros(4, dtype=int)

    print("P: ",p)
    fx = np.dot(f[:,:p-1].T,im_average.T)
    inverse = lin.pinv(fx.T)
    coefficient = np.dot(inverse, im_label)

    #Evaluate my classifier using MNIST test set
    result = np.dot(np.dot(f[:,:p-1].T, list_image_test.T).T, coefficient)
    num = 0
    for k in range(result.size):
        #sign(f(x))
        if result[k] >= 0:
            experiment_label_test[k] = 1
        else:
            experiment_label_test[k] = -1
        #check TP, FP, TN, FN
        if experiment_label_test[k] == 1 and list_label_test[k] == 0:
            num = 0 #then TP
        elif experiment_label_test[k] == 1 and list_label_test[k] !=0:
            num = 1 #then FP
        elif experiment_label_test[k] == -1 and list_label_test[k] == 0:
            num = 2 #then FN
        elif experiment_label_test[k] == -1 and list_label_test[k] !=0:
            num = 3 #then TN
        experiment_average_test[num,:] += list_image_test[k,:]
        experiment_count_test[num] += 1

    for j in range(4):
        experiment_average_test[j, :] /= experiment_count_test[j]
    precision = experiment_count_test[0]*100/(experiment_count_test[0]+experiment_count_test[1])
    recall = experiment_count_test[0]*100/(experiment_count_test[0]+experiment_count_test[2])
    #get the score
    score = (2*precision*recall)/(precision+recall)
    print("score: ",score)
    if Final == score:
        stopChecker += 1

    elif score > Final:
        FinalP = p
        Final = score
        result_average_test = experiment_average_test
        result_count_test = experiment_count_test
    #change incremental direction
    elif Final>score:
        incremental *= -1
    print("best score: ",Final)

    #loop stop condition
    if stopChecker == 3:
        break

print("P of the best score: ",FinalP)
print("The best score(F1): ",Final)

#plot the TP,FP,TN,FN
plt.figure()
for i in range(4):
    result_average_test[i, :] /= result_count_test[i]
    title = ['TP','FP','FN','TN']
    plt.subplot(1, 4, i+1)
    plt.title(title[i])
    plt.imshow(result_average_test[i,:].reshape((size_row, size_col)), cmap='Greys', interpolation='None')

    frame   = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    print(title[i],': ',result_count_test[i])
plt.show()
