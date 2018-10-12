import matplotlib.pyplot as plt
import random
import numpy as np


def normalize(data):
    normalized = (data - min(data)) / (max(data)-min(data))

    return normalized

def distance(x,y):
    d = (x - y)**2
    s = np.sum(d)

    return s

def chooseCentroid(list_image,list_label,numOfk):
    list_centroid =np.empty((size_row*size_col, numOfk), dtype=float)
    list_centroidLabel = np.empty(numOfk,dtype=int)

    rand_num = random.randint(0,len(list_label)-1)
    for i in range(numOfk):
        random.seed()
        while list_label[rand_num] in list_centroidLabel:
            rand_num = random.randint(0,len(list_label)-1)
        list_centroidLabel[i] = list_label[rand_num]
        list_centroid[:,i] = list_image[:,rand_num]

    return [list_centroid, list_centroidLabel]

#when k=10, compute accuracy
def computeAccuracy(list_label,list_clabel):
    num = len(list_label)
    count=0
    for i in range(num):
        if list_label[i]==list_clabel[i]:
            count +=1

    accuracy = (count/num)*100
    return accuracy
def computeEnergy(list_image, list_centroid,list_clabel,list_centroidLabel):
    energy = 0
    for i in range(len(list_clabel)):
        for k in range(len(list_centroidLabel)):
            if list_clabel[i] == list_centroidLabel[k]:
                energy += distance(list_image[:,i],list_centroid[:,k])
    return energy

def assignLabel(list_image,list_clabel,list_centroid,list_centroidLabel):
    for i in range(len(list_clabel)):
        result = distance(list_image[:,i],list_centroid[:,0])
        list_clabel[i] = list_centroidLabel[0]
        for k in range(len(list_centroidLabel)):
            tmp = distance(list_image[:,i],list_centroid[:,k])
            if result > tmp:
                result = tmp
                list_clabel[i] = list_centroidLabel[k]
def makeLabelList(list_clabel):
    list = np.empty(len(list_clabel),dtype=int)
    for i in range(len(list_clabel)):
        list[i] = list_clabel[i]
    return list

def computeCentroid(list_image,list_clabel,list_centroidLabel):
    im_average  = np.zeros((size_row * size_col, len(list_clabel)), dtype=float)
    im_count    = np.zeros(len(list_clabel), dtype=int)

    for i in range(len(list_clabel)):
        for k in range(len(list_centroidLabel)):
            if list_clabel[i] == list_centroidLabel[k]:
                im_average[:, k] += list_image[:, i]
                im_count[k] += 1

    for i in range(len(list_centroidLabel)):
        im_average[:, i] /= im_count[i]

    return im_average

def conditionCheck(before_label,after_label):
    for i in range(len(before_label)):
        if before_label[i] != after_label[i]:
            return True
    return False

k=2  # start = 2 end = 10
while k<11:
    count = 0
    size_row = 28
    size_col = 28
    #get data from .csv file
    file_data = "mnist_test.csv"
    handle_file = open(file_data, "r")
    data = handle_file.readlines()
    num_image = len(data)
    handle_file.close()
    list_image  = np.empty((size_row * size_col, num_image), dtype=float)
    list_label  = np.empty(num_image, dtype=int)
    # split data into list
    for line in data:
        line_data = line.split(',')
        label = line_data[0]
        im_vector = np.asfarray(line_data[1:])
        im_vector = normalize(im_vector)

        list_label[count] = label
        list_image[:,count] = im_vector
        count += 1


    list_clabel = np.empty(num_image, dtype=int)    #cluster label따로
    list_centroid,list_centroidLabel = chooseCentroid(list_image,list_label,k)  #k만큼 centroids 생성
    f1 = plt.figure()

    for i in range(k):

        label       = list_centroidLabel[i]
        im_vector   = list_centroid[:, i]
        im_matrix   = im_vector.reshape((size_row, size_col))   #reshape로 형태 치환

        plt.subplot(1, 10, i+1)        #(x,y,k) = x * y에서 k번째
        plt.title(label)
        plt.imshow(im_matrix, cmap='Greys', interpolation='None')   #이미지 출력하는 함수

        frame   = plt.gca()             #현재의 axes 를 얻어냄
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.show()
    before = makeLabelList(list_clabel)
    print(before)
    assignLabel(list_image,list_clabel,list_centroid,list_centroidLabel)
    after = makeLabelList(list_clabel)
    print(after)

    iterCount = 0
    while conditionCheck(before,after)==True :

        before = after
        print(before)
        list_centroid = computeCentroid(list_image,list_clabel,list_centroidLabel)
        assignLabel(list_image,list_clabel,list_centroid,list_centroidLabel)
        after = makeLabelList(list_clabel)
        print(after)

        energy = computeEnergy(list_image, list_centroid,list_clabel,list_centroidLabel)
        iterCount += 1
        print("Iter: ",iterCount," & Energy: ",energy)

    im_average  = np.zeros((size_row * size_col, k), dtype=float)
    im_count    = np.zeros(k, dtype=int)

    f2 = plt.figure()
    for i in range(len(list_centroidLabel)):

        plt.subplot(1, 10, i+1)
        plt.title(list_centroidLabel[i])
        plt.imshow(list_centroid[:,i].reshape((size_row, size_col)), cmap='Greys', interpolation='None')

        frame   = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.show()
    k += 1
#
#
#for i in range(num_image):
