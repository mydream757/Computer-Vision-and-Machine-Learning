import math
from random import *
import matplotlib
import matplotlib.pyplot as plt

class point:
    x = 0
    y = 0
    label = -1   #initial value is -1, 0 to k-1 is valid
    dist = []
    def initDist(self,k):
        self.dist = [0 for i in range(k)]
    def assignLabel(self):
        self.label = self.dist.index(min(self.dist))
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def printAll(self):
        print(self.x, self.y, self.label)
        print(self.dist)

def generatePointCluster(min,max,k):
    #set seed for random
    #containers of clusters
    points = []
    #generate labels as many as numberOfClusters
    for i in range(k):
        seed()
        #The range of coordinates is preset: x[0:100] y[0:100]
        p = point(randrange(min,max,1),randrange(min,max,1))
        points.append(p)

    return points
def writeRandomPoints(min,max,numberOfPoints,filename):
    f = open(filename,'w')
    x = []
    y = []
    for i in range(numberOfPoints):
        seed()
        tx = str(randrange(min,max,1))
        ty = str(randrange(min,max,1))
        txt = tx+" "+ty+"\n"
        f.writelines(txt)
    f.close()
def readFromText(filename):
    f = open(filename,'r')
    points = []
    while True:
        line = f.readline()
        if not line: break
        x, y = line.split()
        p = point(int(x),int(y))
        points.append(p)

    f.close()
    return points
def computeDistance(pA, pB):
    pow = math.pow((pA.x - pB.x),2) + math.pow((pA.y - pB.y),2)
    distance = math.sqrt(pow)
    return distance
def computeEnergy(points, clusters):
    sum = 0
    for k in range(len(clusters)):
        for i in range(len(points)):
            if points[i].label == k:
                sum = sum + math.pow(computeDistance(points[i],clusters[k]),2)
    return sum/len(points)
def computeCentroid(points, clusters):
    centroids = []
    for k in range(len(clusters)):
        count = 0
        sumX = 0
        sumY = 0
        for i in range(len(points)):
            if points[i].label == k:
                sumX = sumX + points[i].x
                sumY = sumY + points[i].y
                count = count + 1
        if count != 0:
            centroid = point(sumX/count, sumY/count)
        else:
            centroid = clusters[k]
        centroids.append(centroid)
    return centroids
def checkChangeLabel(previous,current):
        for i in range(len(previous)):
            if previous[i]!=current[i]:
                return True
        return False
def makeLabelList(points):
    Labels = []
    for i in range(len(points)):
        Labels.append(points[i].label)
    return Labels
def initializeLabel(points,clusters):
    for i in range(len(points)):
        points[i].initDist(len(clusters))
        for k in range(len(clusters)):
            points[i].dist[k] = computeDistance(clusters[k],points[i])
        points[i].assignLabel()      #tag labels to points
def separatePointsToXY(points,indexOfCluster):
    x = []
    y = []
    for i in range(len(points)):
        if points[i].label == indexOfCluster:
            x.append(points[i].x)
            y.append(points[i].y)
    return [x,y]
def main():
    writeRandomPoints(0,1500,100,"data.txt")       #generate data set and save as .txt
    ######  initial setting  ######
    points = readFromText("data.txt")           #initialize data from .txt file
    Px,Py = separatePointsToXY(points,-1)
    centroids = generatePointCluster(0,1500,8)     #generate initial centroids randomly
    Cx,Cy = separatePointsToXY(centroids,-1)
    fig = plt.figure()
    plt.scatter(Px,Py,color='b',marker='o',s=20,label='input data')
    plt.scatter(Cx,Cy,color='r',marker='o',s=30,label='initial centroids')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Initial setting')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    previous = makeLabelList(points)         #save initial label list
    initializeLabel(points,centroids)        #initialize Label
    current = makeLabelList(points)         #after assign label, make label list

    ######  loop clustering  ######
    count = 0
    while checkChangeLabel(previous,current):
        count = count + 1
        previous = current
        nextCentroids = computeCentroid(points, centroids)
        initializeLabel(points, nextCentroids)
        current = makeLabelList(points)
        centroids = nextCentroids
        energy = computeEnergy(points, nextCentroids)
        print("Energy of iteration %d"%count,"%f"%energy)
        xList = []
        yList = []
        fig = plt.figure()
        for i in range(len(centroids)):
            x,y = separatePointsToXY(points,i)
            xList.append(x)
            yList.append(y)
            color = ['b','g','c','m','y','k','aqua','pink','purple']
            label = "k: "+str(i)+" cluster"
            plt.scatter(xList[i],yList[i],color=color[i],marker='o',s=20,label=label)
        Cx,Cy = separatePointsToXY(centroids,-1)
        plt.scatter(Cx,Cy,color='r',marker='o',s=30,label='changed centroids')
        title = "Iteration: " + str(count)
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.title(title)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
if __name__ == '__main__':
    main()
