# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

import sys
import os
from numpy import *
import numpy as np
import operator
from NBayes import *

k=3

def cosdist(vector1,vector2):
    return dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))

def classify(testdata,trainSet,listClasses,k):
    dataSetSize = trainSet.shape[0]
    distances = array(zeros(dataSetSize))
    for indx in range(dataSetSize):
        distances[indx] = cosdist(testdata,trainSet[indx])
    sortedDistIndicies = argsort(-distances)
    classCount = {}
    for i in range(k):
        voteIlabel = listClasses[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

dataSet,listClasses = loadDataSet()
nb = NBayes()
nb.train_set(dataSet,listClasses)
print(classify(nb.tf[1],nb.tf,listClasses,k))