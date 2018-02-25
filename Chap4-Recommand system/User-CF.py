# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

from numpy import *
import numpy as np
from Recommand_lib import kNN

dataMat = mat([[0.238, 0, 0.1905, 0.1905, 0.1905, 0.1905], [0, 0.1777, 0,0.294,0.235, 0.294],
               [0.2,0.16, 0.12, 0.12, 0.2, 0.2]])
testSet = [0.2174, 0.2174, 0.1304, 0, 0.2174, 0.2174]
classLabel = np.array(['B', 'C', 'D'])

print(kNN(testSet, dataMat, classLabel, 3))