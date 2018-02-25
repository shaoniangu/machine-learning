# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

import sys
import os
from numpy import *
import numpy as np
from NBayes import *

dataSet,listclasses = loadDataSet()
nb = NBayes()
nb.train_set(dataSet,listclasses)
nb.map2vocab(dataSet[5])
print(nb.testset)
print(nb.predict(nb.testset))