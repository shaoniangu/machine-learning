#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

from numpy import *
from kohonen import Kohonen
import matplotlib.pyplot as plt

SOMNet = Kohonen()
SOMNet.loadDataSet('dataset2.txt')
SOMNet.train()
print(SOMNet.w)
SOMNet.showCluster(plt)