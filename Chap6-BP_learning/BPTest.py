#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

from numpy import *
import operator
from bpnet import BPNet
import matplotlib.pyplot as plt

bpnet = BPNet()
bpnet.loadDataSet('testSet2.txt')
bpnet.dataMat = bpnet.normalize(bpnet.dataMat)

bpnet.drawClassScatter(plt)
bpnet.bpTrain()
print(bpnet.out_wb)
print(bpnet.hi_wb)

x,z = bpnet.BPClassfier(-3.0,3.0)
bpnet.classfyLine(plt,x,z)
plt.show()
bpnet.TrendLine(plt)
plt.show()