# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University


from numpy import *
import numpy as np
from Recommand_lib import *
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

k = 4
dataSet = file2matrix("testdata/4k2_far.txt", "\t")
dataMat = mat(dataSet[:, 1:])

kmeans = KMeans(init='k-means++', n_clusters=4)
kmeans.fit(dataMat)

drawScatter_origin(plt, dataMat, size=20, color='b', mrkr='.')
drawScatter_center(plt, kmeans.cluster_centers_, size=60, color='red', mrkr='D')
plt.show()
