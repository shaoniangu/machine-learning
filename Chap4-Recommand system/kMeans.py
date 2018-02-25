# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University


from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from Recommand_lib import *

dataMat = file2matrix('testdata/4k2_far.txt','\t')
dataSet = mat(dataMat[:, 1:])
print(dataSet)

k=4
k = 4  # 外部指定1,2,3... 通过观察数据集有4个聚类中心
m = shape(dataSet)[0]  # 返回矩阵的行数

# 本算法核心数据结构:行数与数据集相同
# 列1：数据集对应的聚类中心,列2:数据集行向量到聚类中心的距离
ClustDist = mat(zeros((m, 2)))

# 随机生成一个数据集的聚类中心:本例为2*4的矩阵
# 确保该聚类中心位于min(dataMat[:,j]),max(dataMat[:,j])之间
clustercents = randCenters(dataSet, k)

flag = True  # 初始化标志位,迭代开始
counter = [];  # 计数器

# 循环迭代直至终止条件为False
# 算法停止的条件：dataSet的所有向量都能找到某个聚类中心,到此中心的距离均小于其他k-1个中心的距离
while flag:
    flag = False  # 预置标志位为False

    # ---- 1. 构建ClustDist： 遍历DataSet数据集,计算DataSet每行与聚类的最小欧式距离 ----#
    #     将此结果赋值ClustDist=[minIndex,minDist]
    for i in range(m):

        # 遍历k个聚类中心,获取最短距离
        distlist = [distEclud(clustercents[j, :], dataSet[i, :]) for j in range(k)]
        minDist = min(distlist)
        minIndex = distlist.index(minDist)

        if ClustDist[i, 0] != minIndex:  # 找到了一个新聚类中心
            flag = True  # 重置标志位为True，继续迭代

        # 将minIndex和minDist**2赋予ClustDist第i行
        # 含义是数据集i行对应的聚类中心为minIndex,最短距离为minDist
        ClustDist[i, :] = minIndex, minDist

        # ---- 2.如果执行到此处，说明还有需要更新clustercents值: 循环变量为cent(0~k-1)----#
    # 1.用聚类中心cent切分为ClustDist，返回dataSet的行索引
    # 并以此从dataSet中提取对应的行向量构成新的ptsInClust
    # 计算分隔后ptsInClust各列的均值，以此更新聚类中心clustercents的各项值
    for cent in range(k):
        # 从ClustDist的第一列中筛选出等于cent值的行下标
        dInx = nonzero(ClustDist[:, 0].A == cent)[0]
        # 从dataSet中提取行下标==dInx构成一个新数据集
        ptsInClust = dataSet[dInx]
        # 计算ptsInClust各列的均值: mean(ptsInClust, axis=0):axis=0 按列计算
        clustercents[cent, :] = mean(ptsInClust, axis=0)

print("Clustercents:",clustercents)

color_cluster(ClustDist[:,0:1],dataSet,plt)
drawScatter_center(plt,clustercents,size=60,color='red',mrkr='D')
plt.show()