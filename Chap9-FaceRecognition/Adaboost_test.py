# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

from numpy import *
import sys
from Adaboost_lib import *
import matplotlib.pyplot as plt
import numpy as np

# 导入训练集
dataArr,labelArr = loadDataSet('horseColicTraining.txt')
# 训练分类器
weakClassArr,aggClassEst = adaBoostTrain(dataArr,labelArr,numIt=10)
print("weakClassArr:",weakClassArr)
# print "aggClassEst:",aggClassEst
# 绘制ROC曲线
plotROC(aggClassEst.T, labelArr)

# 导入测试集
testArr,testLabelArr = loadDataSet('horseColicTest.txt')
ClassEst100 = adaClassify(testArr,weakClassArr)# 用学习好的分类器进行分类
errArr = mat(ones((67,1)))
totalError = errArr[ClassEst100!=mat(testLabelArr).T].sum()
print("totalError:",totalError)