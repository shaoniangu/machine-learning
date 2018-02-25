# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

from numpy import *
import numpy as np

a=b'123\t456'
print(a.decode().split('\t'))

A = np.loadtxt(open("testdata/4k2_far.txt"),delimiter= "\t",skiprows=0)
print(A[:,0])
