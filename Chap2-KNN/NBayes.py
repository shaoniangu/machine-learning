# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University


import numpy as np


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him', 'my'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


class NBayes(object):
    def __init__(self):
        self.vocabulary = []
        self.idf = 0
        self.td = 0
        self.tdm = 0
        self.Pcates = {}
        self.labels = []
        self.doclength = 0
        self.vocablen = 0
        self.testset = 0

    def cate_prob(self,classVec):
        self.labels = classVec
        labeltemps = set(self.labels)
        for labeltemp in labeltemps:
            self.labels.count(labeltemp)
            self.Pcates[labeltemp]=float(self.labels.count(labeltemp))/float(len(self.labels))

    def calc_wordfreq(self,trainset):
        self.idf = np.zeros([1,self.vocablen])
        self.tf = np.zeros([self.doclength,self.vocablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx,self.vocabulary.index(word)] += 1
            for signleword in set (trainset[indx]):
                self.idf[0,self.vocabulary.index(signleword)] += 1

    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates), self.vocablen])
        sumlist = np.zeros([len(self.Pcates), 1])
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx]
            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
        self.tdm = self.tdm / sumlist

    def train_set(self,trainset,classVec):
        self.cate_prob(classVec)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        #self.calc_tfidf(trainset)
        self.calc_wordfreq(trainset)
        self.build_tdm()

    def map2vocab(self,testdata):
        self.testset = np.zeros([1,self.vocablen])
        for word in testdata:
            self.testset[0, self.vocabulary.index(word)] += 1

    def predict(self,testset):
        if np.shape(testset)[1] != self.vocablen:
            exit(0)

        predvalue = 0
        predclass = ""
        for tdm_vect,keyclass in zip(self.tdm,self.Pcates):
            temp = np.sum(testset * tdm_vect * self.Pcates[keyclass])
            if temp > predvalue:
                predvalue = temp
                predclass = keyclass
        return predclass

    def calc_tfidf(self,trainset):
        self.idf = np.zeros([1,self.vocablen])
        self.tf = np.zeros([self.doclength,self.vocablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx,self.vocabulary.index(word)] += 1
            self.tf[indx] = self.tf[indx]/float(len(trainset[indx]))
            for signleword in set(trainset[indx]):
                self.idf[0,self.vocabulary.index(signleword)] += 1
        self.idf = np.log(float(self.doclength)/self.idf)
        self.tf = np.multiply(self.tf,self.idf)