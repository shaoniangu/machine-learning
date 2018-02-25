# -*-coding:utf-8-*-
#__Author__=Youzhi Gu
#Learn Python at Zhejiang University


import sys
import os
from sklearn.datasets.base import Bunch
import pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn import metrics


def readfile(path):
    fp = open(path,'rb')
    content = fp.read()
    fp.close()
    return content

def savefile(savepath,content):
    fp = open(savepath,'wb')
    fp.write(content)
    fp.close()

def readbunchobj(path):
    file_obj = open(path,'rb')
    bunch = pickle.load(file_obj, encoding='iso-8859-1')
    file_obj.close()
    return bunch


def writebunchobj(path,bunchobj):
    file_obj = open(path,'wb')
    pickle.dump(bunchobj, file_obj)
    file_obj.close()


def metrics_result(actual,predict):
    print("精度：{0:.3f}".format(metrics.precision_score(actual,predict)))
    print("召回：{0:.3f}".format(metrics.recall_score(actual, predict)))
    print("F1-score：{0:.3f}".format(metrics.f1_score(actual, predict)))


stopword_path = "train_word_bag/hlt_stop_words.txt"
stpwrdlst = readfile(stopword_path).splitlines()

path = 'train_word_bag/train_set.dat'
bunch = readbunchobj(path)

tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames,
                   tdm=[], vocabulary={})

vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)
transformer = TfidfTransformer()

tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
tfidfspace.vocabulary=vectorizer.vocabulary_

space_path = "train_word_bag/tfdifspace.dat"
writebunchobj(space_path,tfidfspace)


#测试集词向量映射到训练集词袋中，生成向量空间模型
path = "test_word_bag/test_set.dat"
bunch = readbunchobj(path)
testspace = Bunch(target_name=bunch.target_name, label=bunch.label,filenames = bunch.filenames,
                  tdm=[],vocabulary={})

trainbunch = readbunchobj("train_word_bag/tfdifspace.dat")
vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,
                             vocabulary=trainbunch.vocabulary)
transformer = TfidfTransformer()
testspace.tdm = vectorizer.fit_transform(bunch.contents)
testspace.vocabulary = trainbunch.vocabulary

space_path = "test_word_bag/testspace.dat"
writebunchobj(space_path,testspace)

trainpath = "train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)

clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)
predicted = clf.predict(test_set.tdm)
total = len(predicted)
rate= 0
for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
    if flabel != expct_cate:
        rate +=1
        print(file_name,"实际类别：", flabel, '\t'+"预测类别：", expct_cate)

print("error_rate:",float(rate)*100/float(total),"%")
metrics_result(test_set.label,predicted)
