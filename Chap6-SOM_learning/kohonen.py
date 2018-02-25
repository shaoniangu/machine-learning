#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

from numpy import *

class Kohonen(object):
    def __init__(self):
        self.lratemax = 0.8
        self.lratemin = 0.05
        self.rmax = 5.0
        self.rmin = 0.5
        self.Steps = 1000
        self.lratelist = []
        self.rlist = []
        self.w = []
        self.M = 2
        self.N = 2
        self.dataMat = []
        self.classLabel = []

    # 归一化数据
    def normalize(self,dataMat):
        [m,n] = shape(dataMat)
        for i in range(n-1):
            dataMat[:,i] = (dataMat[:,i] - mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
        return dataMat

    def distEclud(self,matA,matB):
        ma, na = shape(matA);
        mb, nb = shape(matB);
        rtnmat = zeros((ma, nb))
        for i in range(ma):
            for j in range(nb):
                rtnmat[i, j] = linalg.norm(matA[i, :] - matB[:, j].T)
        return rtnmat

    def loadDataSet(self,fileName):
        numFeat = len(open(fileName).readline().split('\t')) - 1
        fr = open(fileName)
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            lineArr.append(float(curLine[0]))
            lineArr.append(float(curLine[1]))
            self.dataMat.append(lineArr)
        self.dataMat = mat(self.dataMat)

    def init_grid(self):
        k = 0
        grid = mat(zeros((self.M * self.N, 2)))
        for i in range(self.M):
            for j in range(self.N):
                grid[k,:] = [i,j]
                k += 1
        return grid

    def ratecalc(self,indx):
        lrate = self.lratemax - (float(indx) + 1.0) / float(self.Steps) * (self.lratemax - self.lratemin)
        r = self.rmax - (float(indx) + 1.0) / float(self.Steps) * (self.rmax - self.rmin)
        return lrate, r
        #lrate = self.lratemax-((i+1.0)*(self.lratemax-self.lratemin)/self.Steps)
        #r = self.rmax-((i+1.0)*(self.rmax-self.rmin)/self.Steps)
        #return lrate,r

    def train(self):
        dm,dn = shape(self.dataMat)
        normDataset = self.normalize(self.dataMat)
        grid = self.init_grid()
        self.w = random.rand(dn,self.M*self.N)
        distM = self.distEclud

        if self.Steps < 5*dm:
            self.Steps = 5*dm

        for i in range(self.Steps):
            lrate,r = self.ratecalc(i)
            self.lratelist.append(lrate)
            self.rlist.append(r)

            k = random.randint(0,dm)
            mySample = normDataset[k,:]

            minIndx = (distM(mySample,self.w)).argmin()

            d1 = ceil(minIndx/self.M)
            d2 = mod(minIndx, self.M)
            distMat = distM(mat([d1, d2]), grid.T)
            nodelindx = (distMat < r).nonzero()[1]
            for j in range(shape(self.w)[1]):
                if sum(nodelindx == j):
                    self.w[:, j] = self.w[:,j]+lrate*(mySample[0]-self.w[:,j])

        self.classLabel = list(range(dm))
        for i in range(dm):
            self.classLabel[i] = distM(normDataset[i,:],self.w).argmin()
        self.classLabel = mat(self.classLabel)

    def showCluster(self,plt):
        lst = unique(self.classLabel.tolist()[0])
        i=0
        for cindx in lst:
            myclass = nonzero(self.classLabel==cindx)[1]
            xx = self.dataMat[myclass].copy()
            if i == 0:
                plt.plot(xx[:,0],xx[:,1],'bo')
            if i == 1:
                plt.plot(xx[:,0],xx[:,1],'rd')
            if i == 2:
                plt.plot(xx[:,0],xx[:,1],'gD')
            if i == 3:
                plt.plot(xx[:,0],xx[:,1],'c^')
            i += 1
        plt.show()