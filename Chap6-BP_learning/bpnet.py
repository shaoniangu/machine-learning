#__Author__=Youzhi Gu
#Learn Python at Zhejiang University

#__Author__=Youzhi Gu
#Learn Python at Zhejiang University


from numpy import *

#构造BP类
class BPNet(object):
    # 构造函数
    def __init__(self):
        self.eb = 0.01 #误差容限，当误差小于这个值时，算法收敛
        self.iterator = 0 #算法收敛时的迭代次数
        self.eta = 0.01 #学习率
        self.mc = 0.3 #动量因子
        self.maxiter = 2000
        self.nHidden = 4 #隐含层神经元个数
        self.nOut = 1 #输出层神经元个数

        self.errlist = [] #误差列表
        self.dataMat = 0 #训练集
        self.classLabels = 0 #分类标签集
        self.nSampNum = 0 #样本集行数
        self.nSampDim = 0 #样本列数

    # 隐含层初始化
    def init_hiddenWB(self):
        self.hi_w = 2.0*(random.rand(self.nHidden,self.nSampDim)-0.5)
        self.hi_b = 2.0*(random.rand(self.nHidden,1)-0.5)
        self.hi_wb = mat(self.addcol(mat(self.hi_w),mat(self.hi_b)))

    # 输出层初始化
    def init_OutputWB(self):
        self.out_w = 2.0*(random.rand(self.nOut,self.nHidden)-0.5)
        self.out_b = 2.0*(random.rand(self.nOut,1)-0.5)
        self.out_wb = mat(self.addcol(mat(self.out_w),mat(self.out_b)))

    # 加载数据集
    def loadDataSet(self,filename):
        self.dataMat = []
        self.classLabels = []
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split()
            self.dataMat.append([float(lineArr[0]),float(lineArr[1]),1.0])
            self.classLabels.append(int(lineArr[2]))
        self.dataMat = mat(self.dataMat)
        [m,n] = shape(self.dataMat)
        self.nSampNum = m #样本数量
        self.nSampDim = n-1 #样本维度

    # 归一化数据
    def normalize(self,dataMat):
        [m,n] = shape(dataMat)
        for i in range(n-1):
            dataMat[:,i] = (dataMat[:,i] - mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
        return dataMat

    # 增加新列
    def addcol(self,matrix1,matrix2):
        [m1,n1] = shape(matrix1)
        [m2,n2] = shape(matrix2)
        if m1 != m2:
            print('different rows,can not merge matrix')
            return
        mergMat = zeros((m1,n1+n2))
        mergMat[:,0:n1] = matrix1[:,0:n1]
        mergMat[:,n1:(n1+n2)] = matrix2[:,0:n2]
        return mergMat

    #绘制分类点
    def drawClassScatter(self,plt):
        i = 0
        for mydata in self.dataMat:
            if self.classLabels[i] == 0:
                plt.scatter(mydata[0,0],mydata[0,1],c='blue',marker='o')
            else:
                plt.scatter(mydata[0,0],mydata[0,1],c='red',marker='s')
            i += 1

    #BP网络主程序
    def bpTrain(self):
        SampIn = self.dataMat.T
        expected = mat(self.classLabels)
        self.init_hiddenWB()
        self.init_OutputWB()
        dout_wbOld = 0.0
        dhi_wbOld = 0.0

        #主循环
        for i in range(self.maxiter):
            hi_input = self.hi_wb * SampIn
            hi_output = self.logistic(hi_input)
            hi2out = self.addcol(hi_output.T, ones((self.nSampNum,1))).T
            out_input = self.out_wb * hi2out
            out_output = self.logistic(out_input)

            err = expected - out_output
            sse = self.errorfunc(err)
            self.errlist.append(sse)
            if sse <= self.eb:
                self.iterator = i+1
                break
            DELTA = multiply(err,self.dlogit(out_output))
            delta = multiply(self.out_wb[:,:-1].T * DELTA,self.dlogit(hi_output))
            dout_wb = DELTA * hi2out.T
            dhi_wb = delta * SampIn.T

            if i == 0:
                self.out_wb = self.out_wb + self.eta * dout_wb
                self.hi_wb = self.hi_wb + self.eta * dhi_wb
            else:
                self.out_wb = self.out_wb + (1.0 - self.mc) * self.eta * dout_wb + self.mc * dout_wbOld
                self.hi_wb = self.hi_wb + (1.0 - self.mc ) * self.eta * dhi_wb + self.mc * dhi_wbOld

            dout_wbOld = dout_wb
            dhi_wbOld = dhi_wb

    #BP网络分类器
    def BPClassfier(self,start,end,steps=30):
        x = linspace(start,end,steps)
        xx = mat(ones((steps,steps)))
        xx[:,0:steps] = x
        yy = xx.T
        z = ones((len(xx),len(yy)))
        for i in range(len(xx)):
            for j in range(len(yy)):
                xi=[]
                tauex=[]
                tartemp=[]
                mat(xi.append([xx[i,j],yy[i,j],1]))
                hi_input = self.hi_wb * (mat(xi).T)
                hi_out = self.logistic(hi_input)
                [taumrow,taumcol] = shape(hi_out)
                tauex = mat(ones((1,taumrow+1)))
                tauex[:,0:taumrow] = (hi_out.T)[:,0:taumrow]
                out_input = self.out_wb * (mat(tauex).T)
                out = self.logistic(out_input)
                z[i,j] = out

        return x,z

    #绘制分类结果
    def classfyLine(self,plt,x,z):
        plt.contour(x,x,z,1,colors='black')

    #绘制误差曲线
    def TrendLine(self,plt,color='r'):
        X = linspace(0,self.maxiter,self.maxiter)
        Y = log2(self.errlist)
        plt.plot(X,Y,color)

    #激活函数：logistic
    def logistic(self,net):
        return 1.0/(1.0+exp(-net))

    def errorfunc(self,inX):
        return sum(power(inX,2)) * 0.5

    def dlogit(self,net):
        return multiply(net,(1.0-net))