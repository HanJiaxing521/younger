from numpy import *
import csv  
import copy
import operator
import types
from math import log
from collections import Counter
import matplotlib.pyplot as plt
decisionNode = dict(boxstyle="sawtooth" , fc="0.8") 
leafNode = dict(boxstyle="round4" , fc="0.8")
arrow_args = dict(arrowstyle="<-")



#聚类得到簇心
##########################################################

def getData(): #数据预处理
    with open('E:\\hackthon\\test.csv', 'r', encoding='utf-8') as f:
        dataMat = []
        reader = csv.reader(f)
        m = 50
        for row in reader:
            dataMat.append(row)
            n = shape(row)[0] #一位数组元素个数，即属性个数
        for i in range(0 , m):
            for j in range(8):
                dataMat[i][j] = float(dataMat[i][j]) #数据存成float类型
    return dataMat #返回float类型的数据
        
def distEclud(vecA , vecB): #计算点A和点B之间的欧氏距离
    return sqrt(sum(power(vecA - vecB , 2)))

def randCent(dataSet , k): #生成k个随机质心
    n = shape(dataSet)[1]#数据个数
    centroids = mat(zeros((k , n)))#随机质心数组
    for j in range(n):
        minJ = min(dataSet[:][j])

        rangeJ = float(max(dataSet[:][j]) - minJ)#极差
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)#在最大值和最小值之间生成随机质心
    return centroids

def kMeans(dataSet , k , distMeas = distEclud , creatCent = randCent): #KMeans分类函数
    m = shape(dataSet)[0]#数据个数
    clusterAssment = mat(zeros((m , 2)))#误差值矩阵
    centroids = creatCent(dataSet , k)#簇心向量矩阵，存放k个簇心向量k行n列（每列（每个属性值）是一个选择题的值）
    clusterChanged = True #是否改变标志初始化为0
    while clusterChanged: #只要有改变就循环
        clusterChanged = False
        for i in range(m): #对于第i个数据
            minDist = inf #距离最小值初始化为最大值
            minIndex = -1 #最小值簇心序号初始化为-1
            for j in range(k): #对于每个点寻找距离最近的簇心
                distJI = distMeas(centroids[j][:],dataSet[i][:])
                if distJI < minDist: #若找到，更改簇心点和最小误差值
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids) #输出中心向量矩阵
        for cent in range(k): #用新的到的簇更新质心
            ptsInClust = [dataSet[i] for i in nonzero(clusterAssment[:,0] == cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids , clusterAssment #返回簇心向量矩阵和误差矩阵

def biKmeans(dataSet , k , distMeas = distEclud):
    m = shape(dataSet)[0] #数据个数
    clusterAssment = mat(zeros((m , 2))) #[簇的编号，与簇心距]点簇分配结果及平方误差(该点与所属簇的距离，即对于这个点当前的最小距离值)矩阵
    centroid0 = mean(dataSet , axis = 0).tolist() #初始为一个簇，计算簇心，mean()返回一个一位列均值数组
    centList = [centroid0] #簇心列表
    for j in range(m): 
        clusterAssment[j,1] = distMeas(mat(centroid0) , dataSet[j])**2
    #print(clusterAssment)
    while (len(centList) < k): #当簇的分类小于k时，对于每一个簇，尝试进行二分类，找到使SSE最小值的簇进行二分
        lowestSSE = inf #SSE初始值为最大值
        for i in range(len(centList)): #对于每个簇类
            #print(type(nonzero(clusterAssment[:,0] == i)[0]))
            ptsInCurrCluster=[dataSet[i] for i in nonzero(clusterAssment[:,0] == i)[0]]
            centroidMat , splitClustAss = kMeans(ptsInCurrCluster , 2 , distMeas) #对这个簇尝试二分类
            sseSplit = sum(splitClustAss[:,1]) #本次被分类的误差值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0] != i)[0] , 1]) #本次未被分类的误差值
        #print("sseSplit , and notSplit: ",sseSplit , sseNotSplit)
        if (sseSplit + sseNotSplit) < lowestSSE: #若二分类后SSE值减少
            bestCentToSplit = i #记录最优簇心
            bestNewCents = centroidMat
            bestClustAss = splitClustAss.copy() #记录最优情况下每个点的误差值（列表）
            lowestSSE = sseSplit + sseNotSplit #记录最小SSE值
        bestClustAss[nonzero(bestClustAss[:,0] == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0] == 0)[0],0] = bestCentToSplit #更新簇的分类结果
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('ths len of bestClustAss is ',len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:] #把结果添加到簇心列表上
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0] == bestCentToSplit)[0],:] = bestClustAss
    print("*******************************")
    for i in centList:
        print("簇心")
        print(i)
    return centList , clusterAssment #返回簇心矩阵和误差数据

def labelling(dataSet , clusterAssment): #给数据贴标签
    lable = clusterAssment[:,0]
    #print(lable)
    lable.tolist
    for i in range(50):
        dataSet[i].append(str(lable[i]))
    #print(dataSet)
    return dataSet
    


#构建决策树
#########################################################################

def Init_Data_Set(DataSet , n):#初始化数据,获得训练集和测试集  
    labels = []  #获得属性集合
    m = 50
    for i in range(0 , m):
        for j in range(n):
            DataSet[i][j] = float(DataSet[i][j])
    for k in range(0 , n+1):
        labels.append(k)
    train_num = int(m * 0.7)
    test_num = m - train_num
    train_dataMat = []
    train_labelMat = []
    test_dataMat = []
    test_labelMat = []
    for i in range(0 , train_num+1):  #得到训练数据及其类别
        train_dataMat.append(Data[i][0:n])
        train_dataMat[i].append(Data[i][-1])
        train_labelMat.append(Data[i][-1])
    for j in range(train_num+1 , m):  #得到测试数据及其类别
        test_dataMat.append(Data[j][0:n])
        test_labelMat.append(Data[j][-1])
    return train_dataMat , train_labelMat , train_num , test_dataMat , test_labelMat , test_num , labels    #返回训练数据集合测试数据集的数据数组和类型数组


def calcShannonEnt(dataSet):
    """计算数据集的熵"""
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt
def splitDataSet(dataSet, axis, value):
    """按照给定特征划分数据集"""
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """选择最好的数据集划分方式"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """返回出现次数最多的特征"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """创建树的节点"""
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createTree(dataSet, labels):
    """创建树的代码"""
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def getNumLeafs(myTree):
    """获取叶节点的数目"""
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试节点的数据类型是否为字典
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """获得树的深度"""
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    """在父子节点间填充文本信息"""
    xMid = (parentPt[0] - cntrPt[0] / 2.0 + cntrPt[0])
    yMid = (parentPt[1] - cntrPt[1] / 2.0 + cntrPt[1])
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    """绘制树"""
    numLeafs = getNumLeafs(myTree)

    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # 减少y偏移
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,
                                       plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def static(dataSet):
    """统计"""

    class0 = []
    class1 = []
    class2 = []
    for line in dataSet:
        if (line[-1] == 0.0):
            class0.append(line[:-1])
        elif (line[-1] == 1.0):
            class1.append(line[:-1])
        elif (line[-1] == 2.0):
            class2.append(line[:-1])
        else:
            pass
    print(class0)
    print(class1)
    print(class2)
    cols = len(line[:-1])
    result1 = [Counter(class0[:][i]).most_common(1)[0] for i in range(cols)]
    result2 = [Counter(class0[:][i]).most_common(1)[0] for i in range(cols)]
    result3 = [Counter(class0[:][i]).most_common(1)[0] for i in range(cols)]
    print(result1)
    print(result2)
    print(result3)


def createPlot(inTree):
    """绘制图形"""
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[0, 0.3, 0.6, 0.9, 1.2, 1.5],
                   yticks=[0, 0.3, 0.6, 0.9, 1.2, 1.5])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

def classify(inputTree, featLabels, testVec):
    """使用决策树进行分类"""
    firstStr = list(inputTree.keys())[0]

    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = 10
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

Data = getData()
CentList , ClusterAssment = biKmeans(Data , 3 , distEclud)
newData = labelling(Data , ClusterAssment)
LN = [3 , 6 , 8]
for i in range(3):
    train_data , train_label , train_num , test_data , test_label , test_num , labels = Init_Data_Set(newData , LN[i]) #获得训练集和测试集
    if LN[i] == 3:
        trainLabel = [1,2,3]
    elif LN[i] == 6:
        trainLabel = [1, 2, 3, 4, 5, 6,]
    else:
        trainLabel = [1,2,3,4,5,6,7,8]
    for j in train_data:
        last = float(j[-1][2:3])
        j.pop()
        j.append(last)
    static(train_data)
    print("===========================================") 
    my_Tree = createTree(train_data, trainLabel)  # 根据训练集创建决策树
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")
    createPlot(my_Tree)
    if LN[i] == 3:
        trainLabel = [1, 2, 3]
    elif LN[i] == 6:
        trainLabel = [1, 2, 3, 4, 5, 6, ]
    else:
        trainLabel = [1, 2, 3, 4, 5, 6, 7, 8]
    print("---------------------------------------------")
    print(my_Tree)
    wholeNum = len(train_data)
    correctNum = 0
    for i in train_data:
        if (classify(my_Tree, trainLabel, i[:-1]) == i[-1]):
            correctNum += 1
    print(correctNum / wholeNum)

