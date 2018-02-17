#python 3.5.3rc1#
#################
import numpy as np
import operator
from os import listdir

# 分类函数，其中：
# intX为分类的输入向量
# dataSet为训练样本集
# labels为标签向量
# k为选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] 
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    distances = ((diffMat**2).sum(axis=1))**0.5 # 距离计算公式
    sortedDistIndicies = distances.argsort() # 递增排序
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

# 创建数据和标签
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# 将TXT文件中的数据转为numpy
def file2matrix(filename):
    fr = open(filename,'r')
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    fr = open(filename,'r') #不知道为什么要重新打开才能读出数据
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    fr.close()
    return returnMat,classLabelVector

# 归一化数据    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet / (np.tile(ranges, (m,1))) # 特征值相除，不是矩阵除法
    return normDataSet, ranges, minVals

# 测试正确率   
def datingClassTest(hoRatio=0.5, k=6):
    # hoRatio将数据分成训练、测试两份；可使用交叉方法，此处没有
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt') # 得到numpy数据
    normMat, ranges, minVals = autoNorm(datingDataMat) # 归一化
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],
                                     normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],
                                     k)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


if __name__ == '__main__':
    datingClassTest(0.5, 6)
