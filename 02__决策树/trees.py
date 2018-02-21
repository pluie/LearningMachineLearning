# python 3.5.3rc1 #
###################
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

# 计算香浓熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # 数据集中实例的总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] # 最后一个元素为标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0 # 新增键
        labelCounts[currentLabel] += 1 # 每个键值记录当前类别出现的次数
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries # 计算类别出现的概率
        shannonEnt -= prob * log(prob,2) # 以2为底求对数
    return shannonEnt

# 按给定特征划分数据集    
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
