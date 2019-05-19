import os
import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
import random
import math
import time
import matplotlib.pyplot as plt

pca = PCA(n_components=40)


def Calc_Distance(x, y):
    s = 0
    for i in range(40):
        s += (x[i] - y[i]) ** 2
    return math.sqrt(s)


def TrainData_Init(lst):
    tmpdata = np.zeros(shape=(200, 10304))
    base_Dir = os.path.abspath('.')
    cnt = 0
    for i in range(1, 41):
        for each in lst:
            filename = base_Dir + r"\att_faces\s" + str(i) + '\\' + str(each) + ".pgm"
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            tmp_Img = np.reshape(img, (10304,))
            tmpdata[cnt] = tmp_Img
            cnt += 1
    pca.fit(tmpdata)
    return pca.transform(tmpdata)


def Testdata_Init(lst):
    tmpdata = np.zeros(shape=(200, 10304))
    base_Dir = os.path.abspath('.')
    cnt = 0
    for i in range(1, 41):
        for each in lst:
            filename = base_Dir + r"\att_faces\s" + str(i) + '\\' + str(each) + ".pgm"
            img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            tmp_Img = np.reshape(img, (10304,))
            tmpdata[cnt] = tmp_Img
            cnt += 1
    return pca.transform(tmpdata)


def KNN(trainset, testset, k):
    isRight = 0
    tlst = []
    for i in range(200):
        tlst.append(int(i / 5))
    for test in range(200):
        lst = []
        for train in trainset:
            lst.append(Calc_Distance(testset[test], train))
        tmp = zip(lst, tlst)
        tmp = list(tmp)
        tmp.sort(key=lambda x: x[0])
        dic = {}
        for i in range(k):
            dic[tmp[i][1]] = 0
        for i in range(k):
            dic[tmp[i][1]] += 1
        m = 1
        k = -1
        for each in dic:
            if dic[each] > m:
                m = dic[each]
                k = -1
        if k == -1:
            k = tmp[0][1]
        if k == int(test / 5):
            print("当前图片是第{}类，识别为第{}类，结果正确".format(int(test / 5), k))
            isRight += 1
        else:
            print("当前图片是第{}类，识别为第{}类，结果错误".format(int(test / 5), k))
    print("正确率为: {}%".format(isRight / 200 * 100))
    print(isRight / 200)
    return isRight / 200


if __name__ == '__main__':
    ls = [x for x in range(1, 11)]
    ra = []
    for rou in range(5):
        start = time.time()
        lst = random.sample(ls, 5)
        print("当前选择的数据作为训练集的是", lst)
        trainset = TrainData_Init(lst)  # 200*40 Matrix
        lst2 = []
        for i in range(1, 11):
            if i in lst:
                continue
            else:
                lst2.append(i)
        testset = Testdata_Init(lst2)
        rate = KNN(trainset, testset, 8)
        ra.append(rate * 100)
        end = time.time()
        print("总共花费时间为: {}秒".format(end - start))
        print("===========================================")
    s = 0
    for each in ra:
        s += each
    print(ra)
    print("平均识别率: {}%".format(s / 5))
    x = range(5)
    plt.plot(x, ra)
    plt.title("KNN")
    plt.xlabel("The experiment number")
    plt.ylabel("Correct Rate(%)")
    plt.legend()
    plt.show()
