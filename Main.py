import os
import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
import random


def Data_Init(lst):
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
    pca = PCA(n_components=40)
    return pca.fit_transform(tmpdata)


if __name__ == '__main__':
    lst = []
    for i in range(5):
        lst.append(random.randint(1, 10))
    data = Data_Init(lst)
    print(data)
