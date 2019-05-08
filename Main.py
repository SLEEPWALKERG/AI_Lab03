import os
import cv2 as cv
import numpy as np
import PCA
def Data_Init():
    Base_Dir = os.path.abspath('.')
    filename = Base_Dir + r"\att_faces\s1\1.pgm"
    img = cv.imread(filename, 0)
    rows, cols = img.shape
    print(rows,cols)
    #Vector_Img = np.zeros(1, rows*cols)
    Vector_Img = np.reshape(img, (1,rows*cols))
    print(len(Vector_Img[0]))
    PCA.func(Vector_Img)


if __name__ == '__main__':
    Data_Init()