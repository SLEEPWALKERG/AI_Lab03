from sklearn.decomposition import PCA
import numpy as np
import os

def func(X):
    pca = PCA(n_components=40)
    pca.fit(X)
    print(pca.explained_variance_ratio_)