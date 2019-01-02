# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:45:52 2018

@author: Ashtami
"""

import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans,vq
from pylab import plot, show
import sklearn.preprocessing as skp
import sklearn.decomposition as skd
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(return_X_y=False) #load the diabetes dataset

X = diabetes.data
y = diabetes.target
scaler = skp.StandardScaler().fit(X)

standardized_X = scaler.transform(X)
print(standardized_X)

Normalized_X = skp.normalize(X, norm='l2')
print(Normalized_X)

pca = skd.PCA(n_components=4)
pca.fit(X)
Dataset_Reduced_Dim = pca.transform(X)
print(Dataset_Reduced_Dim )
U, S, V = np.linalg.svd(Dataset)


centroids,_ = kmeans(Dataset_Reduced_Dim,4) # computing K-Means with K = 2
id,_ = vq(Dataset_Reduced_Dim,centroids) # assign each sample to a cluster
plot(Dataset_Reduced_Dim[id==0,0],Dataset_Reduced_Dim[id==0,1],'ob',Dataset_Reduced_Dim[id==1,0],Dataset_Reduced_Dim[id==1,1],'or',
     Dataset_Reduced_Dim[id==2,0],Dataset_Reduced_Dim[id==2,1],'*b',Dataset_Reduced_Dim[id==3,0],Dataset_Reduced_Dim[id==3,1],'og')
plot(centroids[:,0],centroids[:,1],'sg', markersize=15)
show()

#OR
centroids,var = spcv.kmeans(Dataset_Reduced_Dim,4)
id,dist = spcv.vq(Dataset_Reduced_Dim,centroids)
print(centroids)
print('---------------------------------------------------')
print(id)
