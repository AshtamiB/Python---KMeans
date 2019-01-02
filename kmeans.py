# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 10:32:06 2018

@author: Ashtami
"""
import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

import numpy as np
from scipy.cluster.vq import kmeans,vq
from pylab import plot,show
import scipy.cluster.vq as spcv
#generate two clusters: a with 100 points, b with 50:
np.random.seed(4711) # for repeatability of this example
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]],
size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]],
size=[50,])
X = np.concatenate((a,b),)
print(X.shape) # 150 samples with 2 dimensions
centroids,_ = kmeans(X,2)# computing K-Means with K = 2
id,_ = vq(X,centroids)# assign each sample to a cluster
plot(X[id==0,0],X[id==0,1],'ob',X[id==1,0],X[id==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg', markersize=15)
show()


A,B = spcv.kmeans(X,2)
print(A)
print('----------------------------------------------')
print(B)
print('----------------------------------------------')
id,dist=spcv.vq(X,A)
print('----------------------------------------------')
print(id)
print(dist)



#-----------------------------------#

import os
os.chdir("C:/Users/Ashtami/Documents/Python/")

import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans,vq
from pylab import plot, show

DataFrame = pd.read_csv('irisdata.csv',header=None)
DataFrame = DataFrame.iloc[:, :3]
DataMatrix = DataFrame.as_matrix()
InputMatrix = np.matrix(DataMatrix)
centroids,_ = kmeans(InputMatrix,3)
id,_ = vq(InputMatrix,centroids)
print(centroids)
print(id)
plot(InputMatrix[id==0,0],InputMatrix[id==0,1],'*b',InputMatrix[id==
1,0],InputMatrix[id==1,1],'vr',InputMatrix[id==2,0],InputMatrix[id==
2,1],'og',linewidth=5.5)
show()

###############################
#K-means
import scipy.cluster.vq as spcv
centroids,var = spcv.kmeans(Dataset,Number_of_Clusters)
id,dist = spcv.vq(Dataset,centroids)
 
#OR
import numpy as np
from sklearn.cluster import KMeans

### For the purposes of this example, we store feature data from our
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

X=np.matrix(zip(f1,f2))
kmeans = KMeans(n_clusters=2).fit(X)
labels = kmeans.labels_


##############REVISION EXAMPLE########################
#-------------------- Generating Synthetic Data -------------#
X, y_true = make_blobs(n_samples=300, n_features=3, centers=4, cluster_std=0.70,
random_state=0)
x_ax = X[:, 0]
y_ax = X[:, 1]
z_ax = X[:, 2]
#-------------------------- KMEAN ---------------------------#
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_
#-------------------------- Plotting ----------------------- #
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(x_ax, y_ax, z_ax, s=150)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_ax, y_ax, z_ax, c=y_kmeans, s=100, cmap='viridis')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200)
plt.show()