# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:01:08 2018

@author: Farhan
"""

#import numpy as np
#import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('train.csv').as_matrix()
clf = KNeighborsClassifier(n_neighbors=3)

xtrain = data[0:21000,1:]
train_label = data[0:21000,0]

clf.fit(xtrain,train_label)

#testing data
xtest = data[21000:,1:]
actual_label = data[21000:,0]
result = clf.predict(xtest)
#d=xtest[18]
#d.shape = (28,28)
#pt.imshow(255-d,cmap='gray')
#pt.show()
#print(clf.predict([xtest[18]]))
count =0.000
for i in range(0,21000):
    if result[i] == actual_label[i]:
        count = count+1

accuracy = (count/21000) * 100
print(accuracy)