# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 14:57:14 2018

@author: Yash Upadhyay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
lung = pd.read_csv('lung.txt', sep=',',header=0)
breast = pd.read_csv('breast.txt',sep=',',header=0)

#replacing missing values
lung=lung.replace('?',0)
breast = breast.replace('?',0)


X_lung = lung.iloc[:,0:56].values
y_lung = lung.iloc[:, -1].values


X_breast = breast.iloc[:,1:10 ].values
y_breast = breast.iloc[:, -1].values

from sklearn.cross_validation import train_test_split
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lung, y_lung, test_size = 0.25, random_state = 0)

Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_breast, y_breast, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xl_train = sc.fit_transform(Xl_train)
Xl_test = sc.transform(Xl_test)

Xb_train = sc.fit_transform(Xb_train)
Xb_test = sc.transform(Xb_test)


from sklearn.naive_bayes import GaussianNB
Lung_classifier = GaussianNB()
breast_classifier = GaussianNB()

Lung_classifier.fit(Xl_train,yl_train)
yl_pred = Lung_classifier.predict(Xl_test)

breast_classifier.fit(Xb_train,yb_train)
yb_pred = breast_classifier.predict(Xb_test)

from sklearn.metrics import confusion_matrix
lung_cm = confusion_matrix(yl_test, yl_pred)
breast_cm = confusion_matrix(yb_test, yb_pred)


















