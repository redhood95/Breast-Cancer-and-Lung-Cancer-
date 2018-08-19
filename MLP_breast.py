# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:31:29 2018

@author: Yash Upadhyay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

breast = pd.read_csv('breast.txt',sep=',',header=0)

breast = breast.replace('?',0)

X_breast = breast.iloc[:,1:10 ].values
y_breast = breast.iloc[:, -1].values

from sklearn.cross_validation import train_test_split
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_breast, y_breast, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

Xb_train = sc.fit_transform(Xb_train)
Xb_test = sc.transform(Xb_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(Xb_train, yb_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
yb_pred = classifier.predict(Xb_test)
yb_pred = (yb_pred > 0.4)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yb_test, yb_pred)