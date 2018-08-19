# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:46:17 2018

@author: Yash Upadhyay
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
lung = pd.read_csv('lung.txt', sep=',',header=0)
lung=lung.replace('?',0)
X_lung = lung.iloc[:,0:56].values
y_lung = lung.iloc[:, -1].values


from sklearn.cross_validation import train_test_split
Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_lung, y_lung, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xl_train = sc.fit_transform(Xl_train)
Xl_test = sc.transform(Xl_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 56))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(Xl_train, yl_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
yl_pred = classifier.predict(Xl_test)
yl_pred = (yl_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yl_test, yl_pred)

