# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:03:02 2019

@author: mca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
billdata=pd.read_csv("E:\\Srinivas\\SVM\\bill_authentication.csv")
x=billdata.drop('Class',axis=1)
y=billdata['Class']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
svclassifier =SVC(kernel='linear')
svclassifier.fit(x_train,y_train)

y_pred=svclassifier.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))