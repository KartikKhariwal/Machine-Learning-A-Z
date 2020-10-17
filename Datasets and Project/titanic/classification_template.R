# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:23:18 2019

@author: Dell
"""

#Titanic Survivor Predictor 

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing training set
dataset= pd.read_csv("train.csv")
X_train = dataset.iloc[:,[2,3,4,5,6,7,8,9,10,11]].values
y_train = dataset.iloc[:,1].values

dataset.info()