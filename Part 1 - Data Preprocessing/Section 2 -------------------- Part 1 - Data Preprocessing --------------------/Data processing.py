# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:21:33 2019

@author: Dell
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values #OR 3 in place of -1

#Taking care of missing data
from sklearn.impute import SimpleImputer as Imputer
imputer = Imputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])

#Encoding Cateagorial Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:,0] = labelencoder_country.fit_transform(X[:,0])
           #In above there is problem ,since numbers are relative it may think country with 2> country with 1 and similaarly other

           #onehotencoder cannot be used on strings so first convert to int i.e 0,1,2 here and then apply
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
           #the above is known as Dummy Encoding  i.e dividing cateagories into diff tables

labelencoder_dep = LabelEncoder()
y = labelencoder_dep.fit_transform(y)

#Splitting data into training sets and data sets
from sklearn.model_selection import train_test_split 
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

