# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:59:08 2019

@author: Dell
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data
dataset =pd.read_csv("50_Startups.csv")
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values


#Encoding Cateagorial Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_state = LabelEncoder()
X[:,3] = labelencoder_state.fit_transform(X[:,3])
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

#To avoid dummy variable trap
X=X[:,1:]


#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Fitting Multiple regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting result
y_pred = regressor.predict(X_test)



#Building the optimal model using Backward elimination method

import statsmodels.formula.api as sm
X = np.append( arr =np.ones((50,1)).astype(int) , values=X, axis =1)        #adding column of 1 in begg for const term b0 in eqn 
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05    #Significance level for hypothesis
X_opt = X[:, :]
X_Modeled = backwardElimination(X_opt, SL)



#repeating the procedure with new model
X=X_Modeled
#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Fitting Multiple regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting result
y_pred2 = regressor.predict(X_test)




