# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:00:34 2019

@author: Dell
"""
#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Importing Data
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =1/3 ,random_state =0)


#Fitting simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
     

#Predicting the result
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()