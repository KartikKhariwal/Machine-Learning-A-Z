# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 12:54:44 2019

@author: Dell
"""

#Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# dataset import
dataset =pd.read_csv("Position_Salaries.csv")
X= dataset.iloc[:,[1]].values
y=dataset.iloc[:,-1].values








#Fitting the Random Forest regression model to dataset
from sklearn.ensemble import RandomForestRegressor
ran_reg = RandomForestRegressor(n_estimators = 300,random_state = 0)
ran_reg.fit(X,y) 


#predicting a new value
for i in range(0,11):
    print(ran_reg.predict([[i]]))



# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, ran_reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


