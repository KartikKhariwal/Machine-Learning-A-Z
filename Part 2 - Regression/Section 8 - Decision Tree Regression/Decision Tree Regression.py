# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:,[1]].values #[1] as matrix needed
y = dataset.iloc[:,-1].values


#fitting Decision regression
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y) 

#predicting a new value
regressor.predict([[6.5]])

#Visualizing Decision Tree reg model
plt.scatter(X,y,color ="red")
plt.plot(X,regressor.predict(X),color="blue")  
plt.title("Truth Or Bluff(Decision Regression)" )
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()
#Problem with above visualization is its showing continous as we are plotting on Dataset points only

# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


