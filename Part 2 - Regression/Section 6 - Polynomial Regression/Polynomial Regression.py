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

##Train test split
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)


#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
linear_reg =LinearRegression()
linear_reg.fit(X,y)

#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # ie c,x,x^2
X_poly = poly_reg.fit_transform(X)

#Now fitting X_poly into our multiple linear regression
linear_reg_2= LinearRegression()
linear_reg_2.fit(X_poly,y)


#Visualizing Linear reg model
plt.scatter(X,y,color ="red")
plt.plot(X,linear_reg.predict(X),color="blue")
plt.title("Truth Or Bluff(Linear Regression)" )
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()


#Visualizing Polynomial reg model
plt.scatter(X,y,color ="red")
plt.plot(X,linear_reg_2.predict(poly_reg.fit_transform(X)),color="blue")  #Not used X_poly in linear_reg_2.predict() to make it more general
plt.title("Truth Or Bluff(Polynomial Regression)" )
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
linear_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
linear_reg_2.predict(poly_reg.fit_transform([[6.5]]))