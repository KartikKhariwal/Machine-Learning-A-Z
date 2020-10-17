

#My First ML Project

#Iris classification

#Imporfting the libraries
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics #for checking the model accuracy


#Importing the dataset
dataset = pd.read_csv("Iris.csv")

dataset.head(2) #show the first 2 rows from the dataset

dataset.info()  #checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed

dataset.drop('Id',axis=1,inplace=True)
 #dropping the Id column as it is unecessary, axis=1 specifies that it should be column wise, inplace =1 means the changes should be reflected into the dataframe

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

#Splitting independent and dependent var from dataset
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values


#Splitting data into training sets and data sets
from sklearn.model_selection import train_test_split 
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random   _state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS.fit(X_train)
X_train = SS.transform(X_train)
X_test = SS.transform(X_test)
############################ Not required (used for plotting only)


# Applying Kernel PCA for dimensionality reduction
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)




#Applying SVM
from sklearn.svm import SVC  #for Support Vector Machine (SVM) Algorithm
classifier = SVC( kernel ="rbf" )
classifier.fit(X_train,y_train) # we train the algorithm with the training data and the training output
y_svm_pred = classifier.predict(X_test) #now we pass the testing data to the trained algorithm
print('The accuracy of the SVM is:',metrics.accuracy_score(y_svm_pred,y_test))



#Applying Logistic Reression
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_log_pred = classifier.predict(X_test)
print("The Accuracy of the LogisticClassifer is:", metrics.accuracy_score(y_log_pred,y_test))


#Applying Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_dec_pred = classifier.predict(X_test)
print("The Accuracy of the Decision tree is:", metrics.accuracy_score(y_dec_pred,y_test))


#Applying K-nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)
y_knn_pred = classifier.predict(X_test)
print("The Accuracy of the KNN is:", metrics.accuracy_score(y_knn_pred,y_test))


#
#Applying Random Forest classifier model to training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10 , criterion='entropy' ,random_state=0 )
classifier.fit(X_train,y_train)
y_forest_pred = classifier.predict(X_test)
print("The Accuracy of the Random Forest is:", metrics.accuracy_score(y_forest_pred,y_test))


