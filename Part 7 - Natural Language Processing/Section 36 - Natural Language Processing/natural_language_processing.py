# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# delimiter to tell its tsv 
# quoting 3 to avoid quotes

# Cleaning the texts
import re    #pattern lib
import nltk    # to import list of irrelevant word + imp for nlp
nltk.download('stopwords')  #list of irrelevant word in many lang
from nltk.corpus import stopwords    #inpoting list
from nltk.stem.porter import PorterStemmer     #for stemming 

corpus = []
for i in range(0, 1000): #cleaning one review at a time
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])   # to retian only a-zA-z char in review 
    review = review.lower()   # tolower
    review = review.split()   # string to list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    #set used for faster implementation  
    #ps.stem(word) for stemming  
    review = ' '.join(review)
    #list of words to string
    corpus.append(review)

# Creating the Bag of Words model
 #its a model  with review vs distinc words matrix where each word act as a feature 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)  #maxfeatures takes words acc to their freq
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Efficiency = "+ str((cm[0][0]+cm[1][1])/(sum(sum(cm)))) )