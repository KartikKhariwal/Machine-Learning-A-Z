#Kmeans Clusteringr

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,-2:].values
#We have to cluster on the basis of Income and spending score

#Using elbow method to choose number of clusters
from sklearn.cluster import KMeans
WCSS = []  # Another Name is Inertia 
for k in range(1,11):
    kmeans  =KMeans(n_clusters = k, init="k-means++",max_iter =300 , n_init =10, random_state = 0 )
    kmeans.fit(X)
    wcss = kmeans.inertia_
    WCSS.append(wcss)
plt.plot(range(1,11),WCSS)
plt.title("ELBOW Method")
plt.xlabel("No. Of Clusters")
plt.ylabel("WCSS Value")
plt.show()

#From the plot we obtained
K=5

#Applying Kmeans to data
kmeans  =KMeans(n_clusters = K, init="k-means++",max_iter =300 , n_init =10, random_state = 0 )
y_kmeans = kmeans.fit_predict(X)

#Visualizing the clusters of Kmeans Plot
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100, color="red",label="Careful")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100, color="blue",label="Standard")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100, color="green",label="Target")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100, color="cyan",label="Careless")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100, color="magenta",label="Sensible")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,color="yellow",label="Centroids")
plt.title("Kmeans Clustering")
plt.xlabel("Salary")
plt.ylabel("Spening Score")
plt.legend()
plt.show()

