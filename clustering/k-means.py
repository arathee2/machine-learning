'''
Code adapted from concepts presented in the following text:
Chapter 7 of Introduction to Data Mining by Tan, Steinbach, Karpatne and Kumar
https://www-users.cs.umn.edu/~kumar001/dmbook/ch7_clustering.pdf
'''

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cluster

class KMeans:
    '''
    Perform k-means clustering.
    '''
    
    def __init__(self, num_clusters, max_steps=100, convergence_threshold=0.05):
        '''
        Input:
            num_clusters         : int, number of clusters
            max_steps            : int, maximum number of iterations to run the algorithm
            convergence_threshold: if the means change less than this value
                                   in an iteration, declare convergence
        '''
        self.num_clusters = num_clusters
        self.MAX_STEPS = max_steps
        self.CONVERGENCE_THRESHOLD = convergence_threshold
        pass
    
    
    # fit
    def fit(self, X):
        '''
        Run k-means clustering on dataset X.
        Input:
            X : numpy 2D array of floats, rows contain observations and columns contain features
        
        After running the fit() method, one can access the following variables:
            means             : a numpy 2D matrix of floats, each row is a cluster mean 
                                and the number of features are equal to the number of features in the data
            cluster_assignment: a list of the cluster assignments for each sample
            dissimilarity     : sum of squared error of the Euclidean distance from each point to the cluster mean
        '''
        self.X           = X
        num_observations = self.X.shape[0]
        num_features     = self.X.shape[1]
        
        # initialize means: pick k means randomly from the data
        self.means = self.X[np.random.randint(low=0, high=num_observations-1, size=self.num_clusters), :]     
        
        # update means
        num_steps = 0
        while(num_steps <= self.MAX_STEPS):
            self.cluster_assignment = self.assign_points_to_means(self.X, self.means)
            means_updated           = self.update_means()
            
            # if algorithm has not converged, update cluster means
            mean_movement = np.mean(means_updated, axis=1) - np.mean(self.means, axis=1)
            if(np.all(mean_movement < self.CONVERGENCE_THRESHOLD)):
                self.means = means_updated
                break
            else:
                self.means = means_updated
                pass
            
            num_steps += 1
            pass
        
        self.dissimilarity = self.get_dissimilarity()
        pass
      
        
    # sse
    def get_dissimilarity(self):
        '''
        Return sum of squared errors.
        '''
        sse = 0
        
        # compute sse for each cluster
        for c in range(self.num_clusters):
            c_points = np.where(self.cluster_assignment == c)
            points   = self.X[c_points]
            
            # sum all the distances between a centroid and the points that belong to the centroid
            for point in points:
                sse += self.get_distance(self.means[c], point)
                pass
            pass
        
        return sse
    
    # visualize clusters
    def visualize_clusters(self, title='k-means clustering'):
        '''
        Plot X and color them by their clusters.
        '''
        
        # list containing means that the point in X belong to
        mapping         = self.cluster_assignment.reshape((-1, 1))
        
        # plot points in the vicinity of each centroid
        for centroid in range(self.num_clusters):
            plt.scatter(self.X[np.where(mapping == centroid), 0], self.X[np.where(mapping == centroid), 1], label='cluster '+str(centroid), color=np.random.rand(3,))
            pass
        
        # plot means
        plt.scatter(self.means[:, 0], self.means[:, 1], color=np.random.rand(3,), label='means')
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(title)
        pass
    
    
    # helper functions
    def assign_points_to_means(self, X, means):
        '''
        Return a list of length equal to the number of points in the data. Each element of list is the centroid that is closest to that point.
        '''
        cluster_assignment = [0 for i in range(X.shape[0])]
        
        
        for i, point in enumerate(X):
            point_to_centroid_distances = []
            for centroid in means:
                
                # compute distance between a point and a centroid
                point_to_centroid_distances.append(self.get_distance(centroid, point))
                pass
            
            # append the centroid number that is closest to the point
            cluster_assignment[i] = np.argmin(point_to_centroid_distances)
            pass

        return np.array(cluster_assignment)
        

    def update_means(self):
        '''
        Assign each centroid the mean of all the points in their vicinity.
        '''
        means_updated = np.zeros((self.means.shape[0], self.means.shape[1]))
        for c in range(self.means.shape[0]):
            c_points             = np.where(self.cluster_assignment == c)
            
            if(self.X[c_points].shape[0] == 0):
                # respawn means if no points present in their vicinity
                return self.X[np.random.randint(low=0, high=num_observations-1, size=self.num_clusters), :]
                pass
            else:
                # update means if they have some points in their vicinity
                means_updated[c] = np.apply_along_axis(np.mean, 0, self.X[c_points])
            pass

        return means_updated

    def get_distance(self, a, b):
        '''
        Euclidean distance between two n-dimensional points.
        '''
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(np.sum(np.power((a - b), 2)))
    
    pass

'''
Generate data
'''
num_samples  = 5000
num_features = 4
num_blobs    = 2
X, y = datasets.make_blobs(n_samples=num_samples, n_features=num_features, centers=num_blobs, random_state=1)
plt.scatter(X[:, 0], X[:, 1])


'''
Run k-means using self-implemented class
'''
# choose k using elbow method
k_range = range(1, 11)
sse = []
for k in k_range:
    model = KMeans(num_clusters=k)
    model.fit(X)
    sse.append(model.dissimilarity)
    pass

plt.plot(k_range, sse)
plt.xlabel("Number of clusters",)
plt.ylabel("Sum of squared errors")
plt.title("K-means Elbow Curve")

# run k-means with two clusters
k = 2
model = KMeans(num_clusters=k, max_steps=200, convergence_threshold=0.01)
model.fit(X)

# sum of squared error
model.dissimilarity

# visualize clusters
model.visualize_clusters()

'''
Run k-means using sklearn implementation
'''
k = 2
sklearn_kmeans = cluster.KMeans(n_clusters=k, random_state=0)
sklearn_kmeans.fit(X)

# sum of squared error
sklearn_kmeans.inertia_

# visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_)
plt.xlabel("x1",)
plt.ylabel("x2")
plt.title("Clusters")
