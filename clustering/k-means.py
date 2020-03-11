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
    
    def __init__(self, num_clusters):
        '''
        num_clusters : int, number of clusters
        '''
        self.num_clusters = num_clusters
        pass
    
    # fit
    def fit(self, X):
        '''
        Run k-means clustering on dataset X.
        X : numpy 2D array of floats, rows are observations and columns are features
        '''
        self.X           = X
        num_features     = self.X.shape[1]

        # initialize cluster centroids until all centroids have some data points in their vicinity
        properly_initialized = False
        while(properly_initialized == False):
            self.centroids = np.zeros((self.num_clusters, num_features))
            for i in range(self.num_clusters):
                self.centroids[i, :] = self.initialize_centroid()
                pass
            
            if(self.is_initialized_correctly()):
                properly_initialized = True
                pass
            pass        
        
        # update centroids
        are_centroids_unstable = True
        while(are_centroids_unstable):
            self.points_to_centroids_map = self.assign_points_to_centroids()
            centroids_updated            = self.update_centroids()
            
            # centroids are stable when NONE of them change their position
            if((centroids_updated == self.centroids).all()):
                are_centroids_unstable = False
                pass
            
            self.centroids = centroids_updated
            pass
        
        pass
    
    
    def is_initialized_correctly(self):
        '''
        Test whether all centroids have some points in their vicinity
        '''
        is_correctly_initialized = True
        
        points_to_centroids_map = self.assign_points_to_centroids()
        num_observations_in_subset = []
        for c in range(self.centroids.shape[0]):
            num_observations_in_subset.append(self.X[np.where(points_to_centroids_map == c)].shape[0])
            pass
        
        for i in num_observations_in_subset:
            if(i == 0):
                is_correctly_initialized = False
                pass
            pass
        
        return is_correctly_initialized
    
    
    def get_clusters(self):
        '''
        Return a list of centroids that each point belong to.
        '''
        return self.points_to_centroids_map
    
    # transform
    def transform(self, X):
        ''' 
        Assign cluster to observations in X .
        '''
        return self.assign_points_to_centroids(X, self.centroids)
            
        
    # sse
    def get_sse(self):
        '''
        Return sum of squared errors.
        '''
        sse = 0
        
        # compute sse for each cluster
        for c in range(self.num_clusters):
            c_points = np.where(self.points_to_centroids_map == c)
            points   = self.X[c_points]
            
            # sum all the distances between a centroid and the points that belong to the centroid
            for point in points:
                sse += self.get_distance(self.centroids[c], point)
                pass
            pass
        
        return sse
            
        
    # visualize clusters
    def visualize_clusters(self):
        '''
        Plot X and color them by their clusters.
        '''
        
        # list containing centroids that the point in X belong to
        mapping         = self.points_to_centroids_map.reshape((-1, 1))
        
        # plot points in the vicinity of each centroid
        for centroid in range(self.num_clusters):
            plt.scatter(self.X[np.where(mapping == centroid), 0], self.X[np.where(mapping == centroid), 1], label='Cluster '+str(centroid), color=np.random.rand(3,))
            pass
        
        # plot centroids
        plt.scatter(self.centroids[0], self.centroids[1], color=np.random.rand(3,), label='Cluster centroids')
        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Clusters')



    # helper functions
    def initialize_centroid(self):
        '''
        Initialize a centroid of dimension [1, num_features]. Each element of the centroid is randomly drawn from the minimum and maximum values of a feature.
        '''
        num_features    = self.X.shape[1]

        # compute min and max value in each feature
        feature_min_max = []
        for i in range(num_features):
            feature_min_max.append((np.min(self.X[:, i]), np.max(self.X[:, i])))
            pass

        # generate centroid: a random vector of size num_features where each number is drawn uniformly within the min and max of each feature
        centroid = []
        for i in range(len(feature_min_max)):
            centroid.append(np.random.uniform(feature_min_max[i][0], feature_min_max[i][1]))
            pass

        return centroid

    def assign_points_to_centroids(self):
        '''
        Return a list of length equal to the number of points in the data. Each element of list is the centroid that is closest to that point.
        '''
        points_to_centroids_map = [0 for i in range(self.X.shape[0])]
        
        
        for i, point in enumerate(self.X):
            point_to_centroid_distances = []
            for centroid in self.centroids:
                
                # compute distance between a point and a centroid
                point_to_centroid_distances.append(self.get_distance(centroid, point))
                pass
            
            # append the centroid number that is closest to the point
            points_to_centroids_map[i] = np.argmin(point_to_centroid_distances)
            pass

        return np.array(points_to_centroids_map)
        

    def update_centroids(self):
        '''
        Assign each centroid the mean of all the points in their vicinity.
        '''
        centroids_updated = np.zeros((self.centroids.shape[0], self.centroids.shape[1]))
        for c in range(self.centroids.shape[0]):
            c_points             = np.where(self.points_to_centroids_map == c)
            
            if(self.X[c_points].shape[0] == 0):
                # respawn centroids if no points present in their vicinity
                centroids_updated[c] = self.initialize_centroid()
                pass
            else:
                # update centroids if they have some points in their vicinity
                centroids_updated[c] = np.apply_along_axis(np.mean, 0, self.X[c_points])
            pass

        return centroids_updated

    def get_distance(self, a, b):
        '''
        Euclidean distance between two n-dimensional points.
        '''
        a = np.array(a)
        b = np.array(b)
        return np.sqrt(np.sum(np.power((a - b), 2)))
    
    pass


# generate data
num_observations = 200
num_features     = 2
num_blobs        = 4
X, y = datasets.make_blobs(n_samples=num_observations, n_features=num_features, centers=num_blobs)
plt.scatter(X[:, 0], X[:, 1])

'''
Run k-means using self-implemented class
'''
# choose k using elbow method
k_range = range(2, 10)
sse = []
for k in k_range:
    model = KMeans(num_clusters=k)
    model.fit(X)
    sse.append(model.get_sse())
    pass

plt.plot(k_range, sse)
plt.xlabel("K",)
plt.ylabel("Sum of squared errors")
plt.title("Elbow Curve")

# fit k-means using k found through elbow method
k = 4
model = KMeans(num_clusters=k)
model.fit(X)

# sum of squared error
model.get_sse()

# visualize clusters
model.visualize_clusters()


'''
Run k-means using scikit-learn's implementation
'''
# scikit-learn's implementation
sklearn_kmeans = cluster.KMeans(n_clusters=k, random_state=0)
sklearn_kmeans.fit(X)

# sum of squared error
sklearn_kmeans.inertia_

# visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_)
plt.xlabel("x1",)
plt.ylabel("x2")
plt.title("Clusters")
