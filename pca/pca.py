'''
The following paper does an exceptional job of explaining PCA:
https://www.cc.gatech.edu/~lsong/teaching/CX4240spring16/pca_schlens.pdf

My notes available at:

'''


# import required packages
import numpy as np
from numpy import linalg
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class PCA():
    '''
    Perform Principal Component Analysis on given data.
    '''
    
    def __init__(self):
        pass
    
    
    def fit(self, X):
        '''
        Compute principal components for a given dataset.
        
        Parameters:
        ----------
        X: numpy array of shape (n_features, n_samples).
           The dataset for which to compute principal components.
        
        Returns:
        -------
        None
        '''
        
        # center data
        self.X = X
        self.centered_X = self._center(self.X)
        
        # compute eigenvalues and eigen vectors of covariance matrix of X
        cov_X = X @ X.T
        self.eigen_values, self.eigen_vectors = linalg.eig(cov_X)
        
        # sort eigen values and corresponding eigenvectors in descending order
        idx = self.eigen_values.argsort()[::-1]   
        self.eigen_values = self.eigen_values[idx]
        self.eigen_vectors = self.eigen_vectors[:,idx]

        # principal component matrix is inverse of matrix that has eigen vectors
        self.components = self.eigen_vectors.T
    
    def get_principal_components(self):
        '''
        Return principal component matrix computed after fitting a dataset.
        '''
        return self.components
    
    def get_new_data(self):
        '''
        Return transformed dataset matrix Y = PX
        '''
        Y = self.components @ self.centered_X
        return Y
    
    def get_explained_variance(self):
        '''
        Return variance explained by each component.
        '''
        explained_variance = self.eigen_values / self.eigen_values.sum()
        return explained_variance
    
    def _center(self, X):
        '''
        Center given data of the shape (n_features, n_samples).
        '''
        # compute mean for each variable (row-wise mean)
        means = X.mean(axis=1)
        
        # reshape means to subtract mean from each variable (row)
        means = means.reshape(1, -1).T
        
        # subtract mean
        centered = X - means
        return centered


# create synthetic data with redundant information
X, y = make_classification(n_samples=200, 
                           n_features=10, 
                           n_informative=5, 
                           n_redundant=3, 
                           n_repeated=1,
                           random_state=1)

# change shape to (n_features, n_samples)
X = X.T

# fit PCA model with data
pca = PCA()
pca.fit(X)

# check principal components
pcomps = pca.get_principal_components()
pcomps

# check proportion of variance explained by principal components
var = pca.get_explained_variance()
plt.plot(var)

# transform data into new dimension
X_new = pca.get_new_data()

# reduce number of features based on explained variance
X_new = X_new[:7, :]
X_new.shape
