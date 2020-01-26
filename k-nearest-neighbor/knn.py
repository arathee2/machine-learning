from sklearn.model_selection import train_test_split
import math
import numpy as np
import scipy.stats as sp


class KNN:
    
    def __init__(self, k=3, algorithm='classification'):
        """
        algorithm : str, 'classification' or 'regression'
        k         : int, number of nearest neighbors
        """
        
        self.k = k
        self.algorithm = algorithm
        pass

    
    def fit(self, train_X, train_y):
        
        # fit data in memory
        self.train_X = train_X
        self.train_y = train_y
        pass

    
    def predict(self, test_X):
        
        self.test_X = test_X
        
        # get label for each data point in test set
        self.test_y = []
        
        for test_observation in self.test_X:
            
            # compute Euclidean distance between each test observation and training data
            distances = []
            for train_observation in self.train_X:
                distance = np.linalg.norm(train_observation - test_observation, 2)
                distances.append(distance)
                pass


            # get closest targets
            nearest_k_indices = np.argsort(distances)[0:self.k]
            nearest_k_targets = self.train_y[nearest_k_indices]
            
            # make prediction
            if self.algorithm == 'regression':
                # prediction is mean (of nearest targets)
                self.test_y.append(np.mean(nearest_k_targets))
            elif self.algorithm == 'classification':
                # prediction is mode (of nearest targets)
                self.test_y.append(int(sp.mode(nearest_k_targets)[0]))
                pass
            pass
        
        return self.test_y
    

def accuracy(y,y_hat):
    nvalues = len(y)
    accuracy = sum(y == y_hat) / nvalues
    return accuracy  
    


# weigh and height data
X = np.array([[140, 40],
              [190, 90],
              [165, 70],
              [140, 45],
              [170, 100],
              [180, 70],
              [175, 65],
              [173, 75],
              [155, 55],
              [162, 58]])

# labels: 0 = female, 1 = male
y = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 0])

# divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# fit model
knn = KNN(k=3, algorithm='classification')
knn.fit(X_train, y_train)

# train accuracy
y_hat = knn.predict(X_train)
accuracy(y_train, y_hat)

# test accuracy
y_hat = knn.predict(X_test)
accuracy(y_test, y_hat)
