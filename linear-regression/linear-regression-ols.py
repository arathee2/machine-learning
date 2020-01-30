import math
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class LinearRegression:
    
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        """
        fit linear model to the data
        input:
            X_train: matrix of dimension (#training-observations, #features)
            y_train: vector of dimension (#observation, 1)
        """
        assert X_train.shape[0] == len(y_train), "Error! X and y must have same number of observations.\n X has {} observations.\n y has {} observations.".format(X_train.shape[0], len(y_train))
        
        
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.num_train_observations = self.X_train.shape[0]
        self.num_predictors = self.X_train.shape[1]
        
        # weights: dimension = (#features + 1, 1), an extra weight for bias
        self.__weights = np.zeros((self.num_predictors+1)).reshape((self.num_predictors + 1, 1))
        
        # stack a vector with ones to the left of the first column in X_train
        __ones = np.ones(self.num_train_observations).reshape((self.num_train_observations,1))
        self.X_train = np.concatenate((__ones, self.X_train), axis=1)
        
        # compute weights = (X.T * X)^(-1) * X.T * y
        self.__weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.X_train.T, self.X_train)), self.X_train.T), self.y_train)
        
        # store intercept and weights separately
        self.intercept = np.array([self.__weights[0]])
        self.coefficients = self.__weights[1:]  # np.delete(self.__weights, 0)
        pass
    
    
    def predict(self, X_test):
        """
        input:
            X_test: matrix of dimension (#testing-observations, self.num_predictors)
        output:
            y_hat_test: vector of dimension (#testing-observations, 1)
        """
        assert X_test.shape[1] == self.X_train.shape[1] - 1, "Error! Test data must have same number of features as train data.\n Features in trianing set = {}.\n Features in test set = {}.".format(self.X_train.shape[1]-1, X_test.shape[1])
        
        __num_test_observations = X_test.shape[0]
        
        # stack a vector with ones to the left of the first column in X_test
        __ones = np.ones(__num_test_observations).reshape((__num_test_observations, 1))
        __X_test = np.concatenate((__ones, X_test), axis=1)
        
        # make prediction
        __y_hat_test = np.matmul(__X_test, self.__weights)
        
        return __y_hat_test
        
    
    pass


# MAKE DATA

# height (cm)
X = np.array([140, 190, 165, 140, 170, 180, 175, 173, 155, 162]).reshape((10, 1))

# weight (kg)
y = np.array([ 40,  90,  70,  45, 100,  70,  65,  75,  55,  58])

# plot data
sns.scatterplot(X[:, 0], y)


# BUILD MODEL

# divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# fit model
model = LinearRegression()
model.fit(X_train, y_train)
y_hat_train = model.coefficients*X_train + model.intercept

# regression line
plt.scatter(X_train, y_train); plt.plot([min(X_train), max(X_train)], [min(y_hat_train), max(y_hat_train)], color='red'); plt.show()

# make predictions on test set
y_hat_test = model.predict(X_test)


# EVALUATE MODEL
def rmse(y, y_hat):
    assert len(y) == len(y_hat)
    sse = sum((y - y_hat)**2)
    return math.sqrt(sse/len(y))

def r_squared(y, y_hat):
    assert len(y) == len(y_hat)
    sse = sum((y - y_hat) ** 2)
    sst = sum((y - np.mean(y)) ** 2)
    return (1-(sse/sst))

rmse(y_test, y_hat_test)
r_squared(y_test, y_hat_test)    



