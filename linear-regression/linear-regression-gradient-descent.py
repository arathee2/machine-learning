import math
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    
    def __init__(self, method='minimize_cost'):
        self.method = method
        pass
    
    def fit(self, X_train, y_train, epochs=100, lr=0.01):
        """
        fit linear model to the data
        input:
            X_train: matrix of dimension (#training-observations, #features)
            y_train: vector of dimension (#observation, 1)
        """
        
        dimension_error = "Error! X and y must have same number of observations.\n X has {} observations.\n y has {} observations.".format(X_train.shape[0], len(y_train))
        assert X_train.shape[0] == len(y_train), dimension_error
        
        
        self.X_train = X_train
        self.num_train_observations = self.X_train.shape[0]
        self.num_predictors = self.X_train.shape[1]
        self.y_train = y_train.reshape((self.num_train_observations, 1))
        
        # standardize X
        self.X_train, self.__X_mu, self.__X_sigma = self.__standardize_data(self.X_train)
        self.y_train, self.__y_mu, self.__y_sigma = self.__standardize_data(self.y_train)
        
        
        # weights: dimension = (# features + 1, 1), an extra weight for bias
        self.__weights = np.random.normal(0, 1, (self.num_predictors + 1)).reshape(self.num_predictors + 1, 1)
        
        # stack a vector with ones to the left of the first column in X_train
        __ones = np.ones(self.num_train_observations).reshape((self.num_train_observations,1))
        self.X_train = np.concatenate((__ones, self.X_train), axis=1)
        
        # compute weights
        if self.method == 'epochs':
            # run gradient descent for specific epochs
            for i in range(epochs):

                # calculate derivate: dimension(dl) = (num_features + 1, 1)
                dLoss = self.__derivative(self.X_train, self.y_train, self.__weights)

                # update weights
                assert dLoss.shape == self.__weights.shape
                self.__weights = self.__weights - (lr * dLoss)
                pass
        
            pass
        
        elif self.method == 'minimize_cost':
            # run gradient descent until the cost stops improving
            
            # minimum cost improvement criteria
            tolerance = 0.01
            
            # initial improvement
            cost_improvement = math.inf
            
            # stop algorithm if algorithm runs for too long
            MAX_ITER = 1000
            iterations = 0
            
            while (abs(cost_improvement) > tolerance) or (iterations == MAX_ITER):
                
                # calculate derivate: dim(dl) = (num_features + 1, 1)
                dLoss = self.__derivative(self.X_train, self.y_train, self.__weights)
                
                # initial cost
                initial_cost = self.__get_cost(self.X_train, self.y_train, self.__weights)
                
                # update weights
                assert dLoss.shape == self.__weights.shape
                self.__weights = self.__weights - (lr * dLoss)
                
                # final cost
                final_cost = self.__get_cost(self.X_train, self.y_train, self.__weights)
                
                # delta_cost
                cost_improvement = final_cost - initial_cost
                
                iterations += 1
                pass
            
            pass
        
        else:
            ValueError("Please pass either of the following methods: \n 'minimize_cost' \n 'epochs' ") 
            pass
        
        # store intercept and weights separately
        self.intercept = np.array([self.__weights[0]])
        self.coefficients = self.__weights[1:]
        pass
    
    
    def predict(self, X_test):
        """
        input:
            X_test: matrix of dimension (# testing-observations, self.num_predictors)
        output:
            y_hat_test: vector of dimension (# testing-observations, 1)
        """
        
        dimension_error = "Error! Test data must have same number of features as train data.\n Features in trianing set = {}.\n Features in test set = {}.".format(self.X_train.shape[1]-1, X_test.shape[1])
        assert X_test.shape[1] == self.X_train.shape[1] - 1, dimension_error
        
        __num_test_observations = X_test.shape[0]
        
        # standardize X using X_train's mu and sigma
        X_test, _, _ = self.__standardize_data(X_test, self.__X_mu, self.__X_sigma)
        
        # stack a vector with ones to the left of the first column in X_test
        __ones = np.ones(__num_test_observations).reshape((__num_test_observations, 1))
        __X_test = np.concatenate((__ones, X_test), axis=1)
        
        # make prediction using weights
        __y_hat_test = np.matmul(__X_test, self.__weights)
        
        # undo standardize y using y_train's mu and sigma
        __y_hat_test = self.__inverse_standardize(__y_hat_test, self.__y_mu, self.__y_sigma)
        
        # flatten array
        __y_hat_test = __y_hat_test.ravel()
        
        return __y_hat_test

    
    def __derivative(self, X, y, weights):
        """ return derivative of the loss function """
        
        num_observations = X.shape[0]
        y_hat = np.matmul(X, weights)
        
        dLoss = np.zeros((X.shape[1], 1))
        for i in range(num_observations):
            dl = (-2/num_observations) * (float(y[i]) - float(y_hat[i])) * X[i, :]
            dLoss += dl.reshape(X.shape[1], 1)
        
        return dLoss        
    
    
    def __get_cost(self, X, y, weights):
        """ return MSE """
        num_observations = X.shape[0]
        y_hat = np.matmul(X, weights)
        
        return np.sum((y - y_hat) ** 2) / num_observations
    
    
    def __standardize_data(self, M, mu_vector=None, sigma_vector=None):
        """ calculate z-score of all columns of data """
        if mu_vector == None:
            mu_vector = np.apply_along_axis(np.mean, 0, M)
        
        if sigma_vector == None:
            sigma_vector = np.apply_along_axis(np.std, 0, M)
        
        Z = (M - mu_vector) / sigma_vector
        
        return Z, mu_vector, sigma_vector
    
    
    def __inverse_standardize(self, Z, mu, sigma):
        """ change z-scores back to original scale """
        return (Z * sigma) + mu
    
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
model = LinearRegression(method='minimize_cost')
model.fit(X_train, y_train, epochs=500)
y_hat_train = (model.coefficients * X_train) + model.intercept
model.coefficients
model.intercept

# regression line
plt.scatter(X_train, y_train); plt.plot([min(X_train), max(X_train)], [min(y_hat_train), max(y_hat_train)], color='red'); plt.show()

# make predictions on test set
y_hat_test = model.predict(X_test)
y_hat_test
y_test

# EVALUATE MODEL
def rmse(y, y_hat):
    assert len(y) == len(y_hat)
    sse = sum((y - y_hat) ** 2)
    return math.sqrt(sse / len(y))

def r_squared(y, y_hat):
    assert len(y) == len(y_hat)
    sse = sum((y - y_hat) ** 2)
    sst = sum((y - np.mean(y)) ** 2)
    return (1 - (sse / sst))

rmse(y_test, y_hat_test)
r_squared(y_test, y_hat_test)   
