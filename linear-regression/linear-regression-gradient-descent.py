import math
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class LinearRegression:

    def __init__(self, method='minimize_cost'):
        '''
        input:
            method: str, one of 'minimize_cost' or 'epochs'
        '''
        self.method = method
        pass

    def fit(self, X_train, y_train, epochs=100, batch_size=1, lr=0.1):
        """
        fit linear model to the data
        input:
            X_train: matrix of dimension (#training-observations, #features)
            y_train: vector of dimension (#observation, 1)
        """

        dimension_error = "Error! X and y must have same number of observations.\n X has {} observations.\n y has {} observations.".format(X_train.shape[0], len(y_train))
        assert X_train.shape[0] == len(y_train), dimension_error

        self.X_train = X_train
        self.y_train = y_train.reshape((self.X_train.shape[0], 1))

        self.num_train_observations = self.X_train.shape[0]
        self.num_predictors = self.X_train.shape[1]

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # standardize X
        self.X_train, self.X_mu, self.X_sigma = self.standardize_data(self.X_train)
        self.y_train, self.y_mu, self.y_sigma = self.standardize_data(self.y_train)

        # initialize weights with dimension = (#features + 1, 1); +1 for intercept (w_0)
        self.weights = np.random.normal(0, 1, (self.num_predictors + 1)
                                        ).reshape(self.num_predictors + 1, 1)

        # stack a vector with ones to the left of the first column in X_train
        ones = np.ones(self.num_train_observations).reshape((self.num_train_observations, 1))
        self.X_train = np.concatenate((ones, self.X_train), axis=1)

        # optimize weights
        if self.method == 'epochs':
            # run stochastic gradient descent for number of specific number of epochs
            for i in range(self.epochs):
                self.SGD(self.X_train, self.y_train, self.batch_size)
                pass
            pass
        elif self.method == 'minimize_cost':
            # run stochastic gradient descent until either cost stops decreasing or number of iterations reaches its limit

            MAX_ITER = 1000   # maximum iterations after which algorithm stops running
            tolerance = 0.01  # algorithm stops once cost improvement in cost is less than tolerance
            cost_improvement = math.inf  # initial improvement

            iterations = 0
            while (abs(cost_improvement) > tolerance) or (iterations < MAX_ITER):

                # initial cost
                cost_initial = self.get_cost(self.X_train, self.y_train, self.weights)

                # run stochastic gradient descent
                self.SGD(self.X_train, self.y_train, self.batch_size)

                # final cost
                cost_final = self.get_cost(self.X_train, self.y_train, self.weights)

                # improvement
                cost_improvement = cost_final - cost_initial

                iterations += 1
                pass

            pass

        else:
            ValueError("Please pass either of the following methods: \n 'minimize_cost' \n 'epochs' ")
            pass

        # store intercept and weights
        self.intercept = np.array([self.weights[0]])
        self.coefficients = self.weights[1:]
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

        num_test_observations = X_test.shape[0]

        # standardize X using X_train's mu and sigma
        X_test, _, _ = self.standardize_data(X_test, self.X_mu, self.X_sigma)

        # stack a vector with ones to the left of the first column in X_test
        ones = np.ones(num_test_observations).reshape((num_test_observations, 1))
        X_test = np.concatenate((ones, X_test), axis=1)

        # make prediction using weights: y_hat = W*x
        y_hat_test = np.matmul(X_test, self.weights)

        # undo standardize y using y_train's mu and sigma
        y_hat_test = self.inverse_standardize(y_hat_test, self.y_mu, self.y_sigma)

        # flatten array
        y_hat_test = y_hat_test.ravel()

        return y_hat_test

    def SGD(self, X, y, batch_size=32):
        ''' run stochastic gradient over training data '''

        num_observations = X.shape[0]
        batch_start_index = 0

        while (batch_start_index <= num_observations):
            if (batch_start_index + batch_size) > num_observations:
                # last batch
                X_batch = X[batch_start_index:batch_start_index + num_observations - 1, :]
                y_batch = y[batch_start_index:batch_start_index + num_observations - 1]

                # calculate gradient: dimension(dLoss) = (num_features + 1, 1)
                dLoss = self.gradient(X_batch, y_batch, self.weights)

                # update weights
                assert dLoss.shape == self.weights.shape
                self.weights = self.weights - (self.lr * dLoss)
                pass
            else:
                # select batch from training data
                X_batch = X[batch_start_index:batch_start_index + batch_size - 1, :]
                y_batch = y[batch_start_index:batch_start_index + batch_size - 1]

                # calculate gradient: dimension(dLoss) = (num_features + 1, 1)
                dLoss = self.gradient(X_batch, y_batch, self.weights)

                # update weights
                assert dLoss.shape == self.weights.shape
                self.weights = self.weights - (self.lr * dLoss)
                pass

            batch_start_index += batch_size
            pass
        pass

    def gradient(self, X, y, weights):
        """ return gradient of the loss function """

        num_observations = X.shape[0]
        y_hat = np.matmul(X, weights)

        total_dLoss = np.zeros((X.shape[1], 1))
        for i in range(num_observations):
            dLoss = (-2/num_observations) * (float(y[i]) - float(y_hat[i])) * X[i, :]
            total_dLoss += dLoss.reshape(X.shape[1], 1)

        return total_dLoss

    def get_cost(self, X, y, weights):
        """ return MSE """
        num_observations = X.shape[0]
        y_hat = np.matmul(X, weights)

        return np.sum((y - y_hat) ** 2) / num_observations

    def standardize_data(self, M, mu_vector=None, sigma_vector=None):
        """ calculate z-score of all columns of data """
        if mu_vector == None:
            mu_vector = np.apply_along_axis(np.mean, 0, M)

        if sigma_vector == None:
            sigma_vector = np.apply_along_axis(np.std, 0, M)

        Z = (M - mu_vector) / sigma_vector

        return Z, mu_vector, sigma_vector

    def inverse_standardize(self, Z, mu, sigma):
        """ change z-scores back to original scale """
        return (Z * sigma) + mu

    pass


# MAKE DATA

# height (cm)
X = np.array([140, 190, 165, 140, 170, 180, 175, 173, 155, 162]).reshape((10, 1))

# weight (kg)
y = np.array([40,  90,  70,  45, 100,  70,  65,  75,  55,  58])

# plot data
sns.scatterplot(X[:, 0], y)


# BUILD MODEL

# divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# fit model
model = LinearRegression(method='epochs')
model.fit(X_train, y_train, epochs=1000)
y_hat_train = (model.coefficients * X_train) + model.intercept
model.coefficients
model.intercept

# regression line
plt.scatter(X_train, y_train)
plt.plot([min(X_train), max(X_train)], [min(y_hat_train), max(y_hat_train)], color='red')
plt.show()

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
