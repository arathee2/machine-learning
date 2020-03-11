import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

class LogisticRegression:

    def __init__(self):
        pass

    def fit(self, X, y, add_bias=True, epochs=100, batch_size=1, lr=0.1):
        """
        fit logistic regression model to the data
        input:
            X: matrix of dimension (#training-observations, #features)
            y: vector of dimension (#observation, 1)
        """

        #dimension_error = "Error! X and y must have same number of observations.\n X has {} observations.\n y has {} observations.".format(X.shape[0], len(y))
        assert X.shape[0] == len(y)#, dimension_error

        self.X = X
        self.y = y.reshape((self.X.shape[0], 1))

        self.num_train_observations = self.X.shape[0]
        self.num_predictors = self.X.shape[1]

        self.add_bias = add_bias
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        # standardize X
        self.X = self.standardize_data(self.X, training=True)
        
        if self.add_bias:
            # stack a vector with ones to the left of the first column in X
            ones = np.ones(self.num_train_observations)
            ones = ones.reshape((self.num_train_observations, 1))
            self.X = np.concatenate((ones, self.X), axis=1)
            pass
        
        # initialize weights with dimension = (#columns in X, 1)
        self.num_predictors = self.X.shape[1]
        self.weights = self.initialize_weights(nrow=self.num_predictors, ncol=1)
        
        # initial cost
        self.training_cost = self.get_cost(self.X, self.y, self.weights)
        
        # optimize weights
        for i in range(self.epochs):
            self.SGD(self.X, self.y, self.batch_size)
            
            self.training_cost.append(self.get_cost(self.X, self.y, self.weights))
            pass

        # store intercept and weights
        self.intercept = np.array([self.weights[0]])
        self.coefficients = self.weights[1:]
        pass

    def predict(self, X_test, get_prob=True):
        """
        input:
            X_test: matrix of dimension (#testing_observations, self.num_predictors)
        output:
            y_hat_test: vector of dimension (#testing_observations, 1)
        """

        #dimension_error = "Error! Test data must have same number of features as train data.\n Features in trianing set = {}.\n Features in test set = {}.".format(self.X.shape[1]-1, X_test.shape[1])
        assert X_test.shape[1] == self.X.shape[1] - 1 #, dimension_error

        num_test_observations = X_test.shape[0]

        # standardize X using X_train's mu and sigma
        X_test = self.standardize_data(X_test, training=False)

        if self.add_bias:
            # stack a vector with ones to the left of the first column in X_test
            ones = np.ones(num_test_observations)
            ones = ones.reshape((num_test_observations, 1))
            X_test = np.concatenate((ones, X_test), axis=1)
            pass

        # compute log odds: z = W*x
        z = np.matmul(X_test, self.weights)

        # compute probability P(y=1 | x)
        y_hat_test = self.sigmoid(z)

        # flatten array
        y_hat_test = y_hat_test.ravel()
        
        if get_prob == False:
            return np.apply_along_axis(lambda prob: np.round(prob), 0, y_hat_test).astype(int)
        else:
            return y_hat_test

    
    def SGD(self, X, y, batch_size=32):
        ''' run stochastic gradient over training data '''

        num_observations = X.shape[0]
        num_batches = math.ceil(num_observations / batch_size)
        
        for batch_id in range(num_batches):
            X_batch, y_batch = self.get_batch(X, y, batch_id, batch_size)
            
            # compute gradient: dL/dw
            gradient_batch = self.get_gradient(X_batch, y_batch, self.weights)
            assert gradient_batch.shape == self.weights.shape
                
            # update weights
            self.weights -= self.lr * gradient_batch
            pass
        
        pass

    
    def get_gradient(self, X, y, weights):
        """ return gradient of the loss function """

        num_observations = X.shape[0]
        
        # compute z
        z = np.matmul(X, weights)
        
        # compute probability P(y=1 | x)
        p = self.sigmoid(z)
        
        # gradient corresponding to all observations
        total_gradient = np.zeros((X.shape[1], 1))
        
        # compute cumulative 
        for i in range(num_observations):
            # gradient corresponding to ith obervation
            gradient_i = (y[i] - p[i]) * X[i, :]
            total_gradient += gradient_i.reshape(X.shape[1], 1)
            pass
        
        total_gradient *= -1
        
        return total_gradient
    
    
    def get_cost(self, X, y, weights):
        
        num_observations = X.shape[0]
        
        # compute z
        z = np.matmul(X, weights)
        
        # compute probability P(y=1 | x)
        p = self.sigmoid(z)
        
        # cost corresponding to all observations
        total_cost = 0
        for i in range(num_observations):
            # cost corresponding to ith obervation
            cost_i = (y[i] * math.log(p[i])) + ((1 - y[i]) * (math.log(1 - p[i])))
            total_cost += cost_i
            pass
        
        total_cost = float(-1 * total_cost / num_observations)
        
        return total_cost

    
    def get_batch(self, X, y, batch_id, batch_size):
        num_observations = X.shape[0]
        batch_start_index = batch_id * batch_size
        batch_end_index = batch_start_index + batch_size
        if batch_end_index > num_observations - 1:
            batch_end_index = num_observations
            pass
        
        return X[batch_start_index:batch_end_index, :], y[batch_start_index:batch_end_index]
    
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    
    def initialize_weights(self, nrow, ncol, mu=0, sigma=1):
        return np.random.normal(mu, sigma, (nrow, ncol)).reshape(nrow, ncol)
    
    
    def standardize_data(self, M, training=False):
        """ calculate z-score of all columns of data """
        
        if training:
            self.mu_vector = np.apply_along_axis(np.mean, 0, M)
            self.sigma_vector = np.apply_along_axis(np.std, 0, M)
            z_scores = (M - self.mu_vector) / self.sigma_vector
        else:
            z_scores = (M - self.mu_vector) / self.sigma_vector

        return z_scores

    pass


# MAKE DATA
X, y = make_classification(n_samples=200, n_features=7, n_redundant=2)


# BUILD MODEL

# divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit model
model = LogisticRegression()
model.fit(X_train, y_train, batch_size=8, epochs=200, lr=0.01)
model.coefficients
model.intercept

# make predictions on test set
y_hat_test = model.predict(X_test, get_prob=False)
y_hat_test
y_test


# EVALUATE MODEL

def plot_roc_curve(y_true, y_pred_probs, model_name="Model"):
    """
    Input:
        y_true        : array of true binary labels. Eg: [0, 1, 0, 0, 1].
        y_pred_probs  : array of probability scores. Eg: [0.12, 0.43, 0.26, 0.67, 0.49].
    Output:
        ROC and AUC curve
    """
    # generate zero probabilities
    zero_probs = [0 for _ in range(len(y_true))]

    # calculate scores
    zero_probs_auc = roc_auc_score(y_true, zero_probs)
    y_pred_probs_auc = roc_auc_score(y_true, y_pred_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (zero_probs_auc))
    print('Logistic: ROC AUC=%.3f' % (y_pred_probs_auc))

    # calculate roc curves
    base_fpr, base_tpr, _ = roc_curve(y_true, zero_probs)
    model_fpr, model_tpr, _ = roc_curve(y_true, y_pred_probs)

    # plot the roc curve for the model
    plt.plot(base_fpr, base_tpr, linestyle='--', label='No Skill')
    plt.plot(model_fpr, model_tpr, marker='.', label=model_name)

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # show the legend
    plt.legend()

    # show the plot
    plt.show()
    pass

def print_confusion_matrix(y_true, y_pred):
    """
    Input:
        y_true        : array of true binary labels. Eg: [0, 1, 0, 0, 1].
        y_pred_probs  : array of predicted binary labels. Eg: [0, 1, 0, 1, 1].
    Output:
        confusion matrix
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # rates
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    fnr = fn/(tp+fn)
    tnr = tn/(fp+tn)
    
    # PPV and NPV
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    
    # accuracy
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    
    # f-1 score
    f1_score = 2*(ppv*tpr)/(ppv+tpr)
    
    # print results
    margin_length = 40
    print("Sensitivity: {:.4f}\n".format(tpr), "="*margin_length)
    print("Specificity: {:.4f}\n".format(tnr), "="*margin_length)
    print("Positive Predicted Value: {:.4f}\n".format(ppv), "="*margin_length)
    print("Negative Predicted Value: {:.4f}\n".format(npv), "="*margin_length)
    print("Accuracy: {:.4f}\n".format(accuracy), "="*margin_length)
    
    return confusion_matrix(y_true, y_pred)

plot_roc_curve(y_test, y_hat_test)
print_confusion_matrix(y_test, y_hat_test)


# comparing with sklearn
from sklearn import linear_model
# fit model
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
model.coef_
model.intercept_

# make predictions on test set
y_hat_test = model.predict(X_test)

plot_roc_curve(y_test, y_hat_test)
print_confusion_matrix(y_test, y_hat_test)


# rough
