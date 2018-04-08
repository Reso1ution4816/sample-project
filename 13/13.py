if False:
    import numpy as np


    class MyLinearRegression:
        def __init__(self):
            self.w = None

        @staticmethod
        def ones_augment_to_left(X):
            X = np.array(X)
            ones = np.ones(X.shape[0])
            return np.column_stack([ones, X])

        def fit(self, X_train, y_train):
            X = self.ones_augment_to_left(X_train)
            y = np.array(y_train)

            # write your code here #

            return self

        def predict(self, X_test):
            X_test = np.array(X_test)

            # write your code here #

            return  # write your code here #

###############################################

import numpy as np


class MyLinearRegression:
    def __init__(self):
        self.w = None

    @staticmethod
    def ones_augment_to_left(X):
        X = np.array(X)
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])

    def fit(self, X_train, y_train):
        X = self.ones_augment_to_left(X_train)
        y = np.array(y_train)

        product = np.dot(X.T, X)
        theInverse = np.linalg.inv(product)
        self.w = np.dot(np.dot(theInverse, X.T), y)
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        return self.ones_augment_to_left(X_test).dot(self.w)

    def predict_alternative(self, X_test):
        predictions = []
        for i in X_test:
            components = self.w[1:] * i
            predictions.append(sum(components) + self.w[0])
        return predictions


###################################################

# 测试
import numpy as np

mlr = MyLinearRegression()

X = [[1, 5], [3, 2], [6, 1]]
y = [2, 3, 4]
y_pred = mlr.fit(X, y).predict(X)
print('Am I correct? \n', np.isclose(y, y_pred).all())

'''
如果实现正确, 会输出
Am I correct? 
 True
'''

#######################################################

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the diabetes dataset

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create linear regression object
mlr2 = MyLinearRegression()

# Train the model using the training sets
mlr2.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = mlr2.predict(X_test)

# The coefficients
w = mlr2.w
print('w: \n', w)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score (r squared, r^2): %.2f' % r2_score(y_test, y_pred))

#############################################


import matplotlib.pyplot as plt

d = 1

y_pred3 = MyLinearRegression().fit(X_train[:, d], y_train).predict(X_test[:, d])

# Plot outputs
plt.scatter(X_train[:, d], y_train, color='blue', alpha=0.8, label='Train Data')
plt.scatter(X_test[:, d], y_test, color='red', alpha=0.8, label='Test Data')
plt.plot(X_test[:, d], y_pred3, color='blue', linewidth=3, label='Fitted Line')
plt.legend()
###################################################

import numpy as np


def batch_gradient_descent(alpha, x, y, n_iters=100):
    m = x.shape[0]  # number of samples
    theta = np.ones(2)
    x_transpose = x.transpose()
    for iter in range(0, n_iters):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        print("iter %s | J: %.3f" % (iter, J))
        gradient = np.dot(x_transpose, loss) / m
        theta = theta - alpha * gradient  # update
    return theta


###################################################

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the diabetes dataset

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
w_1_to_n = regr.coef_
w0 = regr.intercept_
print('w_0: \n', w0)
print('w_1 to w_n: \n', w_1_to_n)

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score (r squared, r^2): %.2f' % r2_score(y_test, y_pred))
