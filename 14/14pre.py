import numpy as np
from sklearn.metrics import r2_score

class MyLinearRegression:
    def __init__(self):
        self.w = None

    @staticmethod
    def ones_augment_to_left(X):
        X = np.array(X)
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])
    
    @staticmethod
    def gradient_descent(X, y, n_iters=10000, alpha=0.05, weight=None):
        w = weight
        if w is None:
            w = np.random.rand(X.shape[1])
        pass
        
        ###### write your code below ######
        
        
        
        ###### write your code above ######
        
        return w
    
    @staticmethod
    def closed_form(X ,y):
        product = np.dot(X.T, X)
        theInverse = np.linalg.inv(product)
        return np.dot(np.dot(theInverse, X.T), y)
    
    
    def fit(self, X_train, y_train, method='closed form', **kwargs):
        X = self.ones_augment_to_left(X_train)
        y = np.array(y_train)
        
        if method=='closed form':
            self.w = self.closed_form(X ,y)
        elif method == 'gradient descent':
            self.w = self.gradient_descent(X, y, **kwargs)

        return self

    
    def predict(self, X_test):
        X_test = np.array(X_test)
        augX_test = self.ones_augment_to_left(X_test)
        return augX_test.dot(self.w)
    
# 测试
import numpy as np

mlr = MyLinearRegression()

X = np.array([[1, 5], [3, 2], [6, 1]])
y = np.array([2, 3, 4])
y_pred = mlr.fit(X, y, method='gradient descent', 
                 n_iters=10000, 
                 alpha=0.05).predict(X)
print('fitted w is \t', mlr.w)
print('expected w is \t [ 2.42857143  0.28571429 -0.14285714]')
print('Am I correct? \t', np.isclose(y, y_pred, atol=1e-2).all())
