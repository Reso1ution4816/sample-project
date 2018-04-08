import numpy as np


class MyLogisticRegression:
    def __init__(self):
        self.w = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_gradient(x):
        return sigmoid(x) * (1 - sigmoid(x))

    @staticmethod
    def ones_augment_to_left(X):
        X = np.array(X)
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])

    @staticmethod
    def logistic_gradient_descent(X, y, n_iters=10000, alpha=0.05, weight=None):
        w = weight
        if w is None:
            w = np.random.rand(X.shape[1])
            w = np.ones(X.shape[1])
        pass

        ###### write your code below ######
        for i in range(1, n_iters + 1):
            y_pred = MyLogisticRegression.sigmoid(X.dot(w))
            loss = y_pred - y

            grad = MyLogisticRegression.sigmoid_gradient(loss.dot(X) / X.shape[0])
            w = w - alpha * grad  # update

            if i % (n_iters // 10) == 0:
                print('iter:%d \ttraining MSE=%.3f \tMAE=%.3f \tr^2=%.3f' % (
                    i,
                    np.linalg.norm(loss),
                    np.linalg.norm(loss, ord=1),
                    r2_score(y, y_pred)
                )
                      )

        ###### write your code above ######
        return w

    def fit(self, X_train, y_train, **kwargs):
        X = self.ones_augment_to_left(X_train)
        y = np.array(y_train)
        self.w = self.logistic_gradient_descent(X, y, **kwargs)

        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        augX_test = self.ones_augment_to_left(X_test)
        return augX_test.dot(self.w)


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X = X[y <= 1]
y = y[y <= 1]
print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

mlr = MyLogisticRegression()

mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)
accuracy_score(y_test, np.round(y_pred))
