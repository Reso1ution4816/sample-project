import numpy as np


def l2norm(u):
    return np.sqrt(np.sum(u) ** 2)


def euclidean_dist(u, v):
    return l2norm(u - v)


def most_frequent(ar):
    u, indices = np.unique(ar, return_inverse=True)
    return u[np.argmax(np.bincount(indices))]


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        predictions = []
        for i, unknown_sample in enumerate(X_test):
            neighbours = []
            for j, observed_sample in enumerate(self.X_train):
                distance = euclidean_dist(unknown_sample, observed_sample)
                neighbours.append([distance, self.y_train[j]])
            neighbours = np.array(neighbours)
            knn = neighbours[neighbours[:, 0].argsort()][:self.k]
            predictions.append(most_frequent(knn[:, 1]))
        return np.ravel(predictions)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import train_test_split
iris = pd.read_csv('iris.csv', delimiter=',')

X = iris[['Sepal length', 'Sepal width',
          'Petal length', 'Petal width']].as_matrix()
y = iris['Species'].as_matrix()

# 分割数据集, 67%的用于训练, 33%的用于测试
# X_train, X_test = 训练样本, 测试样本
# y_train, y_test = 训练目标, 测试目标
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

knn = KNN(k=3)

# 预测测试样本生成的目标: y_pred
y_pred = knn.fit(X_train, y_train).predict(X_test)

print('accuracy=%s' % accuracy(y_test, y_pred))
