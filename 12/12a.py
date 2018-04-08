import numpy as np
from sklearn.metrics import accuracy_score


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


################### KNN ###########################

import numpy as np


def euclidean_dist(u, v):
    return np.sqrt(np.sum(u - v) ** 2)


def most_frequent(ar):
    u, indices = np.unique(ar, return_inverse=True)
    return u[np.argmax(np.bincount(indices))]


def knn_predict(X_train, X_test, y_train, k=3):
    predictions = []
    for i, unknown_sample in enumerate(X_test):
        neighbours = []
        for j, observed_sample in enumerate(X_train):
            distance = euclidean_dist(unknown_sample, observed_sample)
            neighbours.append([distance, y_train[j]])

        neighbours = np.array(neighbours)
        knn = neighbours[neighbours[:, 0].argsort()][:k]
        predictions.append(most_frequent(knn[:, 1]))
    return np.ravel(predictions)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


##################### End KNN ##########################



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import train_test_split
iris = pd.read_csv('iris.csv', delimiter=',')

X = iris[['Sepal length', 'Sepal width',
          'Petal length', 'Petal width']].as_matrix()
y = iris['Species'].as_matrix()


# for k in range(1, 11):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.33, random_state=42)
#     predictions = knn_predict(X_train, X_test, y_train, k=k)
#     print(k, accuracy(y_test, predictions))
#

def experiment(X, y, k, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state)

    y_pred = knn_predict(X_train, X_test, y_train, k=k)

    return accuracy(y_test, y_pred)


if __name__ == '__main__':
    import time
    from sklearn.externals.joblib import Parallel
    from sklearn.externals.joblib import delayed

    start_t = time.perf_counter()

    all_scores = []

    # k 取 1 到 10
    ks = np.arange(1, 11)

    # 8个 random_states, 8次实验
    randsts = np.arange(8)
    print('random states =', randsts)

    kpscores = []

    for k in ks:
        pscores = Parallel(n_jobs=-1)(
            delayed(experiment)(X, y, k, random_state)
            for random_state in randsts)
        kpscores.append(pscores)

    duration = time.perf_counter() - start_t
    print('time elapsed: %s seconds' % duration)
