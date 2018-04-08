import numpy as np
from matplotlib import pyplot as plt


class TreeNode:
    def __init__(self, value, left_child=None, right_child=None):
        self.value = value
        self.left = None
        self.right = None

    def traverase_tree(self):
        print(self.value)
        if self.left is not None:
            self.left.traverase_tree()
        if self.right is not None:
            self.right.traverase_tree()

    def add_left(self, value):
        tree = TreeNode(value)
        if self.left is None:
            self.left = tree
        else:
            tree.left = self.left
            self.left = tree

    def add_right(self, value):
        tree = TreeNode(value)
        if self.right is None:
            self.right = tree
        else:
            tree.right = self.right
            self.right = tree


def l2norm(u):
    """
    L2 Norm for a vector or a matrix
    :param u:
    :return:
    """
    return np.sqrt(np.sum(u ** 2))


def euclidean_dist(u, v):
    """
    Euclidean distance between two vectors
    :param u:
    :param v:
    :return:
    """
    return l2norm(u - v)


def most_frequent(ar):
    """
    Find the most frequent element in the array
    :param ar: an sequence
    :return: most frequent element in ar
    """
    u, indices = np.unique(ar, return_inverse=True)
    return u[np.argmax(np.bincount(indices))]


def sigmoid(x):
    """
    Sigmoid function
    :param x: scalar, vector or matrix
    :return:
    """
    return 1 / (1 + np.exp(-x))


def ones_augment_to_left(X):
    """
    augment one column of 1s to the left of X
    :param X: the matrix to be augmented
    :return: augmented X
    """
    X = np.array(X)
    ones = np.ones(X.shape[0])
    return np.column_stack([ones, X])


class KNN:
    def __init__(self, k=3):
        """
        K-Nearest Neighbour
        :param k: number of nearest neighbours
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Build KNN model
        :param X_train: training sample matrix
        :param y_train: training label vector
        :return: KNN model
        """
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        """
        Make predictions
        :param X_test: the testing sample matrix
        :return: the predicted labels as a vector
        """
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


class MyLinearRegression:
    """

    """

    def __init__(self, n_iters=10000, alpha=0.05, weight=None, method='closed form'):
        self.w = weight
        self.n_iters = n_iters
        self.alpha = alpha
        self.method = method

    def gradient_descent(self, X, y):
        w = self.w
        if w is None:
            w = np.ones(X.shape[1])

        for i in range(self.n_iters):
            y_pred = X.dot(w)
            loss = y_pred - y

            grad = loss.dot(X) / X.shape[0]
            w = w - self.alpha * grad  # update
        return w

    @staticmethod
    def closed_form(X, y):
        product = np.dot(X.T, X)
        theInverse = np.linalg.inv(product)
        return np.dot(np.dot(theInverse, X.T), y)

    def fit(self, X_train, y_train):
        X = ones_augment_to_left(X_train)
        y = np.array(y_train)

        if self.method == 'closed form':
            self.w = self.closed_form(X, y)
        elif self.method == 'gradient descent':
            self.w = self.gradient_descent(X, y)
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        augX_test = ones_augment_to_left(X_test)

        return augX_test.dot(self.w)


class MyLogisticRegression:
    """

    """

    def __init__(self, n_iters=10000, alpha=0.05, weight=None):
        self.w = weight
        self.n_iters = n_iters
        self.alpha = alpha

    def logistic_gradient_descent(self, X, y):
        w = self.w
        if w is None:
            w = np.ones(X.shape[1])
        pass

        for i in range(self.n_iters):
            scores = X.dot(w)
            y_pred = sigmoid(scores)
            error = y_pred - y
            grad = error.dot(X) / X.shape[0]

            w = w - self.alpha * grad  # update

        return w

    def fit(self, X_train, y_train):
        X = ones_augment_to_left(X_train)
        y = np.array(y_train)
        self.w = self.logistic_gradient_descent(X, y)

        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        augX_test = ones_augment_to_left(X_test)

        return sigmoid(augX_test.dot(self.w))


def plot_bounds2D(X, y, clf):
    # create all pari-wise combinations of
    # x0's possible values and x1's possible values
    #
    # read here for more details:
    # https://stackoverflow.com/questions/36013063/what-is-purpose-of-meshgrid-in-python

    d0_possible_values = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1)
    d1_possible_values = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1)

    d0, d1 = np.meshgrid(d0_possible_values, d1_possible_values)

    # create data set with d0 and d1 as two columns
    dots = np.c_[d0.ravel(), d1.ravel()]

    # num. rows = 1, num. cols = 1, plotting 1st subplot
    plt.subplot(1, 1, 1)

    # convert possible string labels to numerical label for plotting
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    plt.subplots_adjust(wspace=0.6, hspace=0.4)
    clf.fit(X[:, [0, 1]], y)
    dots_predicted = clf.predict(dots)

    # Put the result into a color plot
    dots_predicted = dots_predicted.reshape(d0.shape)

    plt.contourf(d0, d1, dots_predicted, cmap=plt.cm.coolwarm, alpha=0.8)
    #     plt.pcolormesh(d0, d1, coloured_dots, cmap=plt.cm.coolwarm)

    # Plot also the training points
    plt.xlim(d0.min(), d0.max())
    plt.ylim(d1.min(), d1.max())

    # plot the line, the points, and the nearest vectors to the plane
    plt.xlabel('X[:, 0]')
    plt.ylabel('X[:, 1]')

    for yy in np.unique(y):
        plt.scatter(x=X[:, 0][y == yy], y=X[:, 1][y == yy], label='y_train==' + str(yy))
    plt.legend()


if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.datasets import load_iris
    from matplotlib import pyplot as plt

    X, y = load_iris(return_X_y=True)
    # plot_bounds2D(X, y, MyLogisticRegression())
    # plt.show()
    print(MyLogisticRegression().fit(X, y).predict(y))
