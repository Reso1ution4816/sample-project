import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def accuracy(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    return (y_pred == y).sum() / y.shape[0]


def test_dump(X, y, save_path):
    if os.path.exists(save_path):
        res = pd.read_excel(save_path)
    else:
        accs = {}

        for max_depth in [10, 20, 50, 100, 200, 500, 1000]:
            accs[max_depth] = []
            for _ in range(50):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.33, random_state=None)

                clf = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
                accs[max_depth].append(accuracy(y_test, clf.predict(X_test)))

        res = pd.DataFrame(accs)
        res.to_excel(save_path)
    return res


def test_dump_pa(X, y, save_path):
    if os.path.exists(save_path):
        res = pd.read_excel(save_path)
    else:
        accs = {}

        for max_depth in [10, 20, 50, 100, 200, 500, 1000]:
            accs[max_depth] = []
            for _ in range(50):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.33, random_state=None)

                clf = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
                accs[max_depth].append(accuracy(y_test, clf.predict(X_test)))

        res = pd.DataFrame(accs)
        res.to_excel(save_path)
    return res
