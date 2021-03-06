import numpy as np
def euclidean_dist():


class KNN():
    """ K Nearest Neighbors classifier.

    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.
    """

    def __init__(self, k=5):
        self.k = k

    # Do a majority vote among the neighbors
    def _majority_vote(self, neighbors, classes):
        max_count = 0
        most_common = None
        # Count class occurences among neighbors
        for c in np.unique(classes):
            # Count number of neighbors with class c
            count = len(neighbors[neighbors[:, 1] == c])
            if count > max_count:
                max_count = count
                most_common = c
        return most_common

    def predict(self, X_test, X_train, y_train):
        classes = np.unique(y_train)
        y_pred = []
        # Determine the class of each sample
        for test_sample in X_test:
            neighbors = []
            # Calculate the distance form each observed sample to the
            # sample we wish to predict
            for j, observed_sample in enumerate(X_train):
                distance = euclidean_dist(test_sample, observed_sample)
                label = y_train[j]
                # Add neighbor information
                neighbors.append([distance, label])
            neighbors = np.array(neighbors)
            # Sort the list of observed samples from lowest to highest distance
            # and select the k first
            k_nearest_neighbors = neighbors[neighbors[:, 0].argsort()][:self.k]
            # Do a majority vote among the k neighbors and set prediction as the
            # class receing the most votes
            label = self._majority_vote(k_nearest_neighbors, classes)
            y_pred.append(label)
        return np.array(y_pred)
