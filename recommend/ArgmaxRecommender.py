from scipy.stats import moment
import recommend as rcm

from recommend import Recommender

import numpy as np
import scipy
import scipy.stats as spstats


class ArgmaxRecommender(Recommender):
    def __init__(self, name, **kwargs):
        Recommender.__init__(self, name, **kwargs)

    def predict(self, data):
        result = np.zeros(data.shape[0])

        for i in range(data.shape[0]):
            user, item = data[i, 0], data[i, 1]
            if np.all(self.column_histograms[item][0] == 0):
                if np.all(self.row_histograms[item][0] == 0):
                    result[i] = np.random.randint(self.min, self.max +1)
                    print(result[i])
                else:
                    result[i] = np.argmax(self.row_histograms[item][0])
            else:
                result[i] = np.argmax(self.column_histograms[item][0])

        return result

    def fit(self, data):
        sparse_matrix = rcm.to_sparse_matrix(data)
        self.bins = len(set(sparse_matrix.data))
        self.column_histograms = [np.histogram(sparse_matrix[:, i].data, bins=self.bins) for i in
                                   range(sparse_matrix.shape[1])]

        self.row_histograms = [np.histogram(sparse_matrix[i, :].data, bins=self.bins) for i in
                               range(sparse_matrix.shape[0])]

        self.min = np.min(data[:, -1])
        self.max = np.max(data[:, -1])
