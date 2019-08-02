from recommend import Recommender

import numpy as np

class MeanRecommender(Recommender):
    def __init__(self, **kwargs):
        Recommender.__init__(self, "mean", **kwargs)

    def predict(self, data):
        result = np.repeat(self.prediction, data.shape[0] ).astype(np.int)
        return result

    def fit(self, data):
        self.prediction = np.median(data[:, 2]).astype(np.int)