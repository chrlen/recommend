from recommend import Recommender

import numpy as np

class RandomRecommender(Recommender):
    def __init__(self, **kwargs):
        Recommender.__init__(self, "rnd", **kwargs)

    def predict(self, data):
        result = np.random.randint(self.min, self.max +1 , data.shape[0])
        return result

    def fit(self, data):
        self.min = np.min(data[:, 2])
        self.max = np.max(data[:, 2])