from recommend import Recommender
from sklearn.linear_model import LogisticRegression

import numpy as np
from recommend import Recommender
import scipy

class ClassRecommender(Recommender):
    def __init__(self, name, **kwargs):
        Recommender.__init__(self, name, **kwargs)

    def predict(self, data):
        pred = self.lg.predict(data)
        return pred

    def fit(self, data):
        self.lg = LogisticRegression(
            random_state=0,
            solver='newton-cg',
            multi_class='multinomial'
        ).fit(data[:, :-1], data[:, -1])

