import recommend as rcm
import vae
import numpy as np

class VAERecommender(rcm.Recommender):
    def __init__(self, name, **kwargs):
        rcm.Recommender.__init__(self, name, **kwargs)
        self.vae = vae.VAE(**kwargs)

    def predict(self, data):
        result = np.repeat(self.prediction, data.shape[0] )
        return result

    def fit(self, data):
        self.prediction = np.mean(data[:, 2])