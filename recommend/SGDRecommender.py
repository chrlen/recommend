import recommend as rcm
import numpy as np
import scipy as scp

class SGDRecommender(rcm.Recommender):
    def __init__(self,
                 regularisation_factor,
                 **kwargs):
        rcm.Recommender.__init__(self, "sgd", **kwargs)
        self.converged = False
        self.regularistation_factor = regularisation_factor
        self.max_it = kwargs.get('maxit', 10)


    def predict(self, data):
        result = np.repeat(0, data.shape[0] ).astype(np.int)
        return result

    def fit(self, data):
        self.iterations = 0
        while not self.converged:
            print(self.iterations)
            if self.iterations >= self.max_it:
                self.converged = True
            self.iterations +=1


if __name__ == '__main__':
    import os
    train = np.genfromtxt(os.path.join("../", "data", "train.csv"), delimiter=",", dtype=np.int)
    sgd = rcm.SGDRecommender(
        learning_rate= 0.005,
        regularisation_factor=0.001,
        maxit=10
    )
    err = rcm.strong_cross_validation(sgd, train, 10, err=rcm.rmse)
    print('Error: ',err)

