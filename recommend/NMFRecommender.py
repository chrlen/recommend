import recommend as rcm
import numpy as np
import numpy.linalg as npl


class NMFRecommender(rcm.Recommender):
    def __init__(self, **kwargs):
        rcm.Recommender.__init__(self, 'nmf', **kwargs)
        self.maxit = kwargs.get('maxit', 10)
        self.learning_rate = kwargs.get('learning_rate',0.005)


        self.iterations = 0

    def predict(self, data):
        return 0

    def fit(self, data):
        self.p_u = rcm.d_mse(self.p_u, self.learning_rate)
        self.q_i = rcm.d_mse(self.q_i, self.learning_rate)

        self.p_u = rcm.d_mse(self.p_u, self.learning_rate)
        self.q_i = rcm.d_mse(self.q_i, self.learning_rate)

        while self.iterations < self.maxit:
            self.iterations += 1


if __name__ == '__main__':
    import os
    import recommend as rcm
    import numpy as np
    train = np.genfromtxt(os.path.join("../","data", "train.csv"), delimiter=",", dtype=np.int)
    qual = np.genfromtxt(os.path.join("../","data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)

    sgd = rcm.SVDRecommender(
        maxit=2
    )
    sgd.fit(train)
    p = sgd.predict(qual)
    #err = rcm.cross_validation(sgd, train, 10, err=rcm.rmse)
    print('Error: ', err)

