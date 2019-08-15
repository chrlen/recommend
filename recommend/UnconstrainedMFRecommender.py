import recommend as rcm
import numpy as np
import scipy as scp

class UnconstrainedMFRecommender(rcm.Recommender):
    """  """
    def __init__(self, **kwargs):
        rcm.Recommender.__init__(self, "umf", **kwargs)
        self.converged = False
        self.factors = kwargs.get('k', 10)
        self.max_it = kwargs.get('max_it', 5)
        self.alpha = kwargs.get('alpha', 0.001)
        self.verbose = kwargs.get('verbose',True)


    def predict(self, data):
        result = np.repeat(0, data.shape[0] ).astype(np.int)
        return result

    def fit(self, data):
        num_users = np.max(data[:, 0]) +1
        # M
        num_items = np.max(data[:, 1]) +1

        self.sparse_data = rcm.to_sparse_matrix(data)

        self.U = np.zeros(
            shape=(
                num_users,
                self.factors
            )
        )

        self.V = np.zeros(
            shape=(
                self.factors,
                num_items
            )
        )

        self.iterations = 0
        while not self.converged:
            if self.verbose:
                print("-------- {} -------".format(self.iterations))
            R_hat = self.U.dot(self.V)
            E = self.sparse_data - self.U.dot(self.V)
            self.U = self.U + self.alpha * E.dot(self.V.T)

            temp_2 = self.alpha * E.T.dot(self.U).T
            self.V = self.V + temp_2

            if self.iterations >= self.max_it:
                self.converged = True
            self.iterations +=1


if __name__ == '__main__':
    import os
    train = np.genfromtxt(os.path.join("../", "data", "train.csv"), delimiter=",", dtype=np.int)
    sgd = rcm.UnconstrainedMFRecommender()
    err = rcm.cross_validation(sgd, train, 10, err=rcm.rmse)
    print('Error: ',err)

