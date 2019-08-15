import recommend as rcm
import numpy as np
import scipy as scp
import scipy.sparse as sps
import scipy.sparse.linalg as spsl


class UnconstrainedMFRecommender(rcm.Recommender):
    """  """
    def __init__(self, **kwargs):
        rcm.Recommender.__init__(self, "umf", **kwargs)
        self.converged = False
        self.factors = kwargs.get('k', 10)
        self.max_it = kwargs.get('max_it', np.inf)
        self.alpha = kwargs.get('alpha', 0.001)
        self.verbose = kwargs.get('verbose',True)

    def predict_single(self, user, item):
        if user >= self.num_users:
            return self.average
        if item >= self.num_items:
            return self.average
        return self.reconstruction[user,item]
        #return self.U[user, :].dot(self.V[:, item]).reshape(-1)


    def predict(self, data):
        res = np.array([ self.predict_single(data[i,0],data[i,1]) for i in range(data.shape[0]) ]).reshape(-1)
        return res

    def fit(self, data):
        self.sparse_data = rcm.to_sparse_matrix(data)
        self.average = np.mean(data[:,2])
        self.num_users = np.max(data[:, 0]) + 1
        self.num_items = np.max(data[:, 1]) + 1

        self.mask = self.sparse_data != 0
        self.mask = self.mask.todense()
        self.mask = np.logical_not(self.mask)

        self.U = np.random.rand(
                self.num_users,
                self.factors
        )

        self.V = np.random.rand(
                self.num_items,
                self.factors
        )

        self.last_e = 0
        self.iterations = 0
        while not self.converged:
            if self.verbose:
                print("-------- {} -------".format(self.iterations))
            R_hat = self.U.dot(self.V.T)

            E = self.sparse_data - R_hat
            E[self.mask] = 0
            E = sps.csc_matrix(E)

            self.this_e = spsl.norm(E)
            print(self.this_e - self.last_e)

            temp_U = self.U
            self.U = self.U + self.alpha * E.dot(self.V)
            self.V = self.V + self.alpha * E.T.dot(temp_U)

            if self.iterations >= self.max_it:
                self.converged = True

            if self.iterations > 0:
                diff = self.last_e - self.this_e
                if diff < 0.001:
                    self.converged = True
            self.last_e = self.this_e
            self.iterations +=1
        self.reconstruction = self.U.dot(self.V.T)


if __name__ == '__main__':
    import os
    train = np.genfromtxt(os.path.join("../", "data", "train.csv"), delimiter=",", dtype=np.int)
    sgd = rcm.UnconstrainedMFRecommender(
        **{
            'max_it': np.inf
        }
    )
    err = rcm.cross_validation(sgd, train, 5, err=rcm.rmse)
    print('Error: ', err)

