import recommend as rcm
import numpy as np
import scipy as scp

class CPMFRecommender(rcm.Recommender):
    def __init__(self,
                 regularisation_factor,
                 **kwargs):
        rcm.Recommender.__init__(self, "sgd", **kwargs)
        self.converged = False
        self.regularistation_factor = regularisation_factor
        self.max_it = kwargs.get('maxit', 10)
        self.rank = kwargs.get('rank',30)


    def predict(self, data):
        result = np.repeat(0, data.shape[0] ).astype(np.int)
        return result

    def fit(self, data):
        # N
        num_users = np.max(data[:,0])
        # M
        num_items = np.max(data[:,1])
        sparse_data = rcm.to_sparse_matrix(data)

        self.sd = 0
        self.sd_i = 0
        self.sd_u = 0

        #Matrix of size D X N
        self.U = np.zeros((self.rank,num_users))

        #Matrix of size D X M
        self.V = np.zeros((self.rank,num_items))

        #Latent similarity constraint matrix DxM
        self.W = np.zeros((self.rank,num_items))
        
        # Obeserved-Indicator I N X M
        self.I = sparse_data != 0 

        



        self.iterations = 0
        while not self.converged:
            print(self.iterations)
            if self.iterations >= self.max_it:
                self.converged = True
            self.iterations +=1


if __name__ == '__main__':
    import os
    train = np.genfromtxt(os.path.join("../", "data", "train.csv"), delimiter=",", dtype=np.int)
    sgd = rcm.SVDPPRecommender(
        learning_rate= 0.005,
        regularisation_factor=0.001,
        maxit=10
    )
    err = rcm.strong_cross_validation(sgd, train, 10, err=rcm.rmse)
    print('Error: ',err)

