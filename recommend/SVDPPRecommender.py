import recommend as rcm
import tqdm
import numpy as np
from functools import reduce
import scipy as scp
import multiprocessing as mp
import scipy.spatial.distance as ssd


class SVDPPRecommender(rcm.Recommender):
    def __init__(self,
                 **kwargs):
        rcm.Recommender.__init__(self, "sgd", **kwargs)
        self.converged = False
        self.max_it = kwargs.get('maxit', 10)
        self.k = kwargs.get('k', 30)
        self.factors = kwargs.get("factors", 3)
        self.lambda1 = kwargs.get('lambda1', 0.005)
        self.lambda2 = kwargs.get('lambda2', 100)
        self.lambda3 = kwargs.get('lambda3', 100)
        self.lambda4 = kwargs.get('lambda4', 0.01)
        self.lambda5 = kwargs.get('lambda5', 0.01)
        self.lambda6 = kwargs.get('lambda6', 0.01)
        self.lambda7 = kwargs.get('lambda7', 0.015)
        self.lambda8 = kwargs.get('lambda8', 0.015)

        self.gamma1 = kwargs.get('gamma1', 0.00000015)
        self.gamma2 = kwargs.get('gamma2', 0.00000015)
        self.gamma3 = kwargs.get('gamma2', 0.00000015)

        self.verbose = kwargs.get('verbose', True)

    def N(self, user):
        """ return Implicit preference of user """
        res = set(np.argwhere(self.sparse_data[user, :] != 0)[:, 1])
        return res

    def R(self,user):
        """ return items rated by user """
        return set(np.argwhere(self.sparse_data[user, :] != 0)[:, 1])

    def R_k(self, user, item, k):
        """ R_k = R(u) inter S_k(i) """
        ru = self.R(user)
        sk = self.S_k(user, item, self.k)
        res = ru.intersection(sk)
        return res

    def S_k(self, user, item, k):
        """ Let us denote by S k (i) the set of k items most similar i, as determined by the similarity measure s ij """
        v = np.array(self.s_ij[:, item].todense()).reshape(-1)
        res = np.argsort(v)
        k_highest = set(res[(len(res)-k):])
        return k_highest


    def N_k(self, user, item, k):
        """ N_k = N(u) inter S_k(i) """
        sk = self.S_k(user, item, self.k)
        nu = self.N(user)
        res = sk.intersection(nu)
        return res
    def b_ui(self,user,item):
        return self.average + self.b_u[user]+ + self.b_i[item]

    def predict_single(self, user, item):
        #print(user, item)
        if item >= self.sparse_data.shape[1]:
            return self.average
        if user >= self.sparse_data.shape[0]:
            return self.average
        if not user in self.seen:
            return self.average

        N = self.N(user)
        if len(N) == 0:
            card_N = 0
        else:
            card_N =(1 / np.sqrt(len(N)))

        R_k = self.R_k(user, item, self.k)
        if len(R_k) == 0:
            card_R_k = 0
        else:
            card_R_k =(1 / np.sqrt(len(R_k)))

        N_k = self.N_k(user, item, self.k)
        if len(N_k) == 0:
            card_N_k = 0
        else:
            card_N_k =(1 / np.sqrt(len(N_k)))

        bui = self.b_ui(user, item)


        v = self.p_u[user, :] + card_N * np.sum([self.y_j[i, :] for i in N])
        prod = self.q_i[item, :].T.dot(v)

        r_k = card_R_k * np.sum([
            (self.sparse_data[user,j] - self.b_ui(user,j)) * self.w[item,j]
            for j in R_k
        ])

        n_k = card_N_k * np.sum([
            self.c[item, j]
            for j in N_k
        ])
        res = bui + prod + r_k + n_k
        return res

    def predict(self, data):
        result = np.array([self.predict_single(data[i, 0], data[i, 1]) for i in range(data.shape[0])])
        return result

    def fit(self, data):
        self.average = np.mean(data[:, -1])

        num_users = np.max(data[:, 0])
        # M
        num_items = np.max(data[:, 1])

        self.sparse_data = rcm.to_sparse_matrix(data)

        # Bias for every user and item
        self.b_u = np.ones(num_users+1)
        self.b_i = np.ones(num_items+1)

        self.q_i = np.ones((num_items + 1, self.factors))
        self.p_u = np.ones((num_users + 1, self.factors))

        # factor vector in R^f
        self.x_j = np.ones((num_items + 1, self.factors))
        self.y_j = np.ones((num_items + 1 , self.factors))

        # Global weights w
        self.w = np.ones((num_items + 1, num_items + 1))
        self.c = np.ones((num_items + 1 , num_items +1))

        self.seen = {}
        for i in range(data.shape[0]):
            self.seen[data[i,0]] = True

        if self.verbose:
            print("Compute  n_ij")

        self.p_ij = rcm.distanceMatrix(sparse_data=self.sparse_data)

        self.n_ij = rcm.both_rated(self.sparse_data)
        temp = self.n_ij.copy()
        temp.data = temp.data + self.lambda2
        temp.data = 1 / temp.data
        self.n_ij_lambda = temp * self.n_ij
        self.s_ij = self.p_ij * self.n_ij_lambda
        #self.s_ij /= np.max(self.s_ij)

        self.iterations = 0
        while not self.converged:
            for i in tqdm.tqdm(range(data.shape[0]), desc="Epoch: " + str(self.iterations)):

                user = data[i, 0]
                item = data[i, 1]
                #print(user,item)
                rating = data[i, 2]
                e = rating - self.predict_single(user, item)
                #print("Err: ", e)
                #rated_items = self.N(user)

                N = self.N(user)
                if len(N) == 0:
                    card_N = 0
                else:
                    card_N = (1 / np.sqrt(len(N)))

                R_k = self.R_k(user, item, self.k)
                if len(R_k) == 0:
                    card_R_k = 0
                else:
                    card_R_k = (1 / np.sqrt(len(R_k)))

                N_k = self.N_k(user, item, self.k)
                if len(N_k) == 0:
                    card_N_k = 0
                else:
                    card_N_k = (1 / np.sqrt(len(N_k)))

                # Adjust b_u
                self.b_u[user] = self.b_u[user] + self.gamma1 * (
                        e - self.lambda6 * self.b_u[user]
                )

                # Adjust b_i
                self.b_i[item] = self.b_i[item] + self.gamma1 * (
                        e - self.lambda6 * self.b_i[item]
                )

                # Adjust q_i
                s = reduce(lambda x,y : x + y, [self.y_j[x, :] for x in N])
                self.q_i[item, :] = self.q_i[item, :] + self.gamma2 * (
                        e * (
                        self.p_u[user,:] + card_N * s
                ) - self.lambda7 * self.q_i[item, :]
                )

                # Adjust p_u
                self.p_u[user, :] = self.p_u[user,:] + self.gamma2 * (
                        e * self.q_i[item,:] - self.lambda7 * self.p_u[user, :]
                )

                for j in N:
                    self.y_j[j, :] = self.y_j[j, :] + self.gamma2 * (
                            e * card_N * self.q_i[item, :] - self.lambda7 * self.y_j[j, :]
                    )

                for j in R_k:
                    self.w[item, j] =  self.w[item, j] + self.gamma3 *(
                        card_R_k * e * (self.sparse_data[user, j] - self.b_ui(user, j) - self.lambda8 * self.w[item, j] )
                    )

                for j in N_k:
                    self.c[item,j] = self.c[item,j] + self.gamma3 * (
                        card_N_k * e - self.lambda8 * self.c[item, j]
                    )

            self.gamma1 *= 0.9
            self.gamma2 *= 0.9
            self.gamma3 *= 0.9
            if self.iterations >= self.max_it:
                self.converged = True
            self.iterations += 1


if __name__ == '__main__':
    import os
    import recommend as rcm

    train = np.genfromtxt(os.path.join("../", "data", "train.csv"), delimiter=",", dtype=np.int) + 1
    sgd = rcm.SVDPPRecommender(
    )
    err = rcm.cross_validation(sgd, train, 10, err=rcm.rmse)
    print('Error: ', err)
