import recommend as rcm
import numpy as np
import numpy.linalg as npl


class SVDRecommender(rcm.Recommender):
    def __init__(self, **kwargs):
        rcm.Recommender.__init__(self, 'svd', **kwargs)
        self.maxit = kwargs.get('maxit', 10)
        self.iterations = 0
        self.k = kwargs.get('k',10)

    def predict(self, data):
        p = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            user = data[i,0]
            item = data[i,1]
            if user < self.recommendations.shape[0]:
                p[i] = self.recommendations[user, item]
            else:
                #p[i] = self.item_means[0, item]
                p[i] = self.all_mean
        return p


    def fit(self, data):
        self.initial_matrix = data
        self.all_mean = np.mean(data[:, 2])
        self.recommendations = rcm.to_dense_matrix(data)
        self.initial_mask = self.recommendations == 0
        self.item_means = self.recommendations.mean(axis=0).reshape(-1)
        col_means = self.recommendations.mean(axis=1)

        mean_matrix = np.repeat(
            col_means,
            self.recommendations.shape[1]
        ).reshape(self.recommendations.shape)

        self.recommendations = np.where(
            self.initial_mask,
            self.all_mean,
            self.recommendations
        )

        while self.iterations < self.maxit:
            print("it: " + str(self.iterations))
            Q, S, P = npl.svd(
                self.recommendations,
                full_matrices=False
            )

            #smat = np.zeros(shape=(Q.shape[0],P.shape[0]))
            #smat[:P.shape[0],:P.shape[0]] = np.diag(S)

            temp = Q.dot(np.diag(S)).dot(P)
            self.recommendations = np.where(self.initial_mask, temp, self.recommendations)
            self.iterations += 1


if __name__ == '__main__':
    import os
    import recommend as rcm
    import numpy as np
    train = np.genfromtxt(os.path.join("../","data", "train.csv"), delimiter=",", dtype=np.int)
    qual = np.genfromtxt(os.path.join("../","data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)

    sgd = rcm.SVDRecommender(
        maxit=20
    )
    sgd.fit(train)
    p = sgd.predict(qual)
    #err = rcm.cross_validation(sgd, train, 10, err=rcm.rmse)
    #print('Error: ', err)

