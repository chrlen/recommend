import numpy as np
import tqdm
import copy
import scipy.sparse as sps
import multiprocessing as mp
import itertools as it
import functools as ft
import os
import scipy.spatial.distance as ssd


def load_data():
    d = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int) + 1
    d = d[np.random.permutation(range(d.shape[0])), :]
    return d


def load_data_small(n=3):
    d = np.genfromtxt(os.path.join("data", "train.csv"), delimiter=",", dtype=np.int) + 1
    q = np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)

    users = np.unique(d[:, 0])[:n].tolist()
    train = d[np.isin(d[:, 0],  users)]
    qual = q[np.isin(q[:, 0], users)]
    return train, qual


def load_qualifying_data():
    return np.genfromtxt(os.path.join("data", "qualifying_blanc.csv"), delimiter=",", dtype=np.int)


def to_sparse_matrix(train):
    n_cols = np.max(train[:, 1]) + 1
    n_rows = np.max(train[:, 0]) + 1
    sparse = sps.coo_matrix(
        ((train[:, 2], (train[:, 0], train[:, 1]))),
        shape=(n_rows, n_cols)
    )
    # sparse = sps.lil_matrix((n_rows +1, n_cols+1))
    # for i in range(train.shape[0]):
    #     sparse[
    #         train[i,0],
    #         train[i,1]] = train[i,2]
    return sparse.tocsr().asfptype()


def to_dense_matrix(train):
    return to_sparse_matrix(train).todense()


def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))


def confusion(y, y_hat):
    """ calculate confusion matrix """
    n_cols = max(
        np.unique(y).shape[0],
        np.unique(y_hat).shape[0]
    )
    conf = np.zeros((n_cols, n_cols))
    for i in range(y.shape[0]):
        if (y[i] != y_hat[i]):
            conf[y[i], y_hat[i]] += 1
    print(conf)
    return conf


def confusion_norm(y, y_hat):
    c = confusion(y, y_hat)
    return np.linalg.norm(c)


def eval(params, save=True, save_path='models'):
    model, train, test, err = params
    print("eval: " + model.name)
    model.fit(train)
    result = model.predict(data=test[:, 0:-1])
    error = err(result, test[:, -1])
    # if save:
    #     model.save(path=save_path + '/' + )
    return error


def strong_cross_validation(model, data, split=10, err=rmse, random_state=np.random.RandomState()):
    users = np.unique(data[:, 0])
    by_user = [data[data[:, 0] == u, :] for u in users]
    split_length = int(len(by_user) / split)
    split = (by_user[x:x + split_length] for x in range(0, len(by_user), split_length))
    split = [np.vstack(x) for x in split]

    index = range(len(split))

    training_splits = (
        split[:i] + split[(i + 1):]
        for i in index
    )

    training_sets = (
        np.vstack(split) for split in training_splits
    )

    test_sets = (
        split[i] for i in index
    )

    pairs = zip(
        it.repeat(model, len(split)),
        training_sets,
        test_sets,
        it.repeat(err, len(split)))

    # with mp.Pool(mp.cpu_count() - 1 ) as p:
    results = list(map(eval, pairs))

    return np.mean(results)


def cross_validation(model, data, split=10, err=rmse, random_state=np.random.RandomState(), verbose=False):

    split = np.array_split(
        data,
        split
    )

    index = range(len(split))

    training_splits = (
        split[:i] + split[(i + 1):]
        for i in index
    )

    training_sets = (
        np.vstack(split) for split in training_splits
    )

    test_sets = (
        split[i] for i in index
    )

    models = [copy.deepcopy(model) for i in range(len(split))]

    pairs = zip(
        models,
        training_sets,
        test_sets,
        it.repeat(err, len(split)))

    results = list(map(eval, pairs))
    err = np.mean(results)
    if verbose:
        print(err)
    return err


def both_rated(sparse_matrix):
    """ returns counts of users that have rated the same product for every product """

    def distances(par):
        target_index = par[0]
        list_of_pairs = par[1]
        target_column = list_of_pairs[target_index][0]
        target_set = list_of_pairs[target_index][1]

        return [
            (
                target_column,
                i[0],
                len(target_set.intersection(i[1]))
            )
            for i in sets
        ]

    num_items = sparse_matrix.shape[0]
    sets = [(i, set(np.argwhere(sparse_matrix[:, i] != 0)[:, 0].tolist())) for i in range(sparse_matrix.shape[1])]
    non_empty = [s for s in sets if len(s[1]) > 0]
    d = map(
        distances,
        zip(
            range(len(non_empty)),
            it.repeat(non_empty, len(non_empty))
        )
    )
    flat = (item for sublist in d for item in sublist)
    nonzero = list(filter(lambda x: x[2] != 0, flat))

    rows = [x[0] for x in nonzero]
    columns = [x[1] for x in nonzero]
    values = [x[2] for x in nonzero]

    return sps.coo_matrix(
        (values, (rows, columns)),
        shape=(sparse_matrix.shape[1],
               sparse_matrix.shape[1])
    )


def sparsePearson(sparse_data):
    print("bla")
    mu = sparse_data.mean(axis=0)
    # first_moment= sparse_data.subtract(mu)


def distanceMatrix(sparse_data, type='corr'):
    m = sparse_data.todense().T
    d = ssd.squareform(ssd.pdist(m, metric='cosine'))
    d[np.isnan(d)] = 0
    s = sps.csc_matrix(d)
    # d /= np.max(d)
    return s


def correlation(sparse_matrix, add=1e-5):
    m = sparse_matrix.todense() + add
    m = np.corrcoef(m)
    m[m < add] = 0
    return sps.csr_matrix(m)
