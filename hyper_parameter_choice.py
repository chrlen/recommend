import scipy.optimize as scpo
import numpy as np
import recommend as rcm

np.seterr(all='raise')

def svd_parameter(x):
    return rcm.cross_validation(
        rcm.SVDPPRecommender(
            **{
                'maxit': 10,
                'factors': 100,
                'lambda1': x[0],
                'lambda2': x[1],
                'lambda3': x[2],
                'lambda4': x[3],
                'lambda5': x[4],
                'lambda6': x[5],
                'lambda7': x[6],
                'lambda8': x[7],
                'gamma1': x[8],
                'gamma2': x[9],
                'gamma3': x[10]
            }
        ),
        data=rcm.load_data_small(),
        err=rcm.rmse,
        split=3
    )

x0 = np.array([
    0.005,
    100,
    100,
    0.01,
    0.01,
    0.01,
    0.015,
    0.015,
    0.00000015,
    0.00000015,
    0.00000015
    ])

scpo.minimize(
    svd_parameter,
    x0=x0,
    method='Nelder-Mead'
)


