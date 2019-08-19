import os
import numpy as np
import recommend as rcm
import time

np.seterr(all='raise')

train = rcm.load_data()
qualify = rcm.load_qualifying_data()

lambdas = [ 10**n for n in range(-2, 2) ]

nmf = rcm.UnconstrainedRegularisedMFRecommender(
    **{
        'max_it': np.inf,
        'verbose': True,
        'lambda0': 0.001
    }
)

#pred_err = [
#    rcm.cross_validation(model=rcm.UnconstrainedRegularisedMFRecommender(
#        **{
#            'max_it': np.inf,
#            'verbose': False,
#            'lambda0': l,
#            'convergence_error': 0.001
#        }
#    ), data=train, err=rcm.rmse) for l in lambdas
#]

nmf.fit(data=train)
best_model_prediction = nmf.predict(data=qualify).reshape((-1, 1)) - 1
best_model_prediction[best_model_prediction < 0] = 0
best_model_prediction[best_model_prediction > 5 ] = 5
qualifying_prediction = np.append(qualify, np.full((qualify.shape[0], 1), best_model_prediction), axis=1)
qualifying_file_name = 'predictions/' + 'final' + nmf.name + str(time.time()) + ".csv"
np.savetxt(qualifying_file_name, qualifying_prediction, delimiter=",", newline="\n", encoding="utf-8")
