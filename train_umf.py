import os
import numpy as np
import recommend as rcm
import time

np.seterr(all='raise')

train, qualify = rcm.load_data_small()
nmf = rcm.UnconstrainedMFRecommender()
nmf.fit(data=train)
best_model_prediction = nmf.predict(data=qualify).reshape((-1, 1))
qualifying_prediction = np.append(qualify, np.full((qualify.shape[0], 1), best_model_prediction),axis=1)
qualifying_file_name  = 'predictions/' + 'final' + nmf.name + str(time.time()) +  ".csv"
np.savetxt(qualifying_file_name, qualifying_prediction, delimiter=",", newline="\n", encoding="utf-8")


