import os
import numpy as np
import recommend as rcm
import time
np.seterr(all='raise')
#train = rcm.load_data_small()
train = rcm.load_data()
#max_user = np.max(np.unique(train[:,0]))
qualifying_data = rcm.load_qualifying_data()
#qualifying_data = qualifying_data[qualifying_data[:,0] < max_user,:]



sgd = rcm.SVDPPRecommender(
    **{
    'maxit': 10,
    'factors': 100,
    'lambda1': 0.0048961078157, 
    'lambda2': 99.94278003768898, 
    'lambda3': 97.7882099357054, 
    'lambda4': 0.009770785959429812, 
    'lambda5': 0.010323770339885211, 
    'lambda6': 0.009989072461961084, 
    'lambda7': 0.015218737492343207, 
    'lambda8': 0.01525962800451102, 
    'gamma1': 1.5162291905528836e-07, 
    'gamma2': 1.6785725475570412e-07, 
    'gamma3': 1.401672614774242e-07
    }
)

#eval: sgd_maxit:10_factors:100_lambda1:0.004896107815724864_lambda2:99.94278003768898_lambda3:97.7882099357054_lambda4:0.009770785959429812_lambda5:0.010323770339885211_lambda6:0.009989072461961084_lambda7:0.015218737492343207_lambda8:0.01525962800451102_gamma1:1.5162291905528836e-07_gamma2:1.6785725475570412e-07_gamma3:1.401672614774242e-07


sgd.fit(data=train)
best_model_prediction = sgd.predict(data=qualifying_data).reshape((-1, 1))
qualifying_prediction = np.append(qualifying_data, np.full((qualifying_data.shape[0], 1), best_model_prediction),axis=1)
qualifying_file_name  = 'predictions/' + 'final' +  sgd.name + str(time.time()) +  ".csv"
np.savetxt(qualifying_file_name, qualifying_prediction, delimiter=",", newline="\n", encoding="utf-8")


