import numpy as np
import recommend as rcm
import time


##
train = rcm.load_data()
qualifying_data = rcm.load_qualifying_data()

predictors = [
    rcm.UnconstrainedMFRecommender(
        **{
            'max_it': np.inf,
            'alpha': 0.001
        }
    )
]

pred_err = [(model, rcm.cross_validation(model=model, data=train, err=rcm.rmse)) for model in predictors]

best = np.argsort(
    np.array([p[1] for p in pred_err]).reshape(-1)
)[0]

best_model = pred_err[0][0]
best_model_err = str(pred_err[best][1])
best_model.fit(data=train)
best_model_prediction = best_model.predict(data=qualifying_data).reshape((-1, 1)) - 1

#conf = rcm.confusion(train[:, 2], best_model_prediction)

# qualifying_prediction = np.append(qualifying_data, np.full((qualifying_data.shape[0], 1), best_model_prediction),
#                                   axis=1)
# qualifying_file_name  = 'predictions/' + str(best_model_err) + '_' +  best_model.name + str(time.time()) +  ".csv"
#
# np.savetxt(qualifying_file_name, qualifying_prediction,
#            delimiter=",", newline="\n", encoding="utf-8")
#
# model_file_name = 'models/' + best_model_err + '_' +  best_model.name + ".pcl"
#
# best_model.save(model_file_name=model_file_name)


