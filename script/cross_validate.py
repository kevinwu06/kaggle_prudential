from preprocess import clean_data
import numpy as np
import xgboost as xgb
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import StratifiedKFold

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)

def get_params():
    
    params = {}
    params["objective"] = "reg:linear"  
    params["eta"] = 0.05
    params["min_child_weight"] = 60
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.5
    params["silent"] = 1
    params["max_depth"] = 9
    plst = list(params.items())

    return plst
    
# global variables
xgb_num_rounds = 300

# preprocess data
M = clean_data()
train, test = M.data_split()
columns_to_drop = M.columns_to_drop  

# get the parameters for xgboost
plst = get_params()

skf = StratifiedKFold(train['Response'].values, n_folds=3, random_state=1234)
scores = []

for train_index, test_index in skf:
    X_train, X_test = train.iloc[train_index], train.iloc[test_index]
    xgtrain = xgb.DMatrix(X_train.drop(columns_to_drop, axis=1), X_train['Response'].values)  
    xgtest = xgb.DMatrix(X_test.drop(columns_to_drop, axis=1), X_test['Response'].values)  
    model = xgb.train(plst, xgtrain, xgb_num_rounds)
    test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
    score = eval_wrapper(test_preds, X_test['Response'])
    scores.append(score)

print 'Score: %f +/- %.3f' % (np.mean(scores), np.std(scores))
    