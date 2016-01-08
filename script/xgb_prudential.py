# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 15:18:21 2016

@author: kwu
"""

import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer 

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
    params["colsample_bytree"] = 0.50
    params["silent"] = 1
    params["max_depth"] = 9
    plst = list(params.items())

    return plst
    
def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

# global variables
columns_to_drop = ['Id', 'Response']
xgb_num_rounds = 500
num_classes = 8

print("Load the data using pandas")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# combine train and test
all_data = train.append(test)

# not present in test set
'''
Product_Info_7!=2
Insurance_History_3!=2
Medical_History_5!=3
Medical_History_6!=2
Medical_History_9!=3
Medical_History_12!=1
Medical_History_16!=2
Medical_History_17!=1
Medical_History_23!=2
Medical_History_31!=2
Medical_History_37!=3
Medical_History_41!=2
'''

# factorize categorical variables    
# all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
'''
The following variables are all categorical (nominal):
Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41

The following variables are continuous:
Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5

The following variables are discrete:
Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32

Medical_Keyword_1-48 are dummy variables.
'''
categorical = ['Product_Info_1', 'Product_Info_2', 'Product_Info_3', 
               'Product_Info_5', 'Product_Info_6', 'Product_Info_7', 
               'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_5', 
               'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 
               'InsuredInfo_4', 'InsuredInfo_5', 'InsuredInfo_6', 
               'InsuredInfo_7', 'Insurance_History_1', 'Insurance_History_2', 
               'Insurance_History_3', 'Insurance_History_4', 
               'Insurance_History_7', 'Insurance_History_8', 
               'Insurance_History_9', 'Family_Hist_1', 'Medical_History_2', 
               'Medical_History_3', 'Medical_History_4', 'Medical_History_5', 
               'Medical_History_6', 'Medical_History_7', 'Medical_History_8', 
               'Medical_History_9', 'Medical_History_10', 'Medical_History_11',
               'Medical_History_12', 'Medical_History_13', 'Medical_History_14', 
               'Medical_History_16', 'Medical_History_17', 'Medical_History_18', 
               'Medical_History_19', 'Medical_History_20', 'Medical_History_21', 
               'Medical_History_22', 'Medical_History_23', 'Medical_History_25', 
               'Medical_History_26', 'Medical_History_27', 'Medical_History_28', 
               'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 
               'Medical_History_33', 'Medical_History_34', 'Medical_History_35', 
               'Medical_History_36', 'Medical_History_37', 'Medical_History_38', 
               'Medical_History_39', 'Medical_History_40', 'Medical_History_41', 
               'Medical_History_1', 'Medical_History_15', 'Medical_History_24', 
               'Medical_History_32']

for f in categorical:
    all_data_dummy = pd.get_dummies(all_data[f], prefix=f)
    all_data = all_data.drop([f], axis=1)
    all_data = pd.concat((all_data, all_data_dummy), axis=1)


print('Eliminate missing values')    
# Use -1 for any others
all_data.fillna(-1, inplace=True)

# fix the dtype on the label column
all_data['Response'] = all_data['Response'].astype(int)

# Provide split column
all_data['Split'] = np.random.randint(5, size=all_data.shape[0])

# split train and test
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

# convert data to xgb data structure
xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values)
xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values)    

# get the parameters for xgboost
plst = get_params()
print(plst)

# only keep features done during feature selection
feat_keep = pd.read_csv('features.csv')
all_data = all_data[feat_keep.feature.values]      

model = xgb.train(plst, xgtrain, xgb_num_rounds) 
'''
############ tune hyperparameters ############

clf = xgb.sklearn.XGBClassifier()

parameters = {
        'max_depth':[9, 15], 
        'learning_rate':[0.01, 0.05, 0.1],
        'n_estimators':[500, 1000],
        'objective':'reg:linear',
        'min_child_weight':[60],
        'max_delta_step':[0],
        'subsample':[0.5],
        'colsample_bytree':[0.5],
        'seed':[0],     
}

rand = RandomizedSearchCV(clf, parameters, cv=3, n_iter=20, 
                          scoring=make_scorer(eval_wrapper), verbose=1)
rand.fit(train.drop(columns_to_drop, axis=1),train['Response'])
scores =  rand.grid_scores_
print rand.best_score_
print rand.best_params_

###############################################

def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

create_feature_map(list(all_data.columns.values))
importance = model.get_fscore(fmap='xgb.fmap')
importance_df = pd.DataFrame(importance.items(), columns=['feature','fscore'])
importance_df.to_csv('features.csv',index=False)
'''

# get preds
train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
print('Train score is:', eval_wrapper(train_preds, train['Response'])) 
test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)
train_preds = np.clip(train_preds, -0.99, 8.99)
test_preds = np.clip(test_preds, -0.99, 8.99)

# train offsets 
offsets = np.ones(num_classes) * -0.5
offset_train_preds = np.vstack((train_preds, train_preds, train['Response'].values))
for j in range(num_classes):
    train_offset = lambda x: -apply_offset(offset_train_preds, x, j)
    offsets[j] = fmin_powell(train_offset, offsets[j])  

# apply offsets to test
data = np.vstack((test_preds, test_preds, test['Response'].values))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)

preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
preds_out = preds_out.set_index('Id')
preds_out.to_csv('sub.csv')
