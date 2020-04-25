import numpy as np 
import pandas as pd 

from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import xgboost as xgb
import copy
from analyzer.learners import scores

from analyzer.utils import top_features, remove_dir
from skopt import gp_minimize


name_param_xgb = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", "colsample_bytree", "lambda", "alpha"]
name_param_rf = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",]

space_XGB  = [Integer(1, 1000, name="n_estimators"),
          Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, 40, name='max_depth'),
          Integer(2, 200, name='min_child_weight'),
          Real(0, 100, 'uniform',name='gamma'),
          Real(10**-5, 10**0, "log-uniform", name='colsample_bytree'),
          Integer(0, 200, name='lambda'),
          Integer(0, 200, name='alpha')]

space_RF  = [Integer(1, 1000, name = "n_estimators"),
          Integer(1, 40, name='max_depth'),
          Integer(1, 300, name ='min_samples_leaf'),
          Integer(1, 300, name = 'min_samples_split')]

algorithms = [xgb.XGBClassifier(), RandomForestClassifier()]
spaces = [space_XGB, space_RF]
name_params = [name_param_xgb, name_param_rf]

def optimizer(algorithm, space, name_param, X, y):

    @use_named_args(space)
    def objective(**params):

        scores = []

        for seed in range(1,11):
            model = copy.deepcopy(algorithm)
            model.set_params(**params) 

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
            scores.append(np.mean(cross_val_score(model, X_train.astype(np.float64), y_train.astype(int), cv = 10, n_jobs = -1, scoring="roc_auc")))
            print("Seed " + str(seed) + ' completed')

        return -np.mean(scores)

    opt_model = gp_minimize(objective, space, n_calls = 250, random_state = 1, verbose = True, n_random_starts = 20, n_jobs = -1)
    best_params = dict(zip(name_param, opt_model.x))    
    print('Overall AUC = ', - opt_model.fun)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = 1)
    
    best_model = copy.deepcopy(algorithm)
    best_model.set_params(**best_params)
    best_model.fit(X_train.astype(np.float64), y_train.astype(int))

    accTrain_RF, accTest_RF, ofs_fpr_RF, ofs_tpr_RF, isAUC_RF, ofsAUC_RF  = \
            scores(best_model,
                   X_train.astype(np.float64),
                   y_train.astype(int),
                   X_test.astype(np.float64),
                   y_test.astype(int))

    print('In Sample AUC', isAUC_RF)
    print('Out of Sample AUC', ofsAUC_RF)
    print('In Sample Misclassification', accTrain_RF)
    print('Out of Sample Misclassification', accTest_RF)
    top_features(best_model, X_train)
    return best_model
