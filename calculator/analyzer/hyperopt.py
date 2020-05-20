import numpy as np 
import pandas as pd 

from sklearn.model_selection import cross_val_score
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import copy
from analyzer.learners import scores, train_and_evaluate
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from analyzer.utils import top_features, remove_dir, impute_missing

def performance(l):
    return np.mean(l), np.median(l), np.min(l), np.max(l), np.round(np.std(l),2)

name_param_xgb = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", "colsample_bytree", "lambda"]
name_param_rf = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]
name_param_cart = ["max_depth", "min_weight_fraction_leaf", "min_samples_leaf", "min_samples_split", "min_impurity_decrease", "criterion"]
name_param_lr = ["penalty", "tol", "C", "solver"]
name_param_oct = ["max_depth", "criterion", "minbucket", "cp"]

algorithms = [xgb.XGBClassifier, RandomForestClassifier, DecisionTreeClassifier, LogisticRegression]
name_params = [name_param_xgb, name_param_rf, name_param_cart, name_param_lr, name_param_oct]

def optimizer(algorithm, name_param, X, y, cv = 300, n_calls = 500, name_algo = 'xgboost'):

    def objective(trial):

        space = hp.choice('classifier_type', [
            {
                'type': 'random_forest'
            },
            {
                'type': 'xgboost_classification',
                "n_estimators": hp.quniform("n_estimators", 10, 900, 1),
                "learning_rate": hp.loguniform("learning_rate", 1e-8, 1.0),
                "n_estimators": hp.quniform("n_estimators", 3, 10, 1),
                "min_child_weight": hp.loguniform("min_child_weight", 1e-8, 1.0),
                "gamma": hp.loguniform("gamma", 1e-8, 5),
                "colsample_bytree": hp.loguniform("colsample_bytree", 1e-2, 1),
                "lambda": hp.loguniform("lambda", 1e-8, 5),
                "alpha": hp.loguniform("alpha", 1e-8, 5)
            }])

        # Add a callback for pruning.
        model = algorithm()
        model.set_params(**params)
        score = np.mean(cross_val_score(model, X, y, cv = cv, n_jobs = -1, scoring="roc_auc"))
        #score = np.quantile(cross_val_score(model, X, y, cv = cv, n_jobs = -1, scoring="roc_auc"), 0.25)

        return score
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials = n_calls)
    
    best_params = study.best_params
    auc = study.best_value

    print('The best parameters are:')
    print('\n')
    print(pd.DataFrame(best_params.items(), columns = ['Parameter', 'Value']))
    print('\n')
    print('Cross-validation AUC = ', auc)

    best_model = algorithm()
    best_model.set_params(**best_params)

    print('Number of folds = ', cv)
    print('Maximize the first quantile AUC')
    seed = 30
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
    model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params) #gets in sample performance

    if name_algo != 'oct':
        top_features(model, X)
    
    return best_model, best_params

