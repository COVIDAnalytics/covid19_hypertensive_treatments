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

from analyzer.utils import top_features, remove_dir, impute_missing

def performance(l):
    return np.mean(l), np.median(l), np.min(l), np.max(l), np.round(np.std(l),2)

name_param_xgb = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", "colsample_bytree", "lambda"]
name_param_rf = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]
name_param_cart = ["max_depth", "min_weight_fraction_leaf", "min_samples_leaf", "min_samples_split", "min_impurity_decrease", "criterion"]
name_param_lr = ["penalty", "tol", "C", "solver"]
name_param_oct = ["max_depth", "criterion", "minbucket", "cp"]

algorithms = [xgb.XGBClassifier, RandomForestClassifier, DecisionTreeClassifier, LogisticRegression, iai.OptimalTreeClassifier]
name_params = [name_param_xgb, name_param_rf, name_param_cart, name_param_lr, name_param_oct]
algo_names = ['xgboost','rf','cart','lr','oct']

def optimizer(algorithm, name_param, X, y, cv = 300, n_calls = 500, name_algo = 'xgboost'):

    def objective(trial):

        if name_algo == 'xgboost':
            params = {"n_estimators": trial.suggest_int("n_estimators", 10, 900),
                    "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_uniform("min_child_weight", 1e-8, 1.0),
                    "gamma": trial.suggest_uniform("gamma", 1e-8, 5),
                    "colsample_bytree": trial.suggest_uniform("colsample_bytree", 1e-2, 1),
                    "lambda": trial.suggest_uniform("lambda", 1e-8, 5),
                    "alpha": trial.suggest_uniform("alpha", 1e-8, 5), 
                    "eval_metric": "auc"}

            dtrain = xgb.DMatrix(X, label=y)
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
            history = xgb.cv(params, dtrain, num_boost_round = 100, callbacks = [pruning_callback], nfold = cv, stratified = True)
            score = history["test-auc-mean"].values[-1]
            return score

        elif name_algo == 'rf':
            params = {"n_estimators": trial.suggest_int("n_estimators", 10, 900),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_leaf": trial.suggest_uniform("min_samples_leaf", 1e-5, 0.5),
                    "min_samples_split": trial.suggest_uniform("min_samples_split", 1e-5, 0.5),
                    "max_features": trial.suggest_categorical("max_features", ['sqrt', 'log2'])}

        elif name_algo == 'cart':
            params = {"max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_weight_fraction_leaf": trial.suggest_uniform("min_weight_fraction_leaf", 0, 0.5),
                    "min_samples_leaf": trial.suggest_uniform("min_samples_leaf", 1e-5, 0.5),
                    "min_samples_split": trial.suggest_uniform("min_samples_split", 1e-5, 0.5),
                    "min_impurity_decrease": trial.suggest_uniform("min_impurity_decrease", 0, 1),
                    "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy'])}

        elif name_algo == 'lr':
            params = {"penalty": trial.suggest_categorical("penalty", ['l1','l2', 'none']),
                    "tol": trial.suggest_uniform("tol", 1e-5, 10),
                    "C": trial.suggest_uniform("C", 1e-5, 2),
                    "solver": trial.suggest_categorical("solver", ['saga'])}


        elif name_algo == 'oct':

            params = {"max_depth": trial.suggest_int("max_depth", 3, 10),
                    "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy', 'misclassification']),
                    "minbucket": trial.suggest_uniform("minbucket", 10**-6, 0.4),
                    "cp": trial.suggest_uniform("cp", 10**-12, 0.7)}

            params["max_depth"] = int(params["max_depth"])
            grid = iai.GridSearch(iai.OptimalTreeClassifier(random_seed = 1), **params) 

            grid.fit_cv(X, y, n_folds=cv, validation_criterion = 'auc')
            score = float(grid.get_grid_results()[['split' + str(i) + '_valid_score' for i in range(1, cv+1)]].T.mean())
            return score


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
    print('Maximize the average AUC')
    seed = 30
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
    model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params) #gets in sample performance

    if name_algo != 'oct'and name_algo != 'lr':
        top_features(model, X)
    
    return best_model, best_params

