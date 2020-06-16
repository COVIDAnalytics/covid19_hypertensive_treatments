import numpy as np 
import pandas as pd 

from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
import xgboost as xgb
import copy
from analyzer.learners import scores, train_and_evaluate

from analyzer.utils import top_features, remove_dir, impute_missing
from skopt import gp_minimize

def performance(l):
    return np.mean(l), np.median(l), np.min(l), np.max(l), np.round(np.std(l),2)

name_param_xgb = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", "colsample_bytree", "lambda"]
name_param_rf = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]
name_param_cart = ["max_depth", "min_weight_fraction_leaf", "min_samples_leaf", "min_samples_split", "min_impurity_decrease", "criterion"]
name_param_lr = ["penalty", "tol", "C", "solver"]
name_param_oct = ["max_depth", "criterion", "minbucket", "cp"]

algorithms = [xgb.XGBClassifier, RandomForestClassifier, DecisionTreeClassifier, LogisticRegression]
name_params = [name_param_xgb, name_param_rf, name_param_cart, name_param_lr, name_param_oct]

def optimizer(algorithm, name_param, X, y, cv = 200, n_calls = 500, name_algo = 'xgboost'):

    if name_algo == 'xgboost':
        n_features = len(X.columns)
        space  = [Integer(10, 900, name="n_estimators"),
                    Real(10**-4, 10**0, "log-uniform", name='learning_rate'),
                    Integer(3, 10, name='max_depth'),
                    Real(10**-7, 10**0, 'uniform', name='min_child_weight'),
                    Real(10**-7, 20, 'uniform', name='gamma'),
                    Real(10**-2, 10**-0, "uniform", name='colsample_bytree'),
                    Real(10**-7, 20, 'uniform', name='lambda')]

    elif name_algo == 'rf':
        n_features = len(X.columns)
        space  = [Integer(10, 900, name = "n_estimators"),
                    Integer(3, 10, name='max_depth'),
                    Real(10**-4, 0.5, "uniform", name ='min_samples_leaf'),
                    Real(10**-4, 0.5, "uniform", name = 'min_samples_split'),
                    Categorical(['sqrt', 'log2'], name = 'max_features')]

    elif name_algo == 'cart':
        n_features = len(X.columns)
        space  = [Integer(3, 10, name='max_depth'),
                    Real(0, 0.5, 'uniform', name ='min_weight_fraction_leaf'),
                    Real(10**-4, 0.5, "uniform", name ='min_samples_leaf'),
                    Real(10**-4, 0.5, "uniform", name = 'min_samples_split'),
                    Real(0, 1, 'uniform', name = 'min_impurity_decrease'),
                    Categorical(['gini', 'entropy'], name = 'criterion')]

    elif name_algo == 'lr':
        n_features = len(X.columns)
        space  = [Categorical(['l1','l2', 'none'], name = 'penalty'),
                    Real(10**-5, 10, 'uniform', name ='tol'),
                    Real(10**-4, 2, "uniform", name ='C'),
                    Categorical(['saga'], name = 'solver')]

    elif name_algo == 'oct':
        n_features = len(X.columns)
        space  = [Integer(3, 10, name='max_depth'),
                    Categorical(['gini', 'entropy', 'misclassification'], name = 'criterion'),
                    Real(10**-6, 0.4, "uniform", name ='minbucket'), 
                    Real(10**-12, 0.7, "uniform", name ='cp')]

    @use_named_args(space)
    def objective(**params):

        if name_algo != 'oct':
            model = algorithm()
            model.set_params(**params)
            score = np.mean(cross_val_score(model, X, y, cv = cv, n_jobs = -1, scoring="roc_auc"))

        else:
            from julia.api import Julia
            jl = Julia(compiled_modules=False)
            from interpretableai import iai

            params["max_depth"] = int(params["max_depth"])
            grid = iai.GridSearch(iai.OptimalTreeClassifier(random_seed = 1), **params) 

            grid.fit_cv(X, y, n_folds=cv, validation_criterion = 'auc')
            score = float(grid.get_grid_results()[['split' + str(i) + '_valid_score' for i in range(1, cv+1)]].T.mean())

        return - score

    opt_model = gp_minimize(objective, space, n_calls = n_calls, random_state = 1, verbose = True, n_random_starts = 30, n_jobs = -1)
    best_params = dict(zip(name_param, opt_model.x)) 

    print('The parameters are:')
    print(name_param)
    
    print('The parameters values are:')
    print(opt_model.x)


    print('The best parameters are:')
    print('\n')
    print(pd.DataFrame(best_params.items(), columns = ['Parameter', 'Value']))
    print('\n')
    print('Cross-validation AUC = ', - opt_model.fun)

    best_model = algorithm()
    best_model.set_params(**best_params)

    print('Number of folds = ', cv)
    print('Maximize the average AUC')
    seed = 30
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
    model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params) #gets in sample performance

    if name_algo != 'oct' and name_algo != 'lr':
        top_features(model, X)
    
    return best_model, best_params
