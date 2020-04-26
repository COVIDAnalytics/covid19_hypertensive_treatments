import numpy as np 
import pandas as pd 

from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
import xgboost as xgb
import copy
from analyzer.learners import scores, train_and_evaluate

from analyzer.utils import top_features, remove_dir
from skopt import gp_minimize


name_param_xgb = ["n_estimators", "learning_rate", "max_depth", "min_child_weight", "gamma", "colsample_bytree", "lambda", "alpha"]
name_param_rf = ["n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features"]

algorithms = [xgb.XGBClassifier, RandomForestClassifier]
name_params = [name_param_xgb, name_param_rf]

def optimizer(algorithm, name_param, X, y, n_calls = 500, name_algo = 'xgboost'):

    if name_algo == 'xgboost':
        n_features = len(X.columns)
        space  = [Integer(10, 1300, name="n_estimators"),
                    Real(10**-4, 10**0, "log-uniform", name='learning_rate'),
                    Integer(1, n_features, name='max_depth'),
                    Real(10**-4, 20, 'uniform', name='min_child_weight'),
                    Real(10**-4, 40, 'uniform', name='gamma'),
                    Real(10**-4, 10**0, "log-uniform", name='colsample_bytree'),
                    Real(10**-4, 60, 'uniform', name='lambda'),
                    Real(10**-4, 30, 'uniform', name='alpha')]

    elif name_algo == 'rf':
        n_features = len(X.columns)
        space  = [Integer(10, 2000, name = "n_estimators"),
                    Integer(1, n_features, name='max_depth'),
                    Real(1, 200, 'uniform', name ='min_samples_leaf'),
                    Real(2, 200, 'uniform', name = 'min_samples_split'),
                    Categorical(['sqrt', 'log2'], name = 'max_features')]

    @use_named_args(space)
    def objective(**params):

        scores = []

        for seed in range(1,11):
            model = algorithm()
            model.set_params(**params) 

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
            scores.append(np.mean(cross_val_score(model, X_train, y_train, cv = 10, n_jobs = -1, scoring="roc_auc")))

        return -np.mean(scores)

    opt_model = gp_minimize(objective, space, n_calls = n_calls, random_state = 1, verbose = True, n_random_starts = 20, n_jobs = -1)
    best_params = dict(zip(name_param, opt_model.x)) 

    print('The best parameters are:')
    print('\n')
    print(pd.DataFrame(best_params.items(), columns = ['Parameter', 'Value']))
    print('\n')
    print('Cross-validation AUC = ', - opt_model.fun)

    inmis = []
    outmis = []
    inauc = []
    outauc = []

    for seed in range(1,11):
        best_model, accTrain, accTest, ofs_fpr, ofs_tpr, isAUC, ofsAUC = train_and_evaluate(algorithm, X, y, seed, best_params)
        inmis.append(accTrain)
        outmis.append(accTest)
        inauc.append(isAUC)
        outauc.append(ofsAUC)
        
    print('Average In Sample AUC', np.mean(inauc))
    print('Average Out of Sample AUC', np.mean(outauc))
    print('Average In Sample Misclassification', np.mean(inmis))
    print('Average Out of Sample Misclassification', np.mean(outmis))
    top_features(best_model, X_train)
    
    return best_model
