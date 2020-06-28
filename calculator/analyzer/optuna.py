import numpy as np 
import pandas as pd 

from sklearn.model_selection import cross_val_score
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
name_param_kn = ['n_neighbors', 'weights', 'algorithm', 'leaf_size', 'p']
name_param_gb = ['var_smoothing']
name_param_svm = ['C', 'kernel', 'degree', 'probability', 'coef0']
name_param_mlp = ['activation', 'solver', 'alpha', 'learning_rate', 'tol', 'max_iter']
name_param_qda = ['reg_param', 'tol']

algo_names = ['xgboost','rf','cart','lr','oct', 'kn', 'svm', 'mlp', 'qda']

algorithms = {'xgboost': xgb.XGBClassifier,
            'rf': RandomForestClassifier, 
            'cart': DecisionTreeClassifier, 
            'lr': LogisticRegression,
            'kn': KNeighborsClassifier,
            'gb': GaussianNB,
            'svm': SVC,
            'mlp': MLPClassifier,
            'qda': QuadraticDiscriminantAnalysis}

name_params = {'xgboost': name_param_xgb,
            'rf': name_param_rf, 
            'cart': name_param_cart, 
            'lr': name_param_lr,
            'oct': name_param_oct,
            'kn': name_param_kn,
            'gb': name_param_gb,
            'svm': name_param_svm,
            'mlp': name_param_mlp,
            'qda': name_param_qda}

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
            params = {"penalty": trial.suggest_categorical("penalty", ['l2']),
                    "tol": trial.suggest_uniform("tol", 1e-10, 1),
                    "C": trial.suggest_uniform("C", 1e-10, 10),
                    "solver": trial.suggest_categorical("solver", ['lbfgs'])}

        elif name_algo == 'oct':

            from interpretableai import iai

            params = {"max_depth": trial.suggest_int("max_depth", 3, 10),
                    "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy', 'misclassification']),
                    "minbucket": trial.suggest_uniform("minbucket", 10**-6, 0.4),
                    "cp": trial.suggest_uniform("cp", 10**-12, 0.7)}

            params["max_depth"] = int(params["max_depth"])
            grid = iai.GridSearch(iai.OptimalTreeClassifier(random_seed = 1), **params) 

            grid.fit_cv(X, y, n_folds=cv, validation_criterion = 'auc')
            score = float(grid.get_grid_results()[['split' + str(i) + '_valid_score' for i in range(1, cv+1)]].T.mean())
            return score

        elif name_algo == 'kn':
            params = {"n_neighbors": trial.suggest_int("n_neighbors", 1, 80),
                    "weights": trial.suggest_categorical("weights", ['uniform', 'distance']),
                    "algorithm": trial.suggest_categorical("algorithm", ['ball_tree', 'kd_tree']),
                    "leaf_size": trial.suggest_int("leaf_size", 10, 100),
                    "p": trial.suggest_int("p", 1, 10)}

        elif name_algo == 'svm':
            params = {"C": trial.suggest_uniform("C", 1e-10, 25),
                    "kernel": trial.suggest_categorical("kernel", ['poly', 'rbf']),
                    "degree": trial.suggest_int("degree", 1, 5),
                    "probability": trial.suggest_int("probability", 1, 1),
                    "coef0": trial.suggest_uniform("coef0", -5, 5)}

        elif name_algo == 'gb':
            params = {"var_smoothing": trial.suggest_uniform("var_smoothing", 1e-10, 0.5)}

        elif name_algo == 'mlp':
            params = {"activation": trial.suggest_categorical("activation", ['tanh', 'relu']),
                    "solver": trial.suggest_categorical("solver", ['lbfgs', 'adam']),
                    "alpha": trial.suggest_uniform("alpha", 0, 10),
                    "learning_rate": trial.suggest_categorical("learning_rate", ['constant', 'adaptive']),
                    "tol": trial.suggest_uniform("tol", 1e-10, 1),
                    "max_iter": trial.suggest_int("max_iter", 1000, 1000)}

        elif name_algo == 'qda':
            params = {"reg_param": trial.suggest_uniform("reg_param", 1e-10, 1),
                    "tol": trial.suggest_uniform("tol", 1e-10, 1)}


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
    return best_model, best_params

