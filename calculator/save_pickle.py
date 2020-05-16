import pandas as pd
import numpy as np
import analyzer.utils as u
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from analyzer.learners import scores, train_and_evaluate

BEST_PARAMS_AND_SEEDS = {'mortality_with_lab': {'xgboost': {'best_seed': 30, 
                                                        'best_params': {'n_estimators': 1062,
                                                                        'learning_rate': 0.041787,
                                                                        'max_depth': 1,
                                                                        'min_child_weight': 0.0001,
                                                                        'gamma': 0.0001,
                                                                        'colsample_bytree': 0.139031,
                                                                        'lambda': 0.0001,
                                                                        'alpha': 0.0001}},
                                                'cart': {'best_seed': 16, 
                                                        'best_params': { 'max_depth': 22,
                                                        'min_weight_fraction_leaf':  0.0095584,
                                                        'min_samples_leaf': 0.0639822,
                                                        'min_samples_split': 0.14137,
                                                        'min_impurity_decrease':  0,
                                                        'criterion':    'entropy'}},
                                                'lr': {'best_seed': 6, 
                                                        'best_params': {'penalty': 'none',
                                                                        'tol':  0.00501902,
                                                                        'C': 96.9518,
                                                                        'solver': 'saga'}}},

                        'mortality_without_lab': {'xgboost': {'best_seed': 26, 
                                                        'best_params': {'n_estimators': 619,
                                                                        'learning_rate': 0.030293,
                                                                        'max_depth': 2,
                                                                        'min_child_weight': 9.752503,
                                                                        'gamma': 0.000100,
                                                                        'colsample_bytree': 1.00,
                                                                        'lambda': 60,
                                                                        'alpha':   0.000100}},
                                                'cart': {'best_seed': 26, 
                                                        'best_params': { 'max_depth': 6,
                                                                        'min_weight_fraction_leaf':  0.0399055,
                                                                        'min_samples_leaf': 0.0001,
                                                                        'min_samples_split': 0.0001,
                                                                        'min_impurity_decrease':  0,
                                                                        'criterion':    'gini'}},
                                                'lr': {'best_seed': 18, 
                                                        'best_params': {'penalty': 'l1', 
                                                                        'tol': 1e-05, 
                                                                        'C': 100, 
                                                                        'solver': 'saga'}}},

                        'infection_with_lab': {'xgboost': {'best_seed': 30, 
                                                        'best_params': {'n_estimators': 790,
                                                                        'learning_rate': 0.119077,
                                                                        'max_depth': 10,
                                                                        'min_child_weight': 5.611450,
                                                                        'gamma': 1.907042,
                                                                        'colsample_bytree': 0.454366,
                                                                        'lambda': 59.567080,
                                                                        'alpha':1.420127}},
                                                'cart': {'best_seed': 29, 
                                                        'best_params': {'max_depth': 20,
                                                                        'min_weight_fraction_leaf':  0.0167917,
                                                                        'min_samples_leaf': 0.0209234,
                                                                        'min_samples_split': 0.0001,
                                                                        'min_impurity_decrease':  0,
                                                                        'criterion':    'entropy'}},
                                                'lr': {'best_seed': 28, 
                                                        'best_params': {'penalty': 'l1',
                                                                        'tol':  1e-05,
                                                                        'C': 100,
                                                                        'solver': 'saga'}}},

                        'infection_without_lab': {'xgboost': {'best_seed': 6, 
                                                        'best_params': {'n_estimators': 877,
                                                                        'learning_rate': 0.019551,
                                                                        'max_depth': 4,
                                                                        'min_child_weight': 0.0001,
                                                                        'gamma': 0.0001,
                                                                        'colsample_bytree': 1.00,
                                                                        'lambda': 60.00,
                                                                        'alpha': 8.755914}},
                                                'cart': {'best_seed': 29, 
                                                        'best_params': { 'max_depth': 6,
                                                                        'min_weight_fraction_leaf':  0.0558581,
                                                                        'min_samples_leaf': 0.0215371,
                                                                        'min_samples_split': 0.0001,
                                                                        'min_impurity_decrease':  0,
                                                                        'criterion':    'gini'}},
                                                'lr': {'best_seed': 11, 
                                                        'best_params': {'penalty': 'none',
                                                                        'tol':  1e-05,
                                                                        'C': 95.2635,
                                                                        'solver':    'saga'}}}}


predictions = ['mortality_with_lab', 'mortality_without_lab', 'infection_with_lab', 'infection_without_lab']
algorithms_name = ['xgboost', 'cart', 'lr']
algorithms = [xgb.XGBClassifier, DecisionTreeClassifier, LogisticRegression]
website_path = '../../website/assets/risk_calculators'
post_processing_path = '../../covid19_clean_data'

categorical = ['Gender'] 
comorbidities = ['Cardiac dysrhythmias',
                'Chronic kidney disease',
                'Coronary atherosclerosis and other heart disease', 
                #'Essential hypertension',
                'Diabetes']
symptoms = []

for i in range(len(predictions)):
        prediction = predictions[i]
        best_seed = BEST_PARAMS_AND_SEEDS[prediction]['xgboost']['best_seed']
        best_params = BEST_PARAMS_AND_SEEDS[prediction]['xgboost']['best_params']

        if i == 0: #mortality_with_lab
                u.create_and_save_pickle(xgb.XGBClassifier, X, y, best_seed, best_seed, best_params, categorical, #Save on the website
                                         symptoms, comorbidities, prediction, website_path + '/mortality/model_with_lab.pkl', 
                                         data_save = True, data_in_pickle = False, post_processing_path)

                for current_seed in range(1, 41):
                        for j in range(len(algorithms)):
                                u.create_and_save_pickle(algorithm, X, y, current_seed, best_seed, best_params, categorical, #Save for post processing
                                                        symptoms, comorbidities, prediction, website_path + '/mortality/model_with_lab.pkl', 
                                                        data_save = True, data_in_pickle = False, post_processing_path)


