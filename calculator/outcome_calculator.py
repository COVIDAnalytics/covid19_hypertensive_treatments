import numpy as np
import pandas as pd
import os


from sklearn.model_selection import train_test_split

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


import analyzer.loaders.cremona as cremona
from analyzer.dataset import create_dataset
from analyzer.utils import create_dir, export_features_json, plot_correlation
from analyzer.learners import train_oct
from analyzer.learners import xgboost_classifier
from analyzer.learners import rf_classifier

import analyzer.optimizer as o


SEED = 1
prediction = 'Outcome'
folder_name = 'complete_lab_tests_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/outcome'

discharge_data = True
comorbidities_data = True
vitals_data = True
lab_tests = True
anagraphics_data = False
swabs_data = False
icu_data = False

# Load cremona data
data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data)

X, y = create_dataset(data,
                        discharge_data, 
                        comorbidities_data, 
                        vitals_data, 
                        lab_tests, 
                        anagraphics_data, 
                        swabs_data,
                        prediction = prediction)



n_features = len(X.columns)

space_XGB  = [Integer(10, 1300, name="n_estimators"),
          Real(10**-4, 10**0, "log-uniform", name='learning_rate'),
          Integer(1, n_features, name='max_depth'),
          Real(10**-4, 20, 'uniform', name='min_child_weight'),
          Real(10**-4, 40, 'uniform', name='gamma'),
          Real(10**-3, 10**0, "log-uniform", name='colsample_bytree'),
          Real(10**-4, 60, 'uniform', name='lambda'),
          Real(10**-4, 30, 'uniform', name='alpha')]

space_RF  = [Integer(10, 2000, name = "n_estimators"),
          Integer(1, 40, name='max_depth'),
          Integer(1, 300, name ='min_samples_leaf'),
          Integer(2, 300, name = 'min_samples_split'),
          Categorical(['sqrt', 'log2'], name = 'max_features')]

spaces = [space_XGB, space_RF]

algorithm = o.algorithms[0]
space = spaces[0]
name_param = o.name_params[0]

best_xgb = o.optimizer(algorithm, space, name_param, X, y, n_calls = 500)

