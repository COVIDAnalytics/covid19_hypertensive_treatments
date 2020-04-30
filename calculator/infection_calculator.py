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

from analyzer.utils import store_json
import analyzer.dataset as ds

import analyzer.optimizer as o

SEED = 1
prediction = 'Swab'
folder_name = 'swab_prediction_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/swab'

discharge_data = False
comorbidities_data = False
vitals_data = True
lab_tests = True
demographics_data = True
swabs_data = True
icu_data = False

# Load cremona data
data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)

# Create dataset
X, y = ds.create_dataset(data,
                         discharge_data,
                         comorbidities_data,
                         vitals_data,
                         lab_tests,
                         demographics_data,
                         swabs_data,
                         prediction=prediction)

algorithm = o.algorithms[0]
name_param = o.name_params[0]

X, bounds_dict = ds.filter_outliers(X)
store_json(bounds_dict, 'infection_bounds.json')

best_xgb = o.optimizer(algorithm, name_param, X, y, seed_len = 10, n_calls = 400, name_algo = 'xgboost')
