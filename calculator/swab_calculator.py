import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from analyzer.dataset import create_dataset
from analyzer.loaders.cremona.swabs import load_swabs
from analyzer.utils import create_dir, export_features_json, plot_correlation
from analyzer.learners import train_oct
from analyzer.learners import xgboost_classifier
from analyzer.learners import rf_classifier

import analyzer.optimizer as o

SEED = 1
prediction = 'Swab'
folder_name = 'swab_prediction_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/swab'

lab_tests = True
vitals = True

# Load swab data
data = load_swabs('../data/cremona/', lab_tests = lab_tests)

# Create dataset
X, y = create_dataset(data,
        comorbidities = False,
        vitals=vitals,
        lab=lab_tests,
        prediction = prediction)


best_models = []
for i in range(len(o.algorithms)):
        algorithm = o.algorithms[i]
        space = o.spaces[i]
        name_param = o.name_params[i]

        best_models.append(o.optimizer(algorithm, space, name_param, X, y, n_calls = 300))
