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
lab_tests = True

# Load cremona data
data = cremona.load_cremona('../data/cremona/', lab_tests=lab_tests)

# Create dataset
X, y = create_dataset(data,
        lab=lab_tests,
        prediction = prediction)

X = X.astype(np.float64)
y = y.astype(int)

# Split in train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1,
#                                                      random_state=SEED)

for i in range(len(o.algorithms)):
        algorithm = o.algorithms[i]
        space = o.spaces[i]
        name_param = o.name_params[i]
        best_model = o.optimizer(algorithm, space, name_param, X, y, n_calls = 300)

