import pandas as pd
import os


from sklearn.model_selection import train_test_split

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from analyzer.dataset import create_dataset
from analyzer.loaders.swabs import load_swabs
from analyzer.utils import create_dir, export_features_json, plot_correlation
from analyzer.learners import train_oct
from analyzer.learners import xgboost_classifier
from analyzer.learners import rf_classifier


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

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1,
                                                     random_state=SEED)

# Train trees
# output_path = os.path.join(output_folder, folder_name, 'oct')
# create_dir(output_path)
# oct_scores = train_oct(X_train, y_train, X_test, y_test, output_path, seed=SEED)


#PARAMETERS GRID
param_grid_XGB = {
        "n_estimators": [20, 50, 80, 100, 120, 150, 200],
        "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
        "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight" : [ 1, 3, 5, 7 ],
        "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
        "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


output_path_XGB = os.path.join(output_folder, folder_name, 'XGB')
xgboost_classifier(X_train, y_train, X_test, y_test, param_grid_XGB, output_path_XGB, seed=SEED)


param_grid_RF = {
        "bootstrap": [True],
        "max_features": ['sqrt', 'log2'],
        "min_samples_leaf": [1, 4, 8, 12, 18, 20],
        "min_samples_split": [3, 5, 8, 10, 12, 15],
        "max_depth": [3, 5, 8, 10],
        "n_estimators": [20, 50, 80, 100, 120, 150, 200, 400],
}

output_path_RF = os.path.join(output_folder, folder_name, 'RF')
rf_classifier(X_train, y_train, X_test, y_test, param_grid_RF, output_path_RF, seed=SEED)
