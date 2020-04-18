import pandas as pd
import numpy as np
#  import matplotlib.pylab as plt
import os

#Julia
#  from julia.api import Julia
#  jl = Julia(compiled_modules=False)
#  from interpretableai import iai

from sklearn.model_selection import train_test_split

# Other packages
#  import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from analyzer.loaders.cremona import load_cremona
from analyzer.dataset import create_dataset
from analyzer.utils import create_dir, export_features_json
from analyzer.learners import train_oct
from analyzer.learners import xgboost_classifier
from analyzer.learners import rf_classifier


SEED = 1
prediction = 'outcome'
folder_name = 'cv10_script_complete_tuning_seed' + str(SEED) + '_' + prediction
output_folder = 'predictors/outcome'

# Load cremona data
data = load_cremona('../data/cremona/')

# Create dataset
X, y = create_dataset(data, prediction = prediction)
X = X[X.columns[2:]]

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1,
                                                     random_state=SEED)




# Train trees
output_path = os.path.join(output_folder, 'trees', folder_name)
create_dir(output_path)
oct_scores = train_oct(X_train, y_train, X_test, y_test, output_path, seed=SEED)


#PARAMETERS GRID
param_grid_XGB = {
        "n_estimators": [20, 50, 80, 100, 120, 150, 200],
        "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
        "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight" : [ 1, 3, 5, 7 ],
        "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
        "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

#  param_grid_XGB = {
#          "n_estimators": [20],
#          "learning_rate"    : [0.05] ,
#          "max_depth"        : [ 3],
#          "min_child_weight" : [ 1 ],
#          "gamma"            : [ 0.0],
#          "colsample_bytree" : [ 0.3] }



output_path_XGB = os.path.join(output_folder, 'XGB', folder_name)
xgboost_classifier(X_train, y_train, X_test, y_test, param_grid_XGB, output_path_XGB, seed=SEED)


param_grid_RF = {
        "bootstrap": [True],
        "max_features": ['sqrt', 'log2'],
        "min_samples_leaf": [1, 4, 8, 12, 18, 20],
        "min_samples_split": [3, 5, 8, 10, 12, 15],
        "max_depth": [3, 5, 8, 10],
        "n_estimators": [20, 50, 80, 100, 120, 150, 200],
}

#  param_grid_RF = {
#          "bootstrap": [True],
#          "max_features": ['sqrt'],
#          "min_samples_leaf": [1, ],
#          "min_samples_split": [3],
#          "max_depth": [3 ],
#          "n_estimators": [20],
#  }


output_path_RF = os.path.join(output_folder, 'RF', folder_name)
rf_classifier(X_train, y_train, X_test, y_test, param_grid_RF, output_path_RF, seed=SEED)
