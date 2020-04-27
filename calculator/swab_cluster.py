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


jobid = os.getenv('SLURM_ARRAY_TASK_ID')
jobid = int(jobid)

print('Jobid = ', jobid)

SEED = 1

prediction = 'Swab'
folder_name = 'swab_prediction_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/swab'

name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'anagraphics', 'swab'])

if jobid == 0:
    discharge_data = False
    comorbidities_data = False
    vitals_data = True
    lab_tests = True
    anagraphics_data = True
    swabs_data = True
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])

elif jobid == 1:
    discharge_data = False
    comorbidities_data = False
    vitals_data = True
    lab_tests = False
    anagraphics_data = True
    swabs_data = True
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])


elif jobid == 2:
    discharge_data = False
    comorbidities_data = False
    vitals_data = False
    lab_tests = True
    anagraphics_data = True
    swabs_data = True
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])

# Load cremona data
data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data)

# Create dataset
X, y = create_dataset(data,
                        discharge_data, 
                        comorbidities_data, 
                        vitals_data, 
                        lab_tests, 
                        anagraphics_data, 
                        swabs_data,
                        prediction = prediction)

algorithm = o.algorithms[0]
name_param = o.name_params[0]

best_xgb = o.optimizer(algorithm, name_param, X, y, seed_len = 20, n_calls = 500, name_algo = 'xgboost')


# Train trees
# output_path = os.path.join(output_folder, folder_name, 'oct')
# create_dir(output_path)
# oct_scores = train_oct(X_train, y_train, X_test, y_test, output_path, seed=SEED)


#PARAMETERS GRID
# param_grid_XGB = {
#         "n_estimators": [20, 50, 80, 100, 120, 150, 200],
#         "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#         "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#         "min_child_weight" : [ 1, 3, 5, 7 ],
#         "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#         "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


# output_path_XGB = os.path.join(output_folder, folder_name, 'XGB')
# xgboost_classifier(X_train, y_train, X_test, y_test, param_grid_XGB, output_path_XGB, seed=SEED)


# param_grid_RF = {
#         "bootstrap": [True],
#         "max_features": ['sqrt', 'log2'],
#         "min_samples_leaf": [1, 4, 8, 12, 18, 20],
#         "min_samples_split": [3, 5, 8, 10, 12, 15],
#         "max_depth": [3, 5, 8, 10],
#         "n_estimators": [20, 50, 80, 100, 120, 150, 200, 400],
# }

# output_path_RF = os.path.join(output_folder, folder_name, 'RF')
# rf_classifier(X_train, y_train, X_test, y_test, param_grid_RF, output_path_RF, seed=SEED)
