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
prediction = 'Outcome'
folder_name = 'complete_lab_tests_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/outcome'

name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'anagraphics', 'swab'])

if jobid == 0:
    discharge_data = True
    comorbidities_data = True
    vitals_data = True
    lab_tests = True
    anagraphics_data = False
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])

elif jobid == 1:
    discharge_data = True
    comorbidities_data = False
    vitals_data = True
    lab_tests = True
    anagraphics_data = False
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])


elif jobid == 2:
    discharge_data = True
    comorbidities_data = False
    vitals_data = False
    lab_tests = True
    anagraphics_data = False
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])


elif jobid == 3:
    discharge_data = True
    comorbidities_data = True
    vitals_data = False
    lab_tests = False
    anagraphics_data = False
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])

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

print('The number of NaN is ', sum(X.isna().sum()))

algorithm = o.algorithms[0]
space = o.spaces[0]
name_param = o.name_params[0]

best_xgb = o.optimizer(algorithm, space, name_param, X, y, n_calls = 400)

