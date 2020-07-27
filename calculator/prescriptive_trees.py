import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

import analyzer.dataset as ds
import analyzer.optuna as o
from analyzer.utils import impute_missing, train_and_evaluate
import analyzer.utils as utils
import itertools

## Load IAI
from julia import Julia
jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
from interpretableai import iai
json_format = True
name_algo = 'opt'

## Set up experiment based on job specifications
# jobid = os.getenv('SLURM_ARRAY_TASK_ID')
# jobid = int(jobid)-1
# print('Jobid = ', jobid)

version_folder = "matched_all_treatments_der_val_update/"
data_path = "../../covid19_treatments_data/"+version_folder

SEED = 1 
SEED = int(jobid)

prediction = 'DEATH'
## Results path and file names
# file_name = str(dataset)+'_results_treatment_'+str(t)+'_seed' + str(SEED) + '_split_' + str(split_types[split_type]) + '_' + prediction.lower() + '_jobid_' + str(jobid)
# output_folder = 'predictors/treatment_mortality'
results_folder = '../../covid19_treatments_results/' + version_folder + str(name_algo) +'/'
# make folder if it does not exist
Path(results_folder).mkdir(parents=True, exist_ok=True)

## HMW: choose types of columns to include. use all for now (see defined in dataset.py)
name_datasets = np.asarray(['demographics', 'comorbidities', 'vitals', 'lab', 'medical_history', 'other_treatments'])
demographics = True
comorbidities = True
vitals = True
lab_tests=True
med_hx=True
other_tx=True
# mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
# print(name_datasets[mask])

# if matched:
data_train = pd.read_csv(data_path+'hope_hm_cremona_matched_all_treatments_train.csv')
data_test = pd.read_csv(data_path+'hope_hm_cremona_matched_all_treatments_test.csv')
# else: 
# data_train = pd.read_csv(data_path+'hope_hm_cremona_matched_cl_noncl_removed_train.csv')
# data_test = pd.read_csv(data_path+'hope_hm_cremona_matched_cl_noncl_removed_test.csv')
# file_name = str(t) + '_unmatched_' + prediction.lower() + '_seed' + str(SEED) 

X_train, y_train = ds.create_dataset_treatment(data_train, 
                        None, # treatment specification
                        demographics,
                        comorbidities,
                        vitals,
                        lab_tests,
                        med_hx,
                        other_tx,
                        prediction = prediction)

X_test, y_test = ds.create_dataset_treatment(data_test, 
                        None, #treatment specification
                        demographics,
                        comorbidities,
                        vitals,
                        lab_tests,
                        med_hx,
                        other_tx,
                        prediction = prediction)

Z_train = X_train['REGIMEN']
X_train = X_train.drop('REGIMEN', axis=1)

Z_test = X_test['REGIMEN']
X_test = X_test.drop('REGIMEN', axis=1)

X_full =  pd.concat([X_train, X_test], axis = 0)

X_full = pd.get_dummies(X_full, prefix_sep='_', drop_first=True)
X_train = X_full.iloc[0:X_train.shape[0],:]
X_test = X_full.iloc[X_train.shape[0]:,:]

grid = iai.GridSearch(
    iai.OptimalTreePrescriptionMinimizer(
        random_seed=SEED,
    ),
    max_depth=range(3, 6),
    prescription_factor=np.linspace(0.0, 1.0, 11),
)
grid.fit(X_train, Z_train, y_train, validation_criterion='prediction_accuracy')

def evaluate_opt(lnr,X,Z,y):
  preds = lnr.predict_outcomes(X)
  preds['Prescribe'] = lnr.predict(X)[0]
  preds['REGIMEN'] = Z
  preds['DEATH'] = y
  preds['Match'] = preds['Prescribe'] == preds['REGIMEN']
  return preds

lnr =  grid.get_learner()
lnr.write_html(results_folder+'opt.html')

train_preds = evaluate_opt(lnr,X_train, Z_train, y_train)
train_preds.to_csv(results_folder+'train_matched_bypatient_summary_opt.csv')

test_preds = evaluate_opt(lnr,X_test, Z_test, y_test)
test_preds.to_csv(results_folder+'test_matched_bypatient_summary_opt.csv')

data_version = 'validation_cremona'
X, Z, y = u.load_data(data_path,'hope_hm_cremona_matched_cl_noncl_removed_train.csv',
                                split=data_version,matched=matched)

# grid = iai.GridSearch(
#     iai.OptimalTreePrescriptionMinimizer(
#         random_seed=SEED,
#     ),
#     max_depth=range(3, 6),
#     # prescription_factor=np.linspace(0.0, 1.0, 6),
#     treatment_minbucket=[0.01,0.05,0.1,0.2]
# )
# grid.fit_cv(X_train, Z_train, y_train, validation_criterion='combined_performance')

# grid.get_learner()

