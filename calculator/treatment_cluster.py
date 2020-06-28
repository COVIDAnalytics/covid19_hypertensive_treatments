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


## Set up experiment based on job specifications
jobid = os.getenv('SLURM_ARRAY_TASK_ID')
jobid = int(jobid)-1
print('Jobid = ', jobid)

try: 
  algorithm_list = sys.argv[1].split(',')
  print("Algorithms: ", algorithm_list)
  print("Valid Algorithms: ", o.algo_names)
except:
  print("Must provide algorithm list")

#Define the name of the dataset for saving the results
version_folder = "unmatched_and_matched_all_treatments/"
data_path = "../../covid19_treatments_data/"+version_folder

SEED = 1

split_type = 'bycountry'
prediction = 'DEATH'
treatment_list = ['Chloroquine Only', 'All', 'Chloroquine and Anticoagulants',
       'Chloroquine and Antivirals', 'Non-Chloroquine']
match_list = [True,False]

param_list = list(itertools.product(treatment_list, match_list, algorithm_list))

treatment, name_algo, matched = param_list[jobid]
print("Treatment = ", treatment, "; Algorithm = ", name_algo)
if 'oct' == name_algo:
  from julia import Julia
  jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
  from interpretableai import iai
  o.algorithms['oct'] = iai.OptimalTreeClassifier
  json_format = True
else:
  json_format = False

name_param = o.name_params[name_algo]
algorithm = o.algorithms[name_algo]

## Results path and file names
t = treatment.replace(" ", "_")
file_name = str(t) + prediction.lower() + '_seed' + str(SEED) 
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

if matched:
  data_train = pd.read_csv(data_path+'hope_hm_cremona_matched_all_treatments_train.csv')
  data_test = pd.read_csv(data_path+'hope_hm_cremona_matched_all_treatments_test.csv')
else: 
  data_train = pd.read_csv(data_path+'hope_hm_cremona_unmatched_all_treatments_train.csv')
  data_test = pd.read_csv(data_path+'hope_hm_cremona_unmatched_all_treatments_test.csv')

X_train, y_train = ds.create_dataset_treatment(data_train, 
                        treatment,
                        demographics,
                        comorbidities,
                        vitals,
                        lab_tests,
                        med_hx,
                        other_tx,
                        prediction = prediction)

X_test, y_test = ds.create_dataset_treatment(data_test, 
                        treatment,
                        demographics,
                        comorbidities,
                        vitals,
                        lab_tests,
                        med_hx,
                        other_tx,
                        prediction = prediction)

## Need to combine and re-split for consistent one-hot encoding
X_full =  pd.concat([X_train, X_test], axis = 0)
train_inds = pd.Series(range(0,X_train.shape[0]))

X_full = pd.get_dummies(X_full, prefix_sep='_', drop_first=True)
X_train = X_full.iloc[train_inds,:]
X_test = X_full.iloc[-train_inds,:]

best_model, best_params = o.optimizer(algorithm, name_param, X_train, y_train, cv = 20, n_calls = 50, name_algo = name_algo)
# X_test = impute_missing(X_test)

# best_model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params)

print(algorithm)

utils.create_and_save_pickle_treatments(algorithm, treatment, SEED, split_type,
                                      X_train, X_test, y_train, y_test, 
                                      best_params, file_name, results_folder,
                                      data_save = True, data_in_pickle = True, json_model = json_format)


