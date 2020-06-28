import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split


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
version = "hope"
data_path = "../../covid19_hope/hope_matched.csv"
deriv_countries = ['SPAIN']

SEED = 1

prediction = 'DEATH'
treatment_list = ['Chloroquine Only', 'All', 'Chloroquine and Anticoagulants',
       'Chloroquine and Antivirals', 'Non-Chloroquine']
split_list = ["random"]  #["bycountry","random"]

param_list = list(itertools.product(treatment_list, algorithm_list, split_list))

treatment, name_algo, split_type = param_list[jobid]
print("Treatment = ", treatment, "; Algorithm = ", name_algo, "; Split = ", split_type)
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
file_name = version + '_' + prediction.lower() + '_split' + split_type+ '_seed' + str(SEED) + '_results_'  + str(t)
# file_name = str(dataset)+'_results_treatment_'+str(t)+'_seed' + str(SEED) + '_split_' + str(split_types[split_type]) + '_' + prediction.lower() + '_jobid_' + str(jobid)
# output_folder = 'predictors/treatment_mortality'
results_folder = '../../covid19_treatments_results/' + str(name_algo) +'/'


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

data = pd.read_csv(data_path)

X_hope, y_hope = ds.create_dataset_treatment(data, 
                        treatment,
                        demographics,
                        comorbidities,
                        vitals,
                        lab_tests,
                        med_hx,
                        other_tx,
                        prediction = prediction)

# Merge dataset
X = pd.concat([X_hope], join='inner', ignore_index=True)
y = pd.concat([y_hope], ignore_index=True)

#One hot encoding
deriv_inds = pd.Series([x in deriv_countries for x  in X['COUNTRY']])
X.drop('COUNTRY',axis=1)
X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

if split_type == 'bycountry':
  #Split by country, no randomization
  X_deriv, y_deriv = X.loc[deriv_inds,], y[deriv_inds]
  X_test, y_test = X.loc[-deriv_inds,], y[-deriv_inds]
else:
  # Split derivation country into train/test; ignore external validation sets
  np.random.seed(SEED)
  idx = np.arange(len(X_deriv)); np.random.shuffle(idx)
  X_deriv = X_deriv.loc[idx]
  y_deriv = y_deriv.loc[idx]
  X_train, X_test, y_train, y_test = train_test_split(X_deriv, y_deriv, stratify = y_deriv, test_size=0.1, random_state = SEED)

best_model, best_params = o.optimizer(algorithm, name_param, X_train, y_train, cv = 20, n_calls = 300, name_algo = name_algo)
# X_test = impute_missing(X_test)

# best_model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params)

print(algorithm)

utils.create_and_save_pickle_treatments(algorithm, treatment, SEED, split_type,
                                      X_train, X_test, y_train, y_test, 
                                      best_params, file_name, results_folder,
                                      data_save = True, data_in_pickle = True, json_model = json_format)


