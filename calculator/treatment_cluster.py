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
  algorithm_list = sys.argv[1]
  print("Algorithms: ", algorithm_list)
  print("Valid Algorithms: ", o.algorithms)
except:
  print("Must provide algorithm list")

assert set(algorithm_list).issubset(set(o.algo_names)), "Invalid algorithm list"


SEED = 1
#Define the name of the dataset for sacing the results
dataset = "hope"
#Split type
split_type = 0
split_types = ["bycountry","random"]

prediction = 'DEATH'
treatment_list = ['Chloroquine Only', 'All', 'Chloroquine and Anticoagulants',
       'Chloroquine and Antivirals', 'Non-Chloroquine']

param_list = list(itertools.product(treatment_list, algorithm_list))

treatment, name_algo = param_list[jobid]

if 'oct' == name_algo:
  from julia import Julia
  jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
  from interpretableai import iai
  o.algorithms['oct'] = iai.OptimalTreeClassifier

name_param = o.name_params[name_algo]
algorithm = o.algorithms[name_algo]

## Results path and file names
t = treatment.replace(" ", "_")
file_name = str(dataset)+'_results_treatment_'+str(t)+'_seed' + str(SEED) + '_split_' + str(split_types[split_type]) + '_' + prediction.lower() + '_jobid_' + str(jobid)
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

data = pd.read_csv("../../covid19_hope/hope_matched.csv")

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
X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

#X, bounds_dict = ds.filter_outliers(X, filter_lb = 1.0, filter_ub = 99.0, o2 = "")

# store_json(bounds_dict, 'treatment_bounds.json')

## HMW: FOR NOW, deterministic split
# Shuffle
# np.random.seed(SEED)
# idx = np.arange(len(X)); np.random.shuffle(idx)
# X = X.loc[idx]
# y = y.loc[idx]

# seed = 30
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
# X_train = impute_missing(X_train)

#Split by country and then remove all columns related to country
train_inds = X['COUNTRY_SPAIN'] == 1
filter_col = [col for col in X if col.startswith('COUNTRY')]

X_train, y_train = X.loc[train_inds,].drop(filter_col, axis=1), y[train_inds]
X_test, y_test = X.loc[-train_inds,].drop(filter_col, axis=1), y[-train_inds]


#Comment the missing data imputation
#X_train = impute_missing(X_train)

# Train model
## HMW: get error that y_true is always 0 so can't calculate ROC
best_model, best_params = o.optimizer(algorithm, name_param, X_train, y_train, cv = 20, n_calls = 50, name_algo = name_algo)

# X_test = impute_missing(X_test)

best_model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params)

print(algorithm)

utils.create_and_save_pickle_treatments(algorithm, treatment, SEED, split_type,
                                      X_train, X_test, y_train, y_test, 
                                      best_params, file_name, results_folder,
                                      data_save = True, data_in_pickle = True, json_model = True)
        





