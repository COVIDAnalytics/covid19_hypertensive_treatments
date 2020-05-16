import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
import analyzer.dataset as ds
import analyzer.optimizer as o

jobid = os.getenv('SLURM_ARRAY_TASK_ID')
jobid = int(jobid)
print('Jobid = ', jobid)

SEED = 1
prediction = 'Outcome'
folder_name = 'complete_lab_tests_seed' + str(SEED) + '_' + prediction.lower() + '_jobid_' + str(jobid)
output_folder = 'predictors/outcome'

name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])

extra_data = False
demographics_data = True

if jobid == 0:
    discharge_data = True
    comorbidities_data = True
    vitals_data = True
    lab_tests = True
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])

elif jobid == 1:
    discharge_data = True
    comorbidities_data = True
    vitals_data = True
    lab_tests = False
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])


data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)

#Load spanish data
data_spain = hmfundacion.load_fundacionhm('../data/spain/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)


X_cremona, y_cremona = ds.create_dataset(data,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      swabs_data,
                                      prediction = prediction)

X_spain, y_spain =  ds.create_dataset(data_spain,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      extra_data,
                                      prediction = prediction)


# Merge dataset
X = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
y = pd.concat([y_cremona, y_spain], ignore_index=True)

X, bounds_dict = ds.filter_outliers(X)

# Shuffle
np.random.seed(SEED)
idx = np.arange(len(X)); np.random.shuffle(idx)
X = X.loc[idx]
y = y.loc[idx]

if jobid == 0:
    X = X[u.SPANISH_ITALIAN_DATA] 

if jobid == 1:
    X = X.drop(['Systolic Blood Pressure', 'Essential hypertension'], axis = 1)

# Train XGB
# algorithm = o.algorithms[0]
# name_param = o.name_params[0]

# best_xgb = o.optimizer(algorithm, name_param, X, y, seed_len = 40, n_calls = 450, name_algo = 'xgboost')

# Train CART
algorithm = o.algorithms[2]
name_param = o.name_params[2]

best_cart = o.optimizer(algorithm, name_param, X, y, seed_len = 40, n_calls = 450, name_algo = 'cart')

# Train Logistic regression
# algorithm = o.algorithms[3]
# name_param = o.name_params[3]

# best_lr = o.optimizer(algorithm, name_param, X, y, seed_len = 40, n_calls = 450, name_algo = 'lr')

# Train OCT
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from interpretableai import iai

# algorithm = iai.OptimalTreeClassifier
# name_param = o.name_params[4]

# best_oct = o.optimizer(algorithm, name_param, X, y, seed_len = 40, n_calls = 250, name_algo = 'oct')

print(algorithm)