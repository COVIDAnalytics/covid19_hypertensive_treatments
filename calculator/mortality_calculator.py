import numpy as np
import pandas as pd
import os


from sklearn.model_selection import train_test_split

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion

from analyzer.dataset import create_dataset

import analyzer.optimizer as o

SEED = 1
prediction = 'Outcome'
folder_name = 'complete_lab_tests_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/outcome'

discharge_data = True
comorbidities_data = True
vitals_data = True
lab_tests = True
demographics_data = True
swabs_data = False
icu_data = False
extra_data = True

# Load cremona data
data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)

#Load spanish data
data_spain = hmfundacion.load_fundacionhm('../data/spain/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)


X_cremona, y_cremona = create_dataset(data,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      swabs_data,
                                      prediction = prediction)

X_spain, y_spain =   create_dataset(data_spain,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      swabs_data,
                                      prediction = prediction)


# Merge dataset
X = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
y = pd.concat([y_cremona, y_spain], ignore_index=True)

# Shuffle
np.random.seed(SEED)
idx = np.arange(len(X)); np.random.shuffle(idx)
X = X.loc[idx]
y = y.loc[idx]

# Train
algorithm = o.algorithms[0]
name_param = o.name_params[0]

best_xgb = o.optimizer(algorithm, name_param, X, y, seed_len = 10, n_calls = 400, name_algo = 'xgboost')

