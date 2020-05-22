import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import analyzer.loaders.hartford.hartford as hartford
from sklearn.impute import KNNImputer
import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
import analyzer.dataset as ds
import analyzer.optuna as o
from analyzer.utils import impute_missing, train_and_evaluate

jobid = os.getenv('SLURM_ARRAY_TASK_ID')
jobid = int(jobid)
print('Jobid = ', jobid)

SEED = 1
prediction = 'Outcome'
folder_name = 'complete_lab_tests_seed' + str(SEED) + '_' + prediction.lower() + '_jobid_' + str(jobid)
output_folder = 'predictors/outcome'

print('Hartford Other + Spain + Italy')
name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])

extra_data = False
demographics_data = True

if jobid == 0:
    o2_col = 'ABG: Oxygen Saturation (SaO2)'
    discharge_data = True
    comorbidities_data = True
    vitals_data = True
    lab_tests = True
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])

elif jobid == 1:
    o2_col = 'SaO2'
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

data_hartford = hartford.load_hartford('/nfs/sloanlab003/projects/cov19_calc_proj/hartford/hhc_inpatient_other.csv', 
  discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)

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

X_hartford, y_hartford =  ds.create_dataset(data_hartford,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      swabs_data,
                                      prediction = prediction)


# Merge dataset
X = pd.concat([X_cremona, X_spain, X_hartford], join='inner', ignore_index=True)
y = pd.concat([y_cremona, y_spain, y_hartford], ignore_index=True)
X, bounds_dict = ds.filter_outliers(X, filter_lb = 1.0, filter_ub = 99.0, o2 = o2_col)

# Shuffle
np.random.seed(SEED)
idx = np.arange(len(X)); np.random.shuffle(idx)
X = X.loc[idx]
y = y.loc[idx]

if jobid == 0:
    X = X[u.SPANISH_ITALIAN_DATA] 

if jobid == 1:
    X = X.drop(['Systolic Blood Pressure', 'Essential hypertension'], axis = 1)

seed = 30
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.1, random_state = seed)
X_train = impute_missing(X_train)

# Train XGB
# algorithm = o.algorithms[0]
# name_param = o.name_params[0]

# best_xgb, best_params = o.optimizer(algorithm, name_param, X_train, y_train, n_calls = 500, name_algo = 'xgboost')

# Train RF
# algorithm = o.algorithms[1]
# name_param = o.name_params[1]

# best_rf, best_params = o.optimizer(algorithm, name_param, X_train, y_train, n_calls = 500, name_algo = 'rf')

# Train CART
algorithm = o.algorithms[2]
name_param = o.name_params[2]

best_cart, best_params = o.optimizer(algorithm, name_param, X_train, y_train, n_calls = 500, name_algo = 'cart')

# Train Logistic regression
# algorithm = o.algorithms[3]
# name_param = o.name_params[3]

# best_lr, best_params = o.optimizer(algorithm, name_param, X_train, y_train, n_calls = 500, name_algo = 'lr')

# Train OCT
# from julia.api import Julia
# jl = Julia(compiled_modules=False)
# from interpretableai import iai

# algorithm = iai.OptimalTreeClassifier
# name_param = o.name_params[4]

# best_oct, best_params = o.optimizer(algorithm, name_param, X_train, y_train, cv = 40, n_calls = 300, name_algo = 'oct')


X_test = impute_missing(X_test)

best_model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params)

print(algorithm)