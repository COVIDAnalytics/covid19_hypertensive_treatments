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

path = '../../../Data'

train = pd.read_csv(path + '/train.csv', index_col=0)
test = pd.read_csv(path + '/test.csv', index_col=0)

X_train = train[train.columns[:-1]]
y_train = train[train.columns[-1]]
X_test = test[test.columns[:-1]]
y_test = test[test.columns[-1]]

jobid = os.getenv('SLURM_ARRAY_TASK_ID')
jobid = int(jobid)
print('Jobid = ', jobid)

if jobid == 0:
    cv = 5

if jobid == 1:
    cv = 10

if jobid == 2:
    cv = 20

if jobid == 3:
    cv = 50

if jobid == 4:
    cv = 100

if jobid == 5:
    cv = 150

if jobid == 6:
    cv = 200

if jobid == 7:
    cv = 250

if jobid == 8:
    cv = 300

if jobid == 9:
    cv = 350

if jobid == 10:
    cv = 400

if jobid == 11:
    cv = 450

if jobid == 12:
    cv = 500

if jobid == 13:
    cv = 30

if jobid == 14:
    cv = 70

if jobid == 15:
    cv = 220

if jobid == 16:
    cv = 120

if jobid == 17:
    cv = 270

if jobid == 18:
    cv = 320

if jobid == 19:
    cv = 370

if jobid == 20:
    cv = 420

# Train XGB
algorithm = o.algorithms[0]
name_param = o.name_params[0]

best_xgb, best_params = o.optimizer(algorithm, name_param, X_train, y_train, cv = cv, n_calls = 500, name_algo = 'xgboost')

best_model, accTrain, accTest, isAUC, ofsAUC = train_and_evaluate(algorithm, X_train, X_test, y_train, y_test, best_params)