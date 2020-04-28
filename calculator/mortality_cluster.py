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
    comorbidities_data = True
    vitals_data = True
    lab_tests = False
    anagraphics_data = False
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data])
    print(name_datasets[mask])


def load_spanish_data():

    df = pd.read_csv('../../covid19_hmfoundation/fundacionhm_italy_adjusted_clean.csv')

    # Drop duplicates
    df = df.drop_duplicates('PATIENT ID')

    # Clean temperatures
    df['Temperature Celsius'] = df['Temperature Celsius'].apply(lambda x: x.replace(',', '.')).astype(np.float64)
    
    # Change sex
    df['Sex'] = df['Sex'] - 1

    # Adjust wrong diabetes values
    df.loc[df['Diabetes'] == 2, 'Diabetes'] = 1

    # Drop nosologico
    df.drop(['NOSOLOGICO', 'DIAG_TYPE', 'Date_Admission', 'Date_Emergency'],
            axis=1, inplace=True)

    comorbs = ['Acute and unspecified renal failure',
       'Cardiac dysrhythmias', 'Chronic kidney disease',
       'Coronary atherosclerosis and other heart disease',
       'Essential hypertension', 'Diabetes']

    df[comorbs] = df[comorbs].fillna(0)

    # Set index
    df.set_index('PATIENT ID', inplace=True)

    y = df['death']
    X = df.drop('death', axis=1)
    # Impute missing values
    # TODO: Remove horrible import at this line
    from analyzer.loaders.cremona.utils import remove_missing

    X = remove_missing(X, nan_threshold=40)

    return X, y

X_spain, y_spain = load_spanish_data()

# Load cremona data
data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, anagraphics_data, swabs_data)

X_cremona, y_cremona = create_dataset(data,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      anagraphics_data,
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

algorithm = o.algorithms[0]
name_param = o.name_params[0]

best_xgb = o.optimizer(algorithm, name_param, X, y, seed_len = 40, n_calls = 500, name_algo = 'xgboost')