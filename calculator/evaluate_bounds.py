import numpy as np
import pandas as pd
import os
import pickle

# Other packages
import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json, change_SaO2
import analyzer.dataset as ds

import shap
import matplotlib.pyplot as plt

#%% ## Load Model and Data
# Select model type.  
# - **model_type** = 'mortality' or 'infection'
# - **model_lab** = 'with_lab' or 'without_lab'


model_type = 'infection'
model_lab = 'with_lab'

assert model_type in('mortality','infection'), "Invalid outcome"
assert model_lab in('with_lab','without_lab'), "Invalid lab specification"

#%% Set paths
website_path = '/Users/hollywiberg/git/website/'
path_cremona = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_cremona/data/'
path_hm = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_hmfoundation/'

#%% Load model corresponding to *model_type* and *model_lab*.

with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
    model_file = pickle.load(file)
    
model = model_file['model']
features = model_file['json']
columns = model_file['columns']
imputer= model_file['imputer']

#%% Load data: to be replaced once we store X_train and X_test. Currently no imputation

SEED = 1

## Variables determined by input
prediction = 'Outcome'if model_type == 'mortality' else 'Swab'
swabs_data = False if model_type == 'mortality' else True
comorbidities_data = True if model_type == 'mortality' else False
discharge_data = True if model_type == 'mortality' else False 
lab_tests = True if model_lab == 'with_lab' else False

# ## Constant variables
extra_data = False
demographics_data = True
vitals_data = True

name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
print(name_datasets[mask])

## Load Cremona data
data = cremona.load_cremona(path_cremona, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
X_cremona, y_cremona = ds.create_dataset(data, discharge_data, comorbidities_data, vitals_data,
                                      lab_tests, demographics_data, swabs_data, prediction = prediction)

if model_type == "mortality":
    ## Load Spain data
    data_spain = hmfundacion.load_fundacionhm(path_hm, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)
    X_spain, y_spain =  ds.create_dataset(data_spain, discharge_data, comorbidities_data, vitals_data,
                                          lab_tests, demographics_data, swabs_data, prediction = prediction)

    # Merge datasets, filter outliers, match format of stored model
    X0 = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
    y = pd.concat([y_cremona, y_spain], ignore_index=True)
else: 
    X0, y = X_cremona, y_cremona

#%% Process data and impute

X_bounds = np.transpose(ds.evaluate_bounds(X0, [0,.1,1,5,25,50,75,95,99,99.9,100]))
X_bounds.to_csv("/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_clean_data/ranges_"+model_type+"_"+model_lab+".csv")

# X0, bounds_dict = ds.filter_outliers(X0)
# X0 = X0[columns] 

# X = pd.DataFrame(imputer.transform(X0))
# X.columns =  X0.columns
