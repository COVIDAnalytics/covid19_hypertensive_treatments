df#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 23:48:10 2020

@author: hollywiberg
"""

#%% Load Functions
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split

# Other packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json, change_SaO2
import analyzer.dataset as ds

import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

#%% Specify paths
website_path = '/Users/hollywiberg/git/website/'
path_cremona = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_cremona/data/'
path_hm = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_hmfoundation/'

#%% Load Data
model_type = 'infection'
model_lab = 'with_lab'
prediction = 'Outcome'

with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
    model_file = pickle.load(file)
    
model = model_file['model']
features = model_file['json']
columns = model_file['columns']


#%% Larger dataset

SEED = 1

if model_lab == 'with_lab':
    lab_tests = True
elif model_lab == 'without_lab':
    lab_tests = False
    
extra_data = False
demographics_data = True
discharge_data = True
comorbidities_data = True
vitals_data = True
swabs_data = False

name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
print(name_datasets[mask])

## Load Cremona data
data = cremona.load_cremona(path_cremona, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
X_cremona, y_cremona = ds.create_dataset(data, discharge_data, comorbidities_data, vitals_data,
                                      lab_tests, demographics_data, swabs_data, prediction = prediction)

## Load Spain data
data_spain = hmfundacion.load_fundacionhm(path_hm, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)
X_spain, y_spain =  ds.create_dataset(data_spain, discharge_data, comorbidities_data, vitals_data,
                                      lab_tests, demographics_data, swabs_data, prediction = prediction)

# Merge datasets, filter outliers, match format of stored model
X = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
y = pd.concat([y_cremona, y_spain], ignore_index=True)

X, bounds_dict = ds.filter_outliers(X)
X = X[columns] 


#%% Evaluate Performance by Threshold

pred_Y = model.predict_proba(X)[:, 1]

is_fpr, is_tpr, thresh = roc_curve(y, pred_Y)

accuracy_scores = []
for t in thresh:
    accuracy_scores.append(accuracy_score(y, 
                                         [1 if m > t else 0 for m in pred_Y]))
 
#Create a DataFrame
df = pd.DataFrame({
    'Threshold': thresh,
    'Accuracy': accuracy_scores,
    'Specificity': 1-is_fpr,
    "Sensitivity": is_tpr})
 
print("\nMaximum Threshold for Sensitivity > 0.8: ")
print(df.loc[df['Sensitivity'] > 0.8].iloc[0])

# Influenza, Respiratory Frequency


