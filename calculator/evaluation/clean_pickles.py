#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:27:36 2020

@author: hollywiberg
"""

#%% Import packages
import pandas as pd
import numpy as np
import os
import pickle
from sklearn import metrics
from scipy import stats
import math

import analyzer.dataset as ds
import evaluation.treatment_utils as u

#%% Choose variant and specify paths

treatment = 'CORTICOSTEROIDS'
outcome = 'COMORB_DEATH'
path = '../../covid19_treatments_results/matched_single_treatments_der_val_addl_outcomes/'+treatment
website_path = '../../website/assets/treatment_calculators/'+outcome

training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'

X, Z, y = u.load_data('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/',
                       training_set_name, split='train', matched=True, prediction = outcome)

#%% Load pickle for comparison

# with open('../../website/assets/risk_calculators/mortality/model_with_lab.pkl', 'rb') as file:
#     mort_pkl = pickle.load(file)
# json = mort_pkl['json']

#%% Set up feature json

categoric = [{'name':'SAT02_BELOW92',
              'index':[],
              'vals': [0.0, 1.0],
              'default': 0.0,
              'explanation': 'Select if oxygen saturation is below 92'},
             {'name':'BLOOD_PRESSURE_ABNORMAL_B',
              'index':[],
              'vals': [0.0, 1.0],
              'default': 0.0,
              'explanation': 'Select if blood pressure is below X'},
             {'name':'DDDIMER_B',
              'index':[],
              'vals': [0.0, 1.0],
              'default': 0.0,
              'explanation': 'Select if D-Dimer is below X'},
             {'name':'PCR_B',
              'index':[],
              'vals': [0.0, 1.0],
              'default': 0.0,
              'explanation': 'Select if PCR is below X'},
             {'name':'TRANSAMINASES_B',
              'index':[],
              'vals': [0.0, 1.0],
              'default': 0.0,
              'explanation': 'Select if transaminases are below X'},
             {'name':'LDL_B',
              'index':[],
              'vals': [0.0, 1.0],
              'default': 0.0,
              'explanation': 'Select if LDL Cholesterol is below X'},
             {'name': 'GENDER_MALE',
              'index': 1,
              'vals': [0.0, 1.0],
              'default': 0.0,
              'explanation': 'Select if Gender is male'}]
    
multidrop = [
    {'name': 'Comorbidities',
      'index': [],
      'vals': ['DIABETES',
         'HYPERTENSION',
         'DISLIPIDEMIA',
         'OBESITY',
         'RENALINSUF',
         'ANYLUNGDISEASE',
         'AF',
         'VIH',
         'ANYHEARTDISEASE',
         'ANYCEREBROVASCULARDISEASE',
         'CONECTIVEDISEASE',
         'LIVER_DISEASE',
         'CANCER'],
      'explanation': ['Select the existing chronic diseases or conditions.']},
    {'name': 'Treatments',
     'index': [],
     'vals': ['CLOROQUINE',
           'ANTIVIRAL',
           'ANTICOAGULANTS',
           'INTERFERONOR',
           'TOCILIZUMAB',
           'ANTIBIOTICS',
           'ACEI_ARBS',],
     'explanation': ['Select other treatments prescribed.']},
    {'name': 'Race',
     'index': [],
     'vals': ['RACE_CAUC',
           'RACE_LATIN',
           'RACE_ORIENTAL',
           'RACE_OTHER'],
     'explanation': ['Select patient race.']}]

numeric_cols = ['AGE',
    'MAXTEMPERATURE_ADMISSION',
    'CREATININE',
    'SODIUM',
    'LEUCOCYTES',
    'LYMPHOCYTES',
    'HEMOGLOBIN',
    'PLATELETS']


def get_feature_names():
    return {'ACEI_ARBS': 'ACE Inhibitors or ARBs',
     'AF': 'Atrial Fibrllation',
     'AGE': 'Age',
     'ANTIBIOTICS': 'Antibiotics',
     'ANTICOAGULANTS':'Anticoagulants',
     'ANTIVIRAL':'Antivirals',
     'ANYCEREBROVASCULARDISEASE':'Cerebrovascular Disease',
     'ANYHEARTDISEASE':'Heart Disease',
     'ANYLUNGDISEASE':'Lung Disease',
     'BLOOD_PRESSURE_ABNORMAL_B':'Low blood pressure',
     'CANCER':'Cancer',
     'CLOROQUINE':'Hydroxychloroquine',
     'CONECTIVEDISEASE':'Connective Tissue Disease',
     'CREATININE':'Creatinine',
     'DDDIMER_B':'Abnormal D-Dimer',
     'DIABETES':'Diabetes',
     'DISLIPIDEMIA':'Dislipidemia',
     'GENDER_MALE':'',
     'HEMOGLOBIN':'Hemoglobin',
     'HYPERTENSION':'Hypertension',
     'INTERFERONOR':'Interferons',
     'LDL_B':'Abnormal LDL Cholesterol',
     'LEUCOCYTES':'White Blood Cell Count',
     'LIVER_DISEASE':'Liver Disease',
     'LYMPHOCYTES':'Lymphocytes',
     'MAXTEMPERATURE_ADMISSION':'Temperature',
     'OBESITY':'Obesity',
     'PCR_B':'Abnormal PCR',
     'PLATELETS':'Platelets',
     'RACE_CAUC':'Caucasian',
     'RACE_LATIN':'Hispanic',
     'RACE_ORIENTAL':'Asian',
     'RACE_OTHER':'Other',
     'RENALINSUF':'Renal Insufficiency',
     'SAT02_BELOW92':'Low Oxygen Saturation',
     'SODIUM':'Blood Sodium',
     'TOCILIZUMAB':'Tocilizumab',
     'TRANSAMINASES_B':'Abnormal Transaminases',
     'VIH':'HIV'}



#%% Set up variants to iterate

__, bounds_dict = ds.filter_outliers(X[numeric_cols], filter_lb = 0.0, filter_ub = 100.0)

numeric = []
for col in numeric_cols:
    numeric.append({'name':col,
     'index': list(X.columns).index(col),
     'min_val': bounds_dict[col]['min_val'],
     'max_val': bounds_dict[col]['max_val'],
     'default': bounds_dict[col]['default'],
     'explanation': 'Enter value for '+col.lower()})

## Find indices for categorical
for i in range(0,len(categoric)):
    col = categoric[i]['name']
    categoric[i]['index'] = list(X.columns).index(col)

for i in range(0,len(multidrop)):
    col_list = multidrop[i]['vals']
    multidrop[i]['index'] = [list(X.columns).index(col) for col in col_list]
    
        
new_json = {'numeric':numeric,
            'categorical':categoric,
            'multidrop':multidrop}

## Check indices
index_list = []
name_list = []
for k in new_json:
    for i in range(0,len(new_json[k])):
        if k == 'multidrop':
            index_list.extend([x for x in new_json[k][i]['index']])
            name_list.extend([x for x in new_json[k][i]['vals']])
        elif k in ['numeric','categorical']:
            index_list.append(new_json[k][i]['index'])
            name_list.append(new_json[k][i]['name'])
index_list.sort()
name_list.sort()

if np.all(index_list == list(range(0,len(X.columns)))) & np.all(name_list == X.columns.sort_values()):
    web_pkl = {'json':new_json}
else: 
    print("Error: mismatch in indices or feature names. need to debug!")

##
#%% Generate pickle

algorithm_list = ['lr','rf','cart','xgboost','qda','gb'] #exclude OCT

for treated in [True,False]:
    treat_pkl = {}
    for alg in algorithm_list:
        alg_pkl = {}
        for f in os.listdir(os.path.join(path,outcome,alg)):
            # print(f)
            t_name =  f[:(f.find(treatment)+len(treatment))]
            if treated & ('NO_' not in f):
                file_name = os.path.join(path,outcome,alg,f)
            elif (not treated) & ('NO_' in f):
                file_name = os.path.join(path,outcome,alg,f)
            else: 
                file_name = ''
           
            ## Add pickle if needed
            if file_name != '':
                print(f)
                with open(file_name, 'rb') as file:
                    model_file = pickle.load(file)
                alg_pkl['model'] = model_file['model']
                alg_pkl['AUC'] = model_file['AUC']
                alg_pkl['Misclassification'] = model_file['Misclassification']
                cols = model_file['columns']
                if np.any(sorted(cols) != name_list):
                    print("ERROR: columns do not match")
                    break
        treat_pkl[alg] = alg_pkl
    t_key =  'treatment-models' if treated else 'no-treatment-models'
    web_pkl[t_key] = treat_pkl

with open(os.path.join(website_path,treatment+'.pkl'), 'wb') as handle:
    pickle.dump(web_pkl, handle, protocol=4)
                
# # # ## Check files
# file_name = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/website/assets/treatment_calculators/COMORB_DEATH/NO_CORTICOSTEROIDS.pkl'
# with open(file_name, 'rb') as file:
#     model_file = pickle.load(file)