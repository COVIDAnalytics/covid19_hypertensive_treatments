#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:57:19 2020

@author: hollywiberg
"""

#%% Prepare environment
import os

os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import itertools
# import evaluation_utils as u
# import evaluation.importance as imp
import matplotlib.pyplot as plt
import analyzer.optuna as o
import analyzer.dataset as ds
import pickle
import pandas as pd


#%% Set Problem Parameters
#Paths for data access
jobid = 1

def load_data(data_path, split, seed = 1):
    data = pd.read_csv(data_path)

    X_hope, y_hope = ds.create_dataset_treatment(data, treatment = 'All')
    Z_hope = X_hope['REGIMEN']
    X_hope = X_hope.drop('REGIMEN', axis = 1)
    
    # Merge dataset
    X = pd.concat([X_hope], join='inner', ignore_index=True)
    Z = pd.concat([Z_hope], ignore_index=True)
    y = pd.concat([y_hope], ignore_index=True)
    
    #One hot encoding
    X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

    if split == 'bycountry':
        #Split by country and then remove all columns related to country
        train_inds = X['COUNTRY_SPAIN'] == 1
        
    filter_col = [col for col in X if col.startswith('COUNTRY')]
    regimen_col = [col for col in X if col.startswith('REGIMEN')]
        
    X_train = X.loc[train_inds,].drop(filter_col, axis=1)
    Z_train = Z[train_inds]
    y_train = y[train_inds]
    X_test = X.loc[-train_inds,].drop(filter_col, axis=1)
    y_test = y[-train_inds]
    Z_test = Z[-train_inds]
    
    return X_train, Z_train, y_train, X_test, Z_test, y_test
        
data_path = '../../covid19_hope/hope_matched.csv'
X_train, Z_train, y_train, X_test, Z_test, y_test = load_data(data_path, split = 'bycountry')

#%% Generate predictions

jobid = 1

SEED = 1
dataset = "hope"
#Split type
split_type = 0
split_types = ["bycountry","random"]

prediction = 'DEATH'
treatment_list = ['Chloroquine_Only', 'All', 'Chloroquine_and_Anticoagulants',
        'Chloroquine_and_Antivirals', 'Non-Chloroquine']

preds = as.d

algorithm_list = range(0,len(o.algorithms))

param_list = list(itertools.product(treatment_list, algorithm_list))

treatment, algorithm_id = param_list[jobid]
algorithm = o.algorithms[algorithm_id]
name_param = o.name_params[algorithm_id]
name_algo = o.algo_names[algorithm_id]

t = treatment.replace(" ", "_")
result_path = '../../covid19_treatments_results/' + str(name_algo) +'/' + str(dataset)+'_results_treatment_'+str(t)+'_seed' + str(SEED) + '_split_' + str(split_types[split_type]) + '_' + prediction.lower() + '_jobid_' + str(jobid)

def generate_preds(X, result_path): 
    ## Results path and file names
    # output_folder = 'predictors/treatment_mortality'
    
    with open(result_path, 'rb') as file:
          model_file = pickle.load(file)
    
    model = model_file['model']
    prob_pos = model.predict_proba(X)[:, 1]
    
    return prob_pos

prob  = generate_preds(X_train, result_path)

    
    
    
#     model.keys()


# # validation_paths=['../../covid19_greece/general_greek_registry.csv',
# #                   '../../covid19_sevilla/sevilla_clean.csv']

# # validation_paths={'Hellenic CSG':'../../covid19_greece/general_greek_registry.csv',
# #                   'Sevilla':'../../covid19_sevilla/sevilla_clean.csv',
# #                    'Hartford': '../../covid19_hartford/predictions/main'}

# # #Select the model type
# # model_types = ['mortality']
# # model_labs = ['with_lab']
# # seeds = list(range(30, 31))

# # #Extract the seed

# # SEED = [30]
# # SPINE_COLOR = 'gray'
# # model_type = 'mortality'
# # threshold_list = [0.8,0.9]
# # confidence_level = 0.95


# # PAPER_TYPE = 'MORTALITY'  # OR PNAS