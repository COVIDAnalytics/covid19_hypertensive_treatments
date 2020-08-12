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

#%% Set up variants to iterate
treatment = 'CORTICOSTEROIDS'
path = '../../covid19_treatments_results/matched_single_treatments_der_val_addl_outcomes/'+treatment

web_pkl = {'columns':''}
algorithm_list = ['lr','rf','cart','xgboost','qda','gb'] #exclude OCT
outcome = 'COMORB_DEATH'

for treated in [True,False]:
    treat_pkl = {}
    for alg in algorithm_list:
        alg_pkl = {}
        for f in os.listdir(os.path.join(path,outcome,alg)):
            print(f)
            t_name =  f[:(f.find(treatment)+len(treatment))]
            if treated & ('NO_' not in f):
                print('test')
                file_name = os.path.join(path,outcome,alg,f)
            elif (not treated) & ('NO_' in f):
                print('test2')
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
                if web_pkl['columns'] == '':
                    web_pkl['columns'] = cols
                elif web_pkl['columns'] != cols:
                    print("ERROR: columns do not match")
        treat_pkl[alg] = alg_pkl
    t_key =  'treatment-models' if treated else 'no-treatment-models'
    web_pkl[t_key] = treat_pkl

with open(os.path.join(path,outcome,'website_pickles',treatment+'.pkl'), 'wb') as handle:
    pickle.dump(web_pkl, handle, protocol=4)
                
# # # ## Check files
# file_name = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/website/assets/treatment_calculators/COMORB_DEATH/NO_CORTICOSTEROIDS.pkl'
# with open(file_name, 'rb') as file:
#     model_file = pickle.load(file)