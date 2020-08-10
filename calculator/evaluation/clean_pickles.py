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

algorithm_list = ['lr','rf','cart','xgboost','qda','gb'] #exclude OCT
prediction_list = ['COMORB_DEATH']

for outcome in prediction_list:
    for alg in algorithm_list:
        for f in os.listdir(os.path.join(path,outcome,alg)):
            file_name = os.path.join(path,outcome,alg,f)
            t_name =  f[:(f.find(treatment)+len(treatment))]
            with open(file_name, 'rb') as file:
                model_file = pickle.load(file)
            del model_file['train']
            del model_file['test']
            with open(os.path.join(path,outcome,'website_pickles',t_name+'_'+alg+'.pkl'), 'wb') as handle:
                pickle.dump(model_file, handle, protocol=4)
            # importance = pd.concat([pd.Series(model_file['train'].columns[:-1]),
            #                        pd.Series(model_file['model'].feature_importances_)],axis=1)
            
            
## Check files
file_name = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/website/assets/treatment_calculators/COMORB_DEATH/CORTICOSTEROIDS_cart.pkl'
with open(file_name, 'rb') as file:
    model_file = pickle.load(file)