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

for outcome in os.listdir(path):
    for alg in os.listdir(os.path.join(path,outcome)):
        for f in os.listdir(os.path.join(path,outcome,alg)):
            file_name = os.path.join(path,outcome,alg,f)
            with open(file_name, 'rb') as file:
                model_file = pickle.load(file)
            # del model_file['train']
            # del model_file['test']
            importance = pd.concat([pd.Series(model_file['train'].columns[:-1]),
                                   pd.Series(model_file['model'].feature_importances_)],axis=1)