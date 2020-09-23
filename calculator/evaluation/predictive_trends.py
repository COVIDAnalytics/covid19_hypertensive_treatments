#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:59:44 2020

@author: agni
"""

import evaluation.treatment_utils as  u
import evaluation.descriptive_utils as d
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import matplotlib.pyplot as plt
import shap
import pickle
import matplotlib
import os


treatments = ['CORTICOSTEROIDS','ACEI_ARBS','INTERFERONOR']
main_treatment = 'ACEI_ARBS'

data_path = '../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'
outcome = 'COMORB_DEATH'



preload = True
matched = True
match_status = 'matched' if matched else 'unmatched'

SEEDS = range(1, 2)

#%% Generate predictions across all combinations
 #['CORTICOSTEROIDS', 'INTERFERONOR', 'ACEI_ARBS']

treatment = 'ACEI_ARBS'
treatment_list = [treatment, 'NO_'+treatment]

results_path = '../../covid19_treatments_results/'
#Set a seed
SEED=1


#Number of features to present
top_features=10
shap_algorithm_tree_list = ['rf','cart','xgboost']
shap_algorithm_list  = ['lr','rf','cart','qda','gb','xgboost']
training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'


#%%Load the corresponding model
for algorithm in shap_algorithm_list:
    for treatment in treatment_list:
        
        version_folder = 'matched_single_treatments_der_val_addl_outcomes/'+str(main_treatment)+'/'+str(outcome)+'/'
        save_path = results_path + version_folder + 'summary/'
        result_path = results_path+version_folder+algorithm+'/'
        file_start = str(treatment) + '_' + match_status + '_' + outcome.lower() + '_seed' + str(SEED)
        file_name = result_path+file_start
        save_file_name = save_path+file_start+'_'+algorithm+'_shap_values.csv'

        df = u.calculate_average_shap(file_name, treatment, outcome, algorithm, top_features)   
        df.to_csv(save_file_name, index = False)

