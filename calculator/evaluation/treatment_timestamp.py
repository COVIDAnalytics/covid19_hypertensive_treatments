#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:54:02 2020

@author: agni
"""


import os

# os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
import pandas as pd
import numpy as np
from pathlib import Path

#%% Set Problem Parameters - general across treatments

data_path = '../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'
outcome = 'COMORB_DEATH'

matched = True
match_status = 'matched' if matched else 'unmatched'

treatment = 'CORTICOSTEROIDS'
treatment_list = [treatment, 'NO_'+treatment]

results_path = '../../covid19_treatments_results/'
version_folder = 'matched_single_treatments_der_val_addl_outcomes/'+str(treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'

#Pick a dataset - for now it will be just hope

#1. Read the origina HOPE data

hope_orig = pd.read_csv('../../covid19_hope/hope_data.csv')
    
#2. Load the final results for the entire hope dataset
hope_proc = pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_all_treatments_validation_hope.csv')

#3. Load the prescription results 
hope_pres = pd.read_csv(save_path+ 'validation_hope_matched_bypatient_summary_weighted.csv')

#First we need to do a matching for hope_pres and hope_orig
COUNTRY - INV_COUNTRY1
HOSPITAL - INV_HOSPITAL1



