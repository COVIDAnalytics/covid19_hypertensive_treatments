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
import math

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

hope_proc = hope_proc.join(hope_pres.drop(['ID', 'COMORB_DEATH','REGIMEN'], axis=1))


hope_orig['DT_HOSPITAL_ADMISSION'] = pd.to_datetime(hope_orig['DT_HOSPITAL_ADMISSION2']).dt.date
hope_proc['DT_HOSPITAL_ADMISSION'] = pd.to_datetime(hope_proc['DT_HOSPITAL_ADMISSION']).dt.date
hope_orig['DT_USE_CORTICOIDS'] = pd.to_datetime(hope_orig['DT_USE_CORTICOIDS2']).dt.date


hope_orig['Date_Diff'] = (hope_orig['DT_USE_CORTICOIDS']  - hope_orig['DT_HOSPITAL_ADMISSION'])/np.timedelta64(1,'D')

hope_orig.loc[hope_orig.Date_Diff < 0, 'Date_Diff'] = math.nan
hope_orig.loc[hope_orig.Date_Diff > 30, 'Date_Diff'] = math.nan
hope_orig.hist(column='Date_Diff')


hope_orig[['DEATH']==1]

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

data = hope_orig[hope_orig['DEATH'] == 1].dropna(thresh=2).Date_Diff
data = data[data.notnull()]

plt.hist(data, weights=np.ones(len(data)) / len(data))

plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

hope_orig.groupby('Date_Diff', as_index=False)['DEATH'].mean()

df.groupby('StationID', as_index=False)['BiasTemp'].mean()


data[data.notnull()]

rslt_df = hope_orig[hope_orig['DEATH'] == 0] 
rslt_df.hist(column='Date_Diff')


.hist(column='Date_Diff')

hope_orig.hist(column='Date_Diff')



df = pd.merge(hope_proc, hope_orig[['DT_USE_CORTICOIDS2','DT_HOSPITAL_ADMISSION','INV_COUNTRY1','INV_HOSPITAL1','EDAD','CO_GENDER','CO_RACE','DEATH']],  how='left', left_on=['COUNTRY','HOSPITAL','AGE','GENDER','RACE','DEATH','DT_HOSPITAL_ADMISSION'], right_on = ['INV_COUNTRY1','INV_HOSPITAL1','EDAD','CO_GENDER','CO_RACE','DEATH','DT_HOSPITAL_ADMISSION'])

df['DT_USE_CORTICOIDS'] = pd.to_datetime(df['DT_USE_CORTICOIDS2']).dt.date

# Create a column which shows the difference in dates of admission for corticosteroids prescription

df['Date_Diff'] = (df['DT_USE_CORTICOIDS']  - df['DT_HOSPITAL_ADMISSION'])/np.timedelta64(1,'D')

df.hist(column='Date_Diff')

#Remove duplicate rows
#df = df.drop_duplicates(subset=['DT_HOSPITAL_ADMISSION','INV_COUNTRY1','INV_HOSPITAL1','EDAD','CO_GENDER','CO_RACE','DEATH'], keep="first")





