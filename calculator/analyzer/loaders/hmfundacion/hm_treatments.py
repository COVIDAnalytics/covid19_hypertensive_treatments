#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:24:38 2020

@author: agni
"""

import pandas as pd
#  import datetime
import numpy as np
import pickle

import analyzer.loaders.hmfundacion.utils as u
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json

import analyzer.dataset as ds

import analyzer.optimizer as o


SEED = 1
prediction = 'Outcome'
folder_name = 'complete_lab_tests_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/outcome'

discharge_data = True
comorbidities_data = True
vitals_data = True
lab_tests = True
demographics_data = True
swabs_data = False
icu_data = False
extra_data = False

path = '../../COVID_clinical/covid19_hmfoundation/'

#Load spanish data
data_spain = hmfundacion.load_fundacionhm('../../COVID_clinical/covid19_hmfoundation/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)

hm_df = pd.concat([data_spain['discharge'], data_spain['demographics']], axis=1, sort=False)
hm_df['PATIENT ID'] = hm_df.index


#Load the medication file
meds = pd.read_csv('%s/medication.csv' % path, sep=';' , encoding= 'unicode_escape')
med = meds[meds['PATIENT ID'].isin(hm_df['PATIENT ID'])]

agg_med = med.groupby(['PATIENT ID', 'ATC5_NOMBRE/NAME']).size().reset_index(name='counts')

med_wide =  agg_med.pivot(index='PATIENT ID', columns='ATC5_NOMBRE/NAME', values='counts')

na_med =  u.missing_values_table(med_wide)
na_med = na_med[na_med['% of Total Values']<99]

med_wide = med_wide[na_med.index]
med_wide = med_wide.fillna(0)

med_wide[med_wide > 0] = 1
med_descr = med_wide.describe()
med_descr = med_descr.T

med_descr['obs'] = med_descr['mean']*1390

med_descr.to_csv('%s/medication_stats.csv' % path)


com_emerg = pd.read_csv('%s/comorbidities_emerg.csv' % path, sep=';' , encoding= 'unicode_escape')
com_emerg = com_emerg[com_emerg['PATIENT ID'].isin(hm_df['PATIENT ID'])]
com_emerg[com_emerg['PROC_01']=='3E0F7SF']



com_in = pd.read_csv('%s/comorbidities_inpatient.csv' % path, sep=';' , encoding= 'unicode_escape')
com_in = com_in[com_in['PATIENT ID'].isin(hm_df['PATIENT ID'])]
com_in[com_in['PROC_01']=='3E0F7SF']


comorb_list = ['5A09357','3E0F7SF','3E0F73Z','5A09457','5A1935Z','5A1955Z','5A09557','5A1945Z']
columns_list = ['PROC_01', 'PROC_02', 'PROC_03', 'PROC_04', 'PROC_05', 'PROC_06',
       'PROC_07', 'PROC_08', 'PROC_09', 'PROC_10', 'PROC_11', 'PROC_12',
       'PROC_13', 'PROC_14', 'PROC_15', 'PROC_16', 'PROC_17', 'PROC_18',
       'PROC_19', 'PROC_20']

for i in comorb_list:
    n_patients = list()
    print(i)
    for j in columns_list:
       n_patients= n_patients + (list(com_in['PATIENT ID'][com_in[j]==i].unique()))
    print(len(set(n_patients)))
    print(100*len(set(n_patients))/1390)




com_in['PROC_01'].unique()

# 5A09357 - Assistance with Respiratory Ventilation, Less than 24 Consecutive Hours, Continuous Positive Airway Pressure
# 3E0F7SF - 1 - Introduction of Other Gas into Respiratory Tract, Via Natural or Artificial Opening
# 3E0F73Z - Introduction of Anti-inflammatory into Respiratory Tract, Via Natural or Artificial Opening
# 5A09457 - Assistance with Respiratory Ventilation, 24-96 Consecutive Hours, Continuous Positive Airway Pressure
# 5A1935Z - Respiratory Ventilation, Less than 24 Consecutive Hours
# 5A1955Z - Respiratory Ventilation, Greater than 96 Consecutive Hours
# 5A09557 - Assistance with Respiratory Ventilation, Greater than 96 Consecutive Hours, Continuous Negative Airway Pressure
# 5A1945Z - Respiratory Ventilation, 24-96 Consecutive Hours







