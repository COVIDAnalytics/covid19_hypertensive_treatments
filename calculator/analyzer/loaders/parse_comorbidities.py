#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:16:12 2020

@author: hollywiberg
"""

import os
import pandas as pd
import numpy as np
# os.chdir('Dropbox (MIT)/COVID_risk/covid19_calculator/calculator/analyzer/loaders')

mapping = pd.read_csv('../hcup_dictionary_icd10.csv')

comorb_dict = {'DIABETES': [49, 50, 174],
    'HYPERTENSION':[87, 88, 171],
    'DISLIPIDEMIA':[53],
    'OBESITY':[58],
    'AF':[95],
    'HIV':[5],
    'ANYHEARTDISEASE':[90, 92, 93, 95],
    'CONECTIVEDISEASE':[198, 199],
    'LIVER_DISEASE':[6, 139],
    'CANCER':[11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
              31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
              41, 42, 43],
    'RENALINSUF':[145, 146],
    'ANYLUNGDISEASE':[116, 117, 121, 122],
    'ANYCEREBROVASCULARDISEASE':[98, 100, 101, 102]}
               

df_hcup = pd.DataFrame(pd.DataFrame(list(comorb_dict.values()), 
                                    index = comorb_dict.keys()).stack(), columns = ['HCUP_ORDER']).reset_index()


df_hcup = df_hcup.drop(columns='level_1').rename(columns = {'level_0':'Condition'})

df_icd = df_hcup.merge(mapping.drop(columns='GROUP_HCUP_ID'))[['Condition','HCUP_ORDER','GROUP_HCUP','DIAGNOSIS_CODE','SHORT_DESCRIPTION']]
df_icd.to_csv("comorbidities_icd10_mapping.csv", index = False)

icd_list_all = "'"+"', '".join(str(x) for x in df_icd['DIAGNOSIS_CODE'].unique())+"'"
with open("comorbidities_icd10_list.txt", "w") as text_file:
    text_file.write(icd_list_all)
    
df_icd_nohiv = df_icd.query('Condition != "HIV"')
icd_list_nohiv = "'"+"', '".join(str(x) for x in df_icd_nohiv['DIAGNOSIS_CODE'].unique())+"'"
with open("comorbidities_icd10_list_noHIV.txt", "w") as text_file:
    text_file.write(icd_list_nohiv)
    