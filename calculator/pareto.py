#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:12:55 2020

@author: hollywiberg
"""

import os

# os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

#%% Load data and prescriptions

version = 'matched_single_treatments_der_val_addl_outcomes/'
data_path = '../../covid19_treatments_data/'+version
results_path = '../../covid19_treatments_results/'+version
        
matched = True
match_status = 'matched' if matched else 'unmatched'
weighted_status = 'weighted'

treatment = 'ACEI_ARBS'
outcome = 'COMORB_DEATH'
data_version = 'test'

#%% Load 
version_folder = str(treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

res = pd.read_csv(save_path+match_status+'_metrics_summary.csv')
res['pr_avg'] = res.apply(lambda row: (row['pr_low'] + row['pr_high'])/2, axis=1)
res.loc[res['data_version']=='test']

res_plot = res.loc[(res['data_version']==data_version) & (res['weighted_status']==weighted_status)]

plt.close()
fig, ax = plt.subplots()

x = res_plot['presc_count'].tolist()
y = res_plot['pr_avg'].tolist()
y_bar =  abs(res_plot['pr_high']-y) # add min.max indicators
label = res_plot['threshold'].tolist()
# scatter = ax.scatter(x, y, c=label)
scatter = ax.errorbar(x, y, yerr = y_bar, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0)
for i, txt in enumerate(res_plot['threshold']):
    ax.annotate(txt, (x[i],y[i]))
ax.set_title('Prescription Tradeoffs by Threshold')
ax.set_ylabel('PR Range')
ax.set_xlabel('Number of Prescriptions ('+treatment+')')

plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_pareto.png',
            bbox_inches = "tight")

