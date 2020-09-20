#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:27:13 2020

@author: hollywiberg
"""


import os

# os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from pathlib import Path


from sklearn.metrics import (brier_score_loss, precision_score, recall_score,accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
from sklearn import metrics

#%% Version information

version = 'matched_single_treatments_der_val_addl_outcomes/'
data_path = '../../covid19_treatments_data/'+version
results_path = '../../covid19_treatments_results/'+version
        
threshold = 0.01
match_status = 'matched' 
weighted_status = 'no_weights'

SEEDS = range(1, 2)
algorithm_list = ['rf','cart','oct','xgboost','qda','gb']
data_list = ['train','test','validation']

outcome = 'COMORB_DEATH'
treatment = 'ACEI_ARBS'
treatment_list = [treatment, 'NO_'+treatment]

training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'

version_folder = str(treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

#%% Predictive Methods table

columns = pd.MultiIndex.from_product([data_list,treatment_list])
index = algorithm_list

res = pd.DataFrame(columns = pd.MultiIndex.from_product([data_list,treatment_list]), 
             index = algorithm_list)

for data_version in data_list:
    pred_results = pd.read_csv(save_path+data_version+'_'+match_status+'_performance_allmethods.csv')
    pred_results.set_index('Algorithm', inplace = True)
    assert np.all(pred_results.index == algorithm_list)
    res[data_version] = pred_results
  
res.index = res.index.str.upper()
res.loc['Average AUC',:] = res.mean(axis=0)

col_dict = dict({'train':'Training Data', 'test':'Testing Data', 'validation':'Validation Data',
                 'ACEI_ARBS':'ACEI/ARBs','NO_ACEI_ARBS':'No ACEI/ARBs'})
res.rename(col_dict, axis = 1, inplace = True)

res.to_latex(buf = save_path+'latex_predictive_table.txt', 
             column_format = 'l'+'c'*res.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c')

#%% Prescriptive Methods table

summ = pd.read_csv(save_path+'matched_metrics_summary.csv')

summ = summ.query('threshold == %s & weighted_status == "%s"' % (threshold, weighted_status))
summ = summ.loc[summ.data_version.isin(data_list),:]
summ = summ[['data_version', 'match_rate', 'presc_count', 
             'average_auc', 'PE', 'CPE', 'pr_low', 'pr_high']]

summ.data_version = summ.data_version.str.capitalize()
summ.rename({'data_version':'Data Version', 'match_rate':'Match Rate',
            'presc_count':'Presc. Count', 'average_auc':'Avg. AUC', 
            'pr_low':'PR (Low)', 'pr_high':'PR (High)'}, axis=1, inplace = True)

summ.set_index('Data Version', inplace = True)
summ.to_latex(buf = save_path+'latex_prescriptive_table.txt', 
             column_format = 'l'+'c'*summ.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c',
             index_names = False)

#%% Appendix - voting of individual methods

data_version = 'test'
agr = pd.read_csv(save_path+data_version+'_'+match_status+'_t'+str(threshold)+'_'+'agreement_byalgorithm.csv')

agr.set_index('algorithm', inplace = True)
agr = agr[['prescription_count','agreement_no_weights']]
agr['agreement_no_weights'] = ["{:.1%}".format(x) for x in agr['agreement_no_weights']]
agr.rename({'prescription_count':'Prescription Count', 'agreement_no_weights':'Agreement with Prescription'}, 
           axis=1, inplace = True)
agr.index = agr.index.str.upper()
agr.rename({'VOTE':'Prescription'}, axis = 0, inplace = True)

agr.to_latex(buf = save_path+'latex_agreement_table.txt', 
             column_format = 'l'+'c'*agr.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c',
             index_names = False)
# preed_reseults
                  


                   