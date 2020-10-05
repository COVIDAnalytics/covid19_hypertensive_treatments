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
        # plot_file argumemnt calls summary plot - only works for tree models (will skip others)
        df = u.calculate_average_shap(file_name, treatment, outcome, algorithm, top_features)
                                      # plot_file = save_path+file_start+'_'+algorithm+'summary_plot.pdf')   
        df.to_csv(save_file_name, index = False)
        
        
#%% OCT results

from julia import Julia           
jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
from interpretableai import iai

# model = iai.read_json(file_name+'.json')
            
version_folder = 'matched_single_treatments_der_val_addl_outcomes/'+str(main_treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'
result_path = results_path+version_folder+'oct'+'/'

for treatment in treatment_list:
    file_start = str(treatment) + '_' + match_status + '_' + outcome.lower() + '_seed' + str(SEED)
    file_name = result_path+file_start
    model = iai.read_json(file_name+'.json')
    # pull feature importance
    ft_imp = model.variable_importance()
    # save as CSv
    save_file_name = save_path+file_start+'_'+'oct'+'_shap_values.csv'
    ft_imp.rename({'Feature':'Risk Factor'}, axis=1)
    ft_imp.to_csv(save_file_name, index = False)
    
    


#%% Merge all results

shap_algorithm_list  = ['rf','cart','oct','qda','gb','xgboost']

# set up column name remapping
col_mapping = u.col_mapping
col_mapping['ACEI_ARBS'] = 'ACEI/ARBs'
col_mapping['NO_ACEI_ARBS'] = 'No ACEI/ARBs'

version_folder = 'matched_single_treatments_der_val_addl_outcomes/'+str(main_treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

        
importance_all = []
for algorithm in shap_algorithm_list:
    for treatment in treatment_list:
        file_start = str(treatment) + '_' + match_status + '_' + outcome.lower() + '_seed' + str(SEED)
        save_file_name = save_path+file_start+'_'+algorithm+'_shap_values.csv'
        imp = pd.read_csv(save_file_name)
        if algorithm == 'oct':
            imp.rename({'Feature':'Risk Factor'}, axis=1, inplace = True)
        imp['Rank'] = np.arange(len(imp))+1
        imp['Algorithm'] = algorithm.upper()
        imp['Treatment'] = treatment
        importance_all.append(imp)
    
importance_all = pd.concat(importance_all, axis=0, ignore_index = False)
  
ft_limit = '5'
metric = 'Rank'
# imp_table = importance_all.pivot(index = 'Algorithm', columns = 'Risk Factor', values = metric)
imp_table_t = importance_all.query('Rank <= '+ft_limit).pivot(index = ['Treatment','Risk Factor'], columns = ['Algorithm'], values = metric)
imp_table_t.loc[:,'Average'] = imp_table_t.mean(axis=1)
imp_table_final = imp_table_t.reset_index().sort_values(by=['Treatment','Average'], ascending = [True, True]).\
    set_index(['Treatment','Risk Factor'])
    

imp_table_final.rename(index = col_mapping, inplace = True)

imp_table_final.index.names = [None,None]                            
imp_table_final.to_csv(save_path+'variable_importance_byrank_top'+str(ft_limit)+'.csv')
imp_table_final.to_latex(buf = save_path+'latex_variable_importance_byrank_top'+str(ft_limit)+'.txt', 
             column_format = 'l'*2+'c'*imp_table_t.shape[1],
             float_format="%.1f", bold_rows = False, multicolumn = False, multicolumn_format = 'c',
             index_names = False, na_rep = '--',
             multirow = True)



metric = 'Mean Absolute SHAP Value'
imp_table_t = importance_all.query('Rank <= '+ft_limit).pivot(index = ['Treatment','Risk Factor'], columns = ['Algorithm'], values = metric)
imp_table_t.loc[:,'Average'] = imp_table_t.mean(axis=1)
imp_table_final = imp_table_t.reset_index().sort_values(by=['Treatment','Average'], ascending = [True, False]).\
    set_index(['Treatment','Risk Factor'])
    
imp_table_final.rename(index = col_mapping, inplace = True)

imp_table_final.index.names = [None,None]                                      
imp_table_final.to_csv(save_path+'variable_importance_bySHAP_top'+str(ft_limit)+'.csv')
imp_table_final.to_latex(buf = save_path+'latex_variable_importance_bySHAP_top'+str(ft_limit)+'.txt', 
             column_format = 'l'*2+'c'*imp_table_t.shape[1],
             float_format="%.3f", bold_rows = False, multicolumn = True, multicolumn_format = 'c',
             index_names = False, na_rep = '--',
             multirow = True)


