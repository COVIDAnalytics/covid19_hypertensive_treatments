#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:48:05 2020

@author: hollywiberg
"""

import os

# os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
import pandas as pd
import numpy as np
from pathlib import Path

#%% Load data and prescriptions

version = 'matched_single_treatments_der_val_addl_outcomes/'
data_path = '../../covid19_treatments_data/'+version
results_path = '../../covid19_treatments_results/'+version
        
matched = True
weighted = True

match_status = 'matched' if matched else 'unmatched'
weighted_status = 'weighted' if weighted else 'no_weights'

# SEEDS = range(1, 2)
algorithm_list = ['lr','rf','cart','oct','xgboost','qda','gb']

treatment = 'CORTICOSTEROIDS'
treatment_list = [treatment, 'NO_'+treatment]
primary_outcome = 'COMORB_DEATH'
prediction_list = ['OUTCOME_VENT','DEATH','HF','ARF','SEPSIS']
data_list = ['train','test','validation','validation_cremona','validation_hope','validation_hope_italy']


#%% What is baseline incidence?
training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'
data = pd.read_csv(data_path+training_set_name)
grouped = data.groupby('REGIMEN')[prediction_list+[primary_outcome]].mean()
grouped.to_csv(results_path+'outcome_incidence.csv')

#%% Evaluate primary outcome against others

# Load prescriptions, dictated by primary outcome
    
def evaluate_outcome(outcome, primary_outcome, data_version, results_path, treatment,
                     match_status = 'matched', weighted_status = 'weighted'):
    
    prescription_path = results_path + str(treatment)+'/'+str(primary_outcome)+'/' + 'summary/'
    prescs = pd.read_csv(prescription_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'.csv')
    prescs.rename({'Prescribe':'Prescribe_Presc'}, axis=1, inplace = True)
    
    version_folder = str(treatment)+'/'+str(outcome)+'/'
    save_path = results_path + version_folder + 'summary/'
    preds = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'.csv')
    preds.rename({'Prescribe':'Prescribe_Pred'}, axis=1, inplace = True)
    
    ## Where to save results
    comparison_path = prescription_path+'comparison/'+data_version+'_'+match_status+'_'+weighted_status+'_predicted_'+outcome
    Path(prescription_path+'comparison/').mkdir(parents=True, exist_ok=True)
    
    ## Join in prescriptions from primary outcome and find probability associated with this prescription
    df = preds.drop('Match', axis=1).merge(prescs[['ID','Prescribe_Presc']], how = 'left', on = 'ID')
    df['Probability_Presc'] =  df.apply(lambda row: row[row['Prescribe_Presc']], axis = 1)
    
    ## How often do primary outcome and new outcome agree on prescription?
    matches = df.groupby(['REGIMEN','Prescribe_Pred','Prescribe_Presc'])['ID'].count()
    matches.to_csv(comparison_path+'_matches.csv')
    match_rate = (df['Prescribe_Pred']==df['Prescribe_Presc']).mean()
    print("Prescription match ("+primary_outcome+ ", "+outcome+"): "+ str(round(match_rate,3)))

    ## PE = probability of prescription - true event rate
    PE = df['Probability_Presc'].mean()  - df[outcome].mean()
    print("PE of "+primary_outcome+" prescriptions on "+outcome+": "+str(round(PE,3)))
    
    res = pd.DataFrame(columns = ['Data','Primary Outcome','Outcome','Agreement_Rate','Outcome_Rate','Outcome_Rate_Presc','PE'])
    res.loc[0,:] =[data_version,primary_outcome, outcome, match_rate, df[outcome].mean(),  df['Probability_Presc'].mean(), PE]
                 
    ## Prescription robustness - load outcome predictions
    preds_bymethod = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods.csv')
    for alg in algorithm_list:
        probs_alg = preds_bymethod.loc[preds_bymethod['Algorithm']==alg][['ID']+treatment_list]
        df_alg = df[['ID','Prescribe_Presc','REGIMEN']].merge(probs_alg, how = 'left', on = 'ID')
        probs_given = df_alg.apply(lambda row: row[row['REGIMEN']], axis=1)
        probs_presc = df_alg.apply(lambda row: row[row['Prescribe_Presc']], axis=1)
        ## PR = probability of prescription - probability of true given drug (using ML models for desired outcome)
        PR = probs_presc.mean() - probs_given.mean()
        res['PR_'+alg] = PR
    
    return res

res_list = []
for data_version in data_list:
    for outcome in prediction_list:
        res = evaluate_outcome(outcome, primary_outcome, data_version, results_path, treatment)
        res_list.append(res)
        
res_all = pd.concat(res_list, axis=0, ignore_index = False)

prescription_path = results_path + str(treatment)+'/'+str(primary_outcome)+'/' + 'summary/'    
res_all.to_csv( prescription_path+'comparison/full_comparison.csv', index = False)
    
