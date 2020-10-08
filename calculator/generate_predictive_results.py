
import os

# os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
import pandas as pd
import numpy as np
from pathlib import Path

#%% Version-specific parameters

# version = 'matched_single_treatments_hope_bwh/'
# train_file = '_hope_matched_all_treatments_train.csv'
# data_list = ['train','test','validation_all','validation_partners']

version = 'matched_single_treatments_hypertension/'
train_file = '_hope_hm_cremona_matched_all_treatments_train.csv'
data_list = ['train','test','validation_all','validation_partners','validation_hope','validation_hope_italy','validation_cremona']


#%% General parameters

data_path = '../../covid19_treatments_data/'+version
results_path = '../../covid19_treatments_results/'+version

# train_file = '_hope_hm_cremona_matched_all_treatments_train.csv'
        
preload = True
matched = True
match_status = 'matched' if matched else 'unmatched'

SEEDS = range(1, 2)
algorithm_list = ['rf','cart','oct','xgboost','qda','gb']
# prediction_list = ['COMORB_DEATH','OUTCOME_VENT','DEATH','HF','ARF','SEPSIS']
prediction_list = ['COMORB_DEATH']

treatment = 'ACEI_ARBS'
treatment_list = [treatment, 'NO_'+treatment]

training_set_name = treatment+train_file

#%% For each method, generate predictions and accuracies of each method
## Part 1: generate results for all outcomes
for outcome in prediction_list:
    version_folder = str(treatment)+'/'+str(outcome)+'/'
    save_path = results_path + version_folder + 'summary/'
    # create summary folder if it does not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for data_version in data_list:
    # for data_version in ['train']:
            print("Data = ", data_version, "; Prediction = ", outcome)
            X, Z, y = u.load_data(data_path,training_set_name,
                                split=data_version, matched=matched, prediction = outcome,
                                med_hx  = False, other_tx = False, 
                                replace_na =  'NO_'+treatment)
            print("X observations: "
                  , str(X.shape[0]))
            result = pd.concat([u.algorithm_predictions(X, treatment_list = treatment_list, 
                                                        algorithm = alg,  matched = matched, 
                                                        prediction = outcome,
                                                        result_path = results_path+version_folder) 
                                for alg in algorithm_list], axis = 0)
            # Find optimal prescription across methods
            result['Prescribe'] = result.idxmin(axis=1)
            result['Prescribe_Prediction'] = result.min(axis=1)
            #  Save result file
            result.to_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods.csv')
            # =============================================================================
            # Predictive Performance evaluation:
            # - Given a combination of treatment and method calculate the AUC 
            # - All results are saved in a panda where every column is a treatment and every row is a different algorithm
            # =============================================================================
            pred_results = u.algorithms_pred_evaluation(X, Z, y, treatment_list, algorithm_list, 
                                                        matched = matched, prediction = outcome,
                                                        result_path = results_path+version_folder)
            pred_results.to_csv(save_path+data_version+'_'+match_status+'_performance_allmethods.csv', index_label = 'Algorithm')
       
    

#%% Run with LR for comparison
algorithm_list = ['lr','rf','cart','oct','xgboost','qda','gb']
# data_list = ['validation_all','validation_partners']

for outcome in prediction_list:
    version_folder = str(treatment)+'/'+str(outcome)+'/'
    save_path = results_path + version_folder + 'summary/'
    # create summary folder if it does not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for data_version in data_list:
    # for data_version in ['train']:
            print("Data = ", data_version, "; Prediction = ", outcome)
            X, Z, y = u.load_data(data_path,training_set_name,
                                split=data_version, matched=matched, prediction = outcome,
                                med_hx  = False, other_tx = False,
                                replace_na =  'NO_'+treatment)
            print("X observations: "
                  , str(X.shape[0]))
            pred_results = u.algorithms_pred_evaluation(X, Z, y, treatment_list, algorithm_list, 
                                                        matched = matched, prediction = outcome,
                                                        result_path = results_path+version_folder)
            pred_results.to_csv(save_path+data_version+'_'+match_status+'_performance_allmethods_withlr.csv', index_label = 'Algorithm')
            
algorithm_list = ['rf','cart','oct','xgboost','qda','gb']

