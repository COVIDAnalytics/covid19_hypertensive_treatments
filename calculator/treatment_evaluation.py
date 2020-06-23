"""
Script for aggregating and evaluating performance of predictive models across treatments

Created on Wed Jun 17 13:57:19 2020

@author: hollywiberg
"""

#%% Prepare environment
import os

os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
# import evaluation.importance as imp
# import matplotlib.pyplot as plt
# import analyzer.optuna as o
import pandas as pd

#%% Set Problem Parameters
#Paths for data access
  
data_path = '../../covid19_hope/hope_matched.csv'
save_path = '../../covid19_treatments_results/summary/'
preload = True

dataset = "hope"
treatment_list = ['Chloroquine_Only', 'All', 'Chloroquine_and_Anticoagulants',
                              'Chloroquine_and_Antivirals', 'Non-Chloroquine']


# algorithm_list = range(0,len(o.algo_names))
algorithm_list = ['lr','rf','cart','xgboost','oct', 'kn', 'qda']
# algorithm_list = ['lr','rf','cart','xgboost','oct']
# algorithm_list = ['lr','rf','cart','xgboost']


#%% Generate predictions

split = 'bycountry'
X_train, Z_train, y_train, X_test, Z_test, y_test = u.load_data(data_path, split = split)

data_version = 'test'

if data_version == 'train':
    X = X_train
    Z = Z_train
    y = y_train
else: 
    X = X_test
    Z = Z_test
    y = y_test

if preload: 
    result = pd.read_csv(save_path+data_version+'_bypatient_allmethods.csv')
    result.set_index(['ID','Algorithm'], inplace = True)
    pred_results = pd.read_csv(save_path+data_version+'_performance_allmethods.csv')
    pred_results.set_index('Algorithm', inplace = True)
else:
    result = pd.concat([u.algorithm_predictions(X, algorithm = alg, dataset = dataset, treatment_list = treatment_list) 
                        for alg in algorithm_list], axis = 0)
    
    # Find optimal prescription across methods
    result['Prescribe'] = result.idxmin(axis=1)
    result['Prescribe_Prediction'] = result.min(axis=1)
    
    result.to_csv(save_path+data_version+'_bypatient_allmethods.csv')

    # =============================================================================
    # Predictive Performance evaluation:
    # - Given a combination of treatment and method calculate the AUC 
    # - All results are saved in a panda where every column is a treatment and every row is a different algorithm
    # =============================================================================
    pred_results = u.algorithms_pred_evaluation(X, Z, y, treatment_list, algorithm_list, dataset, SEED = 1, prediction = 'DEATH', split = 'bycountry')
    pred_results.to_csv(save_path+data_version+'_performance_allmethods.csv', index_label = 'Algorithm')


#%% Evaluate Methods
  
# =============================================================================
# Summary contains:
# - Mean probability for each treatment (averaged across methods)
# - Prescribe: most common prescription (mode of prescriptions across methods), list if there are ties
# - REGIMEN: true prescribed treatment
# - Match: indicator of whether true regimen is (in list of) optimal prescription(s)
# =============================================================================
summary = pd.concat([result.groupby('ID')[treatment_list].agg({'mean'}),
          result.groupby('ID')['Prescribe'].apply(
              lambda x:  ', '.join(pd.Series.mode(x).sort_values())),  
          Z, y], axis=1)

summary['Prescribe_list'] =   summary['Prescribe'].str.split(pat = ', ')

#Resolve Ties among treatments by selecting the treatment whose models have the highest average AUC
summary = u.resolve_ties(summary, result, pred_results)
summary['Match'] = [x.replace(' ','_') in(y) for x,y in zip(summary['REGIMEN'], summary['Prescribe'])]
summary.to_csv(save_path+data_version+'_bypatient_summary.csv')

# =============================================================================
# n_summary contains:
# - More detailed information per patient regarding the specific methods that propose the suggested treatmet
# - The proposed probability per method
# =============================================================================
merged_summary = u.retrieve_proba_per_prescription(result, summary, pred_results)
merged_summary.to_csv(save_path+data_version+'detailed_bypatient_summary.csv')


#Add the average probability and participating algorithms for each patient prescription
d1 = merged_summary.groupby('ID')['Prescribe_Prediction'].agg({'mean'})
d2 = merged_summary.reset_index().groupby('ID')['Algorithm'].apply(
              lambda x:  ', '.join(pd.Series(x))).to_frame()

n_summary = pd.merge(summary, d1, left_index=True, right_index=True)
n_summary = pd.merge(n_summary, d2, left_index=True, right_index=True)

n_summary.rename(columns={"mean":"AverageProbability"},inplace=True)

# =============================================================================
# Prescription_summary contains:
# - Frequencies of prescriptions and how often they were actually prescribed
# =============================================================================
prescription_summary = pd.crosstab(index = summary.Prescribe, columns = summary.Match, 
                                   margins = True, margins_name = 'Total')
prescription_summary.columns = ['No Match', 'Match', 'Total']
prescription_summary.drop('Total',axis=0)
prescription_summary.sort_values('Total', ascending = False, inplace = True)
prescription_summary.to_csv(save_path+data_version+'_bytreatment_summary.csv')

# ===================================================================================
# Prescription Accuracy evaluation:
# - Given the prescription and the probability rule, what is the AUC of the overall prediction
# ===================================================================================
average_auc = u.get_prescription_AUC(n_summary)
average_auc

# ===================================================================================
# Prescription Effectiveness
# We will show the difference in the percent of the population that survives.
# Prescription Effectiveness compares the outcome with the algorithm's suggestion versus what happened in reality
# ===================================================================================
# This is prescription effectiveness of the prescriptive algorithm versus reality
PE = n_summary['AverageProbability'].mean() - n_summary['DEATH'].mean()

#We can also compute the probability of that decision for a different method and then compare the outcome
pe_list = u.prescription_effectiveness(result, summary, pred_results,algorithm_list)

# ===================================================================================
# Prescription Robustness
# We will show the difference in the percent of the population that survives.
# Prescription Robustness compares the outcome with the algorithm's suggestion versus a ground truth estimated by an algorithm
# ===================================================================================
# This is prescription robustness of the prescriptive algorithm versus reality, when reality is calculated by alternative ground truths
PR = u.algorithm_prescription_robustness(result, n_summary, pred_results,algorithm_list)

# This is prescription robustness of the prescriptive algorithm versus reality when both decisions take as input alternative ground truths
pr_table = u.prescription_robustness_a(result, summary, pred_results,algorithm_list)

#We can create a table and save all the results
pr_table['PE'] = pe_list
PR.append(PE)
pr_table.loc['prescr'] = PR
pr_table.to_csv(save_path+data_version+'prescription_robustness_summary.csv')















