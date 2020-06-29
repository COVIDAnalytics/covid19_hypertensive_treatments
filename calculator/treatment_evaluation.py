"""
Script for aggregating and evaluating performance of predictive models across treatments

Created on Wed Jun 17 13:57:19 2020

@author: hollywiberg
"""

#%% Prepare environment
import os

os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
import pandas as pd
from pathlib import Path


#%% Set Problem Parameters
#Paths for data access

data_path = '../../covid19_treatments_data/'
results_path = '../../covid19_treatments_results/'
version_folder = "unmatched_and_matched_all_treatments/"
save_path = results_path + version_folder + 'summary/'
preload = False

treatment_list = ['Chloroquine_Only', 'All', 'Chloroquine_and_Anticoagulants',
                              'Chloroquine_and_Antivirals', 'Non-Chloroquine']
algorithm_list = ['lr','rf','cart','xgboost','oct']

#%% Generate predictions across all combinations

if not preload:
    # create summary folder if it does not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for data_version in ['train','test','validation']:
        for matched in [True,False]:
            match_status = 'matched' if matched else 'unmatched'
            print(data_version + ' - ' + match_status)
            X, Z, y = u.load_data(data_path+version_folder,'hope_hm_cremona_matched_all_treatments_train.csv',
                                split=data_version,matched=matched)
            print("X observations: ", str(X.shape[0]))
            result = pd.concat([u.algorithm_predictions(X, treatment_list = treatment_list, 
                                                        algorithm = alg,  matched = matched, 
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
                                                        matched = matched, 
                                                        result_path = results_path+version_folder)
            pred_results.to_csv(save_path+data_version+'_'+match_status+'_performance_allmethods.csv', index_label = 'Algorithm')
    

#%% Evaluate Methods for a single variant

data_version = 'validation'
matched = True
match_status = 'matched' if matched else 'unmatched'

result = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods.csv')
result.set_index(['ID','Algorithm'], inplace = True)
pred_results = pd.read_csv(save_path+data_version+'_'+match_status+'_performance_allmethods.csv')
pred_results.set_index('Algorithm', inplace = True)

  
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
summary.to_csv(save_path+data_version+'_'+match_status+'_bypatient_summary.csv')

# =============================================================================
# n_summary contains:
# - More detailed information per patient regarding the specific methods that propose the suggested treatmet
# - The proposed probability per method
# =============================================================================
merged_summary = u.retrieve_proba_per_prescription(result, summary, pred_results)
merged_summary.to_csv(save_path+data_version+'_'+match_status+'_detailed_bypatient_summary.csv')


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
prescription_summary.to_csv(save_path+data_version+'_'+match_status+'_bytreatment_summary.csv')

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
pr_table.to_csv(save_path+data_version+'_'+match_status+'_prescription_robustness_summary.csv')



#%%  Alternative voting scheme

# wm = lambda x: np.average(x, weights=result.iloc[x.index,1])




# result_join = pd.DataFrame(result.loc[:,col]).merge(pred_results.loc[:,col], on = 'Algorithm')

# result_join.groupby('ID')

# summary = pd.concat([result.groupby('ID')[treatment_list].agg({'mean'}),
#           result.groupby('ID')['Prescribe'].apply(
#               lambda x:  ', '.join(pd.Series.mode(x).sort_values())),  
#           Z, y], axis=1)



# result.groupby('ID')








