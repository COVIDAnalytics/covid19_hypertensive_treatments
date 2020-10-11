
import os

# os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.metrics import precision_recall_curve, roc_curve

#%% Version-specific parameters

# version = 'matched_single_treatments_hope_bwh/'
# train_file = '_hope_matched_all_treatments_train.csv'
# data_list = ['train','test','validation_all','validation_partners']

version = 'matched_single_treatments_hypertension/'
train_file = '_hope_hm_cremona_matched_all_treatments_train.csv'
data_list = ['train','test','validation_all','validation_partners',
             'validation_hope','validation_hope_italy']


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
outcome = 'COMORB_DEATH'

treatment = 'ACEI_ARBS'
treatment_list = [treatment, 'NO_'+treatment]

training_set_name = treatment+train_file

#%% Prescriptie results
algorithm_list = ['rf','cart','oct','xgboost','qda','gb']

metrics_agg = pd.DataFrame(columns = ['data_version','weighted_status','threshold','match_rate','presc_count','average_auc','PE','CPE','pr_low','pr_high'])

for outcome in prediction_list:
    version_folder = str(treatment)+'/'+str(outcome)+'/'
    save_path = results_path + version_folder + 'summary/'
    # create summary folder if it does not exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for data_version in data_list:
        print(data_version)
        for threshold in [0,0.01,0.02,0.05,0.1]:
            print('Threshold = '+str(threshold))
            #Read in the relevant data
            X, Z, y = u.load_data(data_path,training_set_name,
                                split=data_version, matched=matched, prediction = outcome,
                                replace_na = 'NO_'+treatment)
            result = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods.csv')
            
            #Filter only to algorithms in the algorithms list
            result =result.loc[result['Algorithm'].isin(algorithm_list)]             
            
            # result.set_index(['ID','Algorithm'], inplace = True)
            pred_results = pd.read_csv(save_path+data_version+'_'+match_status+'_performance_allmethods.csv')
            pred_results.set_index('Algorithm', inplace = True)
            
            #Predictive performance table to base decisions from
            pred_perf_results = pd.read_csv(save_path+'train'+'_'+match_status+'_performance_allmethods.csv')
            pred_perf_results.set_index('Algorithm', inplace = True)
          
            #Compare different schemes
            
            schemes = ['no_weights']
            # schemes = ['weighted','no_weights']
            for weighted_status in schemes:
                print("Prescription scheme = ", weighted_status)
                
                ## Get summary by patient
                if weighted_status == 'weighted':
                    summary = pd.DataFrame(index = result['ID'].unique())
                    for col in treatment_list:
                        pred_auc = pred_perf_results.loc[:,col].rename('AUC', axis = 1)
                        result_join = result.merge(pred_auc, on = 'Algorithm').groupby('ID').apply(u.wavg, col, 'AUC')
                        summary = pd.concat([summary,result_join.rename(col)], axis = 1)
                
                    # if threshold == 0:
                    #     summary['Prescribe'] = summary.idxmin(axis=1)
                    #     summary['AverageProbability'] = summary.min(axis=1)
                    #     summary['Benefit'] = summary.apply(lambda row: (row['NO_'+treatment] -  row[treatment])/row['NO_'+treatment], axis = 1)
                    # else: 
                    summary['NO_'+treatment] = summary['NO_'+treatment].replace(0,1e-4)
                    summary['Benefit'] = summary.apply(lambda row: (row['NO_'+treatment] -  row[treatment])/row['NO_'+treatment], axis = 1)
                    summary['Prescribe'] = summary.apply(lambda row: treatment if row['Benefit'] > threshold else 'NO_'+treatment, axis = 1)
                    summary['AverageProbability'] = summary.apply(lambda row: row[row['Prescribe']], axis = 1)        
    
                    summary = pd.concat([summary, Z, y],axis=1)
                    summary['Match'] = [x.replace(' ','_') == y for x,y in zip(summary['REGIMEN'], summary['Prescribe'])]
                
                    summary.to_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
                    n_summary = summary
                else:
                    result['NO_'+treatment] = result['NO_'+treatment].replace(0,1e-4)
                    result['Benefit'] = result.apply(lambda row: (row['NO_'+treatment] -  row[treatment])/row['NO_'+treatment], axis = 1)
                    result.to_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods_benefit.csv',
                                  index = False)
                    # result['Benefit_Start'] = result.apply(lambda row: (row['NO_'+treatment] -  row[treatment])/row['NO_'+treatment], axis = 1)
                    # result['Benefit_Stop'] = result.apply(lambda row: (row[treatment]- row['NO_'+treatment])/row[treatment], axis = 1)
                    result['Prescribe'] = result.apply(lambda row: treatment if row['Benefit'] > threshold else 'NO_'+treatment, axis = 1)
                    result['Prescribe_Prediction'] = result.apply(lambda row: row[row['Prescribe']], axis = 1)    
                    summary = pd.concat([result.groupby('ID')[treatment_list].agg({'mean'}),
                      result.groupby('ID')['Prescribe'].apply(
                          lambda x:  ', '.join(pd.Series.mode(x).sort_values())), Z, y], axis=1)

                    ## two treatments and 7 methods -- no ties
                    #Resolve Ties among treatments by selecting the treatment whose models have the highest average AUC
                    summary['Prescribe_list'] =   summary['Prescribe'].str.split(pat = ', ')
                    summary = u.resolve_ties(summary, result, pred_perf_results)
                    summary['Match'] = [x.replace(' ','_') == y for x,y in zip(summary['REGIMEN'], summary['Prescribe'])]
                    
                    merged_summary = u.retrieve_proba_per_prescription(result, summary, pred_perf_results)
                    merged_summary.to_csv(save_path+data_version+'_'+match_status+'_detailed_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
                    
                    #Add the average probability and participating algorithms for each patient prescription
                    d1 = merged_summary.groupby('ID')['Prescribe_Prediction'].agg({'mean'})
                    d2 = merged_summary.reset_index().groupby('ID')['Algorithm'].apply(
                                  lambda x:  ', '.join(pd.Series(x))).to_frame()
                    
                    n_summary = pd.merge(summary, d1, left_index=True, right_index=True)
                    n_summary = pd.merge(n_summary, d2, left_index=True, right_index=True)
                    
                    n_summary.rename(columns={"mean":"AverageProbability"},inplace=True)
                    n_summary.to_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
            
            
                prescription_summary = pd.crosstab(index = summary.Prescribe, columns = summary.Match, 
                                                margins = True, margins_name = 'Total')
                prescription_summary.columns = ['No Match', 'Match', 'Total']
                prescription_summary.drop('Total',axis=0)
                prescription_summary.sort_values('Total', ascending = False, inplace = True)
                prescription_summary.to_csv(save_path+data_version+'_'+match_status+'_bytreatment_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
                
                # ===================================================================================
                # Calibration Evaluation
                # We will create calibration plots: line plots of the relative frequency of what was observed (y-axis) versus the predicted probability frequency  (x-axis).
                # The better calibrated or more reliable a forecast, the closer the points will appear along the main diagonal from the bottom left to the top right of the plot.
                # ===================================================================================
                u.simple_calibration_plot(n_summary,  outcome, save_path, data_version, match_status, weighted_status, threshold)
                # ===================================================================================
                # Prescription Effectiveness
                # We will show the difference in the percent of the population that survives.
                # Prescription Effectiveness compares the outcome with the algorithm's suggestion versus what happened in reality
                # ===================================================================================
                #Calculate the baseline mortality rate on the training set.
                X_train, Z_train, y_train = u.load_data(data_path,training_set_name,
                                split='train', matched=matched, prediction = outcome,
                                replace_na = 'NO_'+treatment)

                
                PE = n_summary['AverageProbability'].mean() - n_summary[outcome].mean()
                pe_list = u.prescription_effectiveness(result, summary, pred_results,algorithm_list, y_train, calibration=False, prediction=outcome)
            
                # ===================================================================================
                # Calibrated Prescription Effectiveness
                # We will show the difference in the percent of the population that survives.
                # Prescription Effectiveness compares the outcome with the algorithm's suggestion versus what happened in reality
                # ===================================================================================
                                
                n_summary['CalibratedAverageProbability'] = n_summary['AverageProbability']*(n_summary[outcome].mean()/y_train.mean())
                CPE = n_summary['CalibratedAverageProbability'].mean() - n_summary[outcome].mean()            
                cpe_list = u.prescription_effectiveness(result, summary, pred_results,algorithm_list,y_train, calibration=True, prediction=outcome)
                
                # ===================================================================================
                # Prescription Robustness
                # We will show the difference in the percent of the population that survives.
                # Prescription Robustness compares the outcome with the algorithm's suggestion versus a ground truth estimated by an algorithm
                # ===================================================================================
                # This is prescription robustness of the prescriptive algorithm versus reality, when reality is calculated by alternative ground truths
                PR = u.algorithm_prescription_robustness(result, n_summary, pred_results,algorithm_list,prediction=outcome)
                
                # This is prescription robustness of the prescriptive algorithm versus reality when both decisions take as input alternative ground truths
                pr_table = u.prescription_robustness_a(result, summary, pred_results,algorithm_list,prediction=outcome)
                pr_min = np.diag(pr_table).min()
                pr_max = np.diag(pr_table).max()
                
                #We can create a table and save all the results
                pr_table['PE'] = pe_list
                pr_table['CPE'] = cpe_list
                PR.append(PE)
                PR.append(CPE)
                pr_table.loc['prescr'] = PR
                pr_table.to_csv(save_path+data_version+'_'+match_status+'_prescription_robustness_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
                    
            
                match_rate = n_summary['Match'].mean()
                average_auc = u.get_prescription_AUC(n_summary,prediction=outcome)
                print("Match Rate: ", match_rate)
                print("Average AUC: ", average_auc)
                print("PE: ", PE)
                print("PR Range: ", round(pr_max,3),  " - ", round(pr_min,3))
                
                presc_count = sum(summary['Prescribe']==treatment)
                
                metrics_agg.loc[len(metrics_agg)] = [data_version, weighted_status, threshold, 
                                                      match_rate, presc_count, average_auc, 
                                                      PE, CPE, pr_max, pr_min]
        
        metrics_agg.to_csv(save_path+match_status+'_metrics_summary.csv')
        
#%% 
version_folder = str(treatment)+'/COMORB_DEATH/'
save_path = results_path + version_folder + 'summary/'
    
match_status = 'matched'
weighted_status = 'no_weights'
threshold = 0.05

for data_version in data_list:
    print(data_version)
    # data_version = 'train'
    # detailed_summary = pd.read_csv(save_path+data_version+'_'+match_status+
    #         '_detailed_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
            
    summary = pd.read_csv(save_path+data_version+'_'+match_status+
            '_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
            
    # t = summary.query('Match')
    # print("Match: ")
    # print('Probability: %.3f' % t.AverageProbability.mean())
    # print('Avg. Outcome: %.3f'% t.COMORB_DEATH.mean())
    
    # t = summary.query('~Match')
    # print("No Match: ")
    # print('Probability: %.3f' % t.AverageProbability.mean())
    # print('Avg. Outcome: %.3f' % t.COMORB_DEATH.mean())
    
    summary['OutcomeProb'] = summary.apply(lambda row: row['COMORB_DEATH'] if row['Match'] else row['AverageProbability'], axis=1)
    print("Modified PE: ")
    print('Probability: %.3f' % summary.OutcomeProb.mean())
    print('Avg. Outcome: %.3f' % summary.COMORB_DEATH.mean())


train_bypatient = pd.read_csv(save_path+data_version+'_'+match_status+
        '_bypatient_allmethods.csv')

train_bypatient.groupby('Algorithm')[['ACEI_ARBS','NO_ACEI_ARBS']].mean()
        
#%% Mortality by dataset

for data_version in data_list:
    X, Z, y = u.load_data(data_path,training_set_name,
                        split=data_version, matched=matched, prediction = outcome,
                        replace_na = 'NO_'+treatment)
    print(data_version + format(y.mean()))
                                                                                      
#%% PE Alternative
summary_match = summary.query('Match')


fpr, tpr, thresholds = roc_curve(summary_match['COMORB_DEATH'],  summary_match['AverageProbability'])

roc_table = pd.DataFrame({'threshold':thresholds,
                          'fpr':fpr, 
                          'tpr':tpr})

accuracy_scores = []
for thresh in thresholds:
    accuracy_scores.append(accuracy_score(summary_match['COMORB_DEATH'], 
                                         [1 if m > thresh else 0 for m in summary_match['AverageProbability']]))

accuracies = np.array(accuracy_scores)
max_accuracy = accuracies.max() 
max_accuracy_threshold =  thresholds[accuracies.argmax()]

t = roc_table.loc[roc_table['tpr'] > 0.9, 'threshold'].iloc[0]

summary['PrescribeProb_Binary'] = summary['AverageProbability'].apply(lambda x: 1 if x > 0.5 else 0)   

print('Probability: %.3f' % summary.AverageProbability.mean())
print('Binarized Probability: %.3f' % summary.PrescribeProb_Binary.mean())
print('Avg. Outcome: %.3f' % summary.COMORB_DEATH.mean())
