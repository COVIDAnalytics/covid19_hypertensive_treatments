#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:03:21 2020

@author: hollywiberg
"""
import pandas as pd
import numpy as np
import os
import pickle
from sklearn import metrics
from scipy import stats
import math

import analyzer.dataset as ds

def load_data(folder, train_name, split, matched, prediction = 'DEATH', 
    med_hx=False, other_tx = True, treatment=None):
    file = train_name
    #if split == 'validation':
    if 'validation'in split:
        file  = file.replace('_matched','')
    elif not matched:
        file  = file.replace('matched','unmatched')
    file  = file.replace('train',split)
    print(file)
    df = pd.read_csv(folder+file)
    X, y = ds.create_dataset_treatment(df, prediction = prediction, 
                                       med_hx=med_hx, other_tx = other_tx, include_regimen=True)
    X.index.name = 'ID'
    y.index.name = 'ID'
    Z = X['REGIMEN']
    X = X.drop('REGIMEN', axis = 1)
    X = pd.get_dummies(X, prefix_sep='_', drop_first=True)
    
    return X, Z, y

def generate_preds(X, treatment, algorithm, matched, result_path, SEED = 1, prediction = 'DEATH'): 
    ## Results path and file names
    match_status =  'matched' if matched else 'unmatched'
    result_path = result_path + str(algorithm) +'/'
    file_list = os.listdir(result_path)
    file_start = str(treatment) + '_' + match_status + '_' + prediction.lower() + '_seed' + str(SEED)
    file_name = ''
    for f in file_list:
        if f.startswith(file_start) & ~f.endswith('.json'):
            file_name = result_path+f

    if file_name == '':
        print("Invalid treatment/algorithm combination (" + str(treatment) + ", " + str(algorithm)+ ")")
        prob_pos = np.empty(X.shape[0])
        prob_pos[:] = np.nan
        return prob_pos
    else:
        with open(file_name, 'rb') as file:
             model_file = pickle.load(file)
              
        ## Match data to dummy variables for this dataframe
        train = model_file['train'].drop(prediction, axis=1)
        X = X.reindex(labels = train.columns,  axis = 1).replace(np.nan,0)
        
                    
        if algorithm  == 'oct':
            from julia import Julia           
            jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
            from interpretableai import iai

            model = iai.read_json(file_name+'.json')
            prob_pos = model.predict_proba(X).iloc[:,1]
        else:
            model = model_file['model']
            prob_pos = model.predict_proba(X)[:, 1]
        
        return prob_pos
    

def algorithm_predictions(X, treatment_list, algorithm, matched, result_path, SEED = 1, prediction = 'DEATH'):
    pred_list = [generate_preds(X, t, algorithm, matched, result_path, SEED, prediction) for t in treatment_list]
    df = pd.DataFrame(np.column_stack(pred_list))
    df.columns = treatment_list
    df.index = X.index
    df.index.name = 'ID'
    df['Algorithm'] = algorithm
    df.set_index('Algorithm', append = True, inplace = True)
    return df


def algorithm_prediction_evaluation(X, Z, y, treatment_list, algorithm, matched, result_path, SEED = 1, prediction = 'DEATH'):
        
    #Create a list of all the treatment predictions for a given algorithm
    df = algorithm_predictions(X, treatment_list = treatment_list, algorithm = algorithm, matched = matched,
                               result_path  = result_path, SEED  = SEED, prediction = prediction)       
    
    #Rename the Z to match the names of the treatments  
    Z  = [sub.replace(' ', '_') for sub in list(Z)] 
    
    #Create a new column for the treatment and the outcome
    df['Regimen'] = Z
    df['Outcome'] = list(y)

    
    df_results = list()  
    for t in treatment_list:
        #For a given treatment and algorithm get the predicted probability 
        probs_t = df.loc[df['Regimen'] == t][t]
        y_t = df.loc[df['Regimen'] == t]['Outcome']
        if y_t.unique().size == 1:
            auc_t = np.nan
        else: 
            auc_t = metrics.roc_auc_score(y_t, probs_t)
        df_results.append(auc_t)
                
    return df_results
    

def algorithms_pred_evaluation(X, Z, y, treatment_list, algorithm_list, matched, result_path, SEED = 1, prediction = 'DEATH'):

    #creates a new dataframe that's empty where we will store all the results
    auc_results = pd.DataFrame(columns = treatment_list)

    #Retrieve the AUCs for every algorithm
    for alg in algorithm_list:
       res_alg = algorithm_prediction_evaluation(X, Z, y, treatment_list, alg, matched, result_path, SEED, prediction)
       res_alg = pd.Series(res_alg, index = auc_results.columns)
       auc_results = auc_results.append(res_alg,ignore_index=True)

    auc_results.index = algorithm_list
    
    return auc_results

def filter_by(df, constraints):
    """Filter MultiIndex by sublevels."""
    indexer = [constraints[name] if name in constraints else slice(None)
               for name in df.index.names]
    return df.loc[tuple(indexer)] if len(df.shape) == 1 else df.loc[tuple(indexer),]


def resolve_ties(summary, result, pred_results):
    #Resolve ties by looking at the predictive performance of the algorithms
    for i, row in summary.iterrows():
        #Find all the patients for which there is not agreement on the prescription
        if len(row['Prescribe_list'])>1:
                        
             #For every option we will find what are the methods that suggest that
             prescriptions = row['Prescribe_list']
             #We will suggest the option with the highest average AUC

             #Get from the results table the relevant rows for this ID
             temp_res =result.loc[result['ID'] == i]             
             #temp_res = filter_by(result, {'ID' : i})

             #Reset the index to access easily the algorithm name
             temp_res.reset_index(inplace=True)
             
             #Create a score for the each option
             list_score = list()
             for t in prescriptions:
             
                 #Add an AUC for that recommendation
                 #Find what are the algorithms for this recommendation
                 algs = temp_res[temp_res['Prescribe']==t]['Algorithm']                
                 list_score.append(pred_results[pred_results.index.isin(algs)][t].mean())
                    
             best_treatment_idx = list_score.index(max(list_score))    
             best_treatment = prescriptions[best_treatment_idx]
             
             summary.at[i,'Prescribe']=best_treatment
    return summary


def retrieve_proba_per_prescription(result, summary, pred_results):

    result_red = result[{'ID','Algorithm','Prescribe','Prescribe_Prediction'}]
    #result_red.reset_index(inplace=True)        

    merged_summary = pd.merge(result_red, summary, left_on='ID', right_index=True)
    #Keep only the algorithms for which we have agreement
    merged_summary = merged_summary[merged_summary['Prescribe_y']==merged_summary['Prescribe_x']]

    merged_summary.drop(columns=['Prescribe_y'], inplace=True)
    merged_summary.rename(columns={"Prescribe_x":"Prescribe"}, inplace=True)

    #Add the AUC for each treatment and algorithm
    merged_summary['AUC'] = 0
    for i, row in merged_summary.iterrows():    
        merged_summary.loc[i,'AUC'] = pred_results.loc[row['Algorithm'],row['Prescribe']]
    
    return merged_summary

def prescription_effectiveness(result_df, summary, pred_results,algorithm_list, prediction = 'DEATH'):
    
    
    result_df = result_df.reset_index()        
    #Add the prescription decision and the outcome for every patient
    merged_summary = pd.merge(result_df, summary[{'Prescribe',prediction}], left_on='ID', right_index=True)
    
    merged_summary.drop(columns={'Prescribe_x'}, inplace=True)
    merged_summary.rename(columns={"Prescribe_y":"Prescribe"}, inplace=True)
    
    merged_summary = merged_summary.melt(id_vars=['ID', 'Algorithm','Prescribe_Prediction', prediction, 'Prescribe'])
    
    pe_list = list()
    for alg in algorithm_list: 
        #Filter to the appropriate ground truth
        # Convert to long format
        res = merged_summary[(merged_summary['Algorithm']==alg) & (merged_summary['Prescribe']==merged_summary['variable'])]

        pe = res.value.mean() - res[prediction].mean()
        pe_list.append(pe)
        
    pe_list = pd.Series(pe_list, index = algorithm_list)

    return pe_list

def prescription_robustness_a(result, summary, pred_results,algorithm_list, prediction = 'DEATH'):
    
    result_df = result
    
    result_df = result_df.reset_index()        
    #Add the prescription decision and the outcome for every patient
    merged_summary = pd.merge(result_df, summary[{'Prescribe',prediction,'REGIMEN'}], left_on='ID', right_index=True)
    
    merged_summary.drop(columns={'Prescribe_x'}, inplace=True)
    merged_summary.rename(columns={"Prescribe_y":"Prescribe"}, inplace=True)
    
    merged_summary = merged_summary.melt(id_vars=['ID', 'Algorithm','Prescribe_Prediction', prediction, 'REGIMEN','Prescribe'])

    #Add the pads in the columns
    merged_summary['REGIMEN']  = [sub.replace(' ', '_') for sub in list(merged_summary['REGIMEN'])]   
    
    df = pd.DataFrame(columns = algorithm_list, index = algorithm_list)

    for ground_truth in algorithm_list:
        pr_list = list()
        for alg in algorithm_list: 
            #Filter to the appropriate ground truth
            # Convert to long format
            res = merged_summary[(merged_summary['Algorithm']==alg) & (merged_summary['Prescribe']==merged_summary['variable'])]
            gt = merged_summary[(merged_summary['Algorithm']==ground_truth) & (merged_summary['REGIMEN']==merged_summary['variable'])]
            
            pr = res.value.mean() - gt.value.mean()
            pr_list.append(pr)
        
        df[ground_truth] = pr_list        
    return df


def algorithm_prescription_robustness(result, n_summary, pred_results,algorithm_list, prediction = 'DEATH'):
    
    result_df = result
    
    result_df = result_df.reset_index()        
    #Add the prescription decision and the outcome for every patient
    merged_summary = pd.merge(result_df, n_summary[{'Prescribe',prediction,'REGIMEN','AverageProbability'}], left_on='ID', right_index=True)
    
    merged_summary.drop(columns={'Prescribe_x'}, inplace=True)
    merged_summary.rename(columns={"Prescribe_y":"Prescribe"}, inplace=True)
    
    merged_summary = merged_summary.melt(id_vars=['ID', 'Algorithm','Prescribe_Prediction', prediction, 'REGIMEN','Prescribe','AverageProbability'])

    #Add the pads in the columns
    merged_summary['REGIMEN']  = [sub.replace(' ', '_') for sub in list(merged_summary['REGIMEN'])] 

    alg = algorithm_list[0]
    ground_truth = algorithm_list[0]
    
     
    pr_list = list()

    for ground_truth in algorithm_list:
        #Filter to the appropriate ground truth
        # Convert to long format
        gt = merged_summary[(merged_summary['Algorithm']==ground_truth) & (merged_summary['REGIMEN']==merged_summary['variable'])]
            
        pr = gt.AverageProbability.mean() - gt.value.mean()
        pr_list.append(pr)
                
    return pr_list

def CI_printout(series, interval = 0.95, method = 't'):
  mean_val = series.mean()
  n = series.count()
  stdev = series.std()
  if method == 't':
    test_stat = stats.t.ppf((interval + 1)/2, n)
  elif method == 'z':
    test_stat = stats.norm.ppf((interval + 1)/2)
  lower_bound =  round(mean_val - test_stat * stdev / math.sqrt(n),3)
  upper_bound =  round(mean_val + test_stat * stdev / math.sqrt(n),3)

  output = str(round(mean_val,3))+' ('+str(lower_bound)+':'+str(upper_bound)+')'
  return output



def get_prescription_AUC(n_summary, prediction = 'DEATH'):
    
    y_t = n_summary[n_summary['Match']==True][prediction]
    pred_t = n_summary[n_summary['Match']==True]['AverageProbability']
    if y_t.unique().size == 1:
        auc_res = np.nan
    else: 
        auc_res = metrics.roc_auc_score(y_t, pred_t)
    
    return auc_res


def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

