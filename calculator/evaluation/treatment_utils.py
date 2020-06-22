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


import analyzer.dataset as ds

def load_data(data_path, split, seed = 1):
    data = pd.read_csv(data_path)

    X_hope, y_hope = ds.create_dataset_treatment(data)
    X_hope.index.name = 'ID'
    y_hope.index.name = 'ID'
    Z_hope = X_hope['REGIMEN']
    X_hope = X_hope.drop('REGIMEN', axis = 1)
    
    # Merge dataset
    X = pd.concat([X_hope], join='inner', ignore_index=True)
    Z = pd.concat([Z_hope], ignore_index=True)
    y = pd.concat([y_hope], ignore_index=True)
    
    #One hot encoding
    X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

    if split == 'bycountry':
        #Split by country and then remove all columns related to country
        train_inds = X['COUNTRY_SPAIN'] == 1
        
    filter_col = [col for col in X if col.startswith('COUNTRY')]
        
    X_train = X.loc[train_inds,].drop(filter_col, axis=1)
    Z_train = Z[train_inds]
    y_train = y[train_inds]
    X_test = X.loc[-train_inds,].drop(filter_col, axis=1)
    y_test = y[-train_inds]
    Z_test = Z[-train_inds]
    
    return X_train, Z_train, y_train, X_test, Z_test, y_test

def generate_preds(X, treatment, algorithm, dataset, SEED = 1, prediction = 'DEATH', split = 'bycountry'): 
    ## Results path and file names
    result_path = '../../covid19_treatments_results/' + str(algorithm) +'/'
    file_list = os.listdir(result_path)
    file_start = str(dataset)+'_results_treatment_'+str(treatment)+'_seed' + str(SEED) + '_split_' + str(split) + '_' + prediction.lower()
    file_name = ''
    for  f in file_list:
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

def algorithm_predictions(X, treatment_list, algorithm, dataset, SEED = 1, prediction = 'DEATH', split = 'bycountry'):
    pred_list = [generate_preds(X, t, algorithm, dataset, SEED, prediction, split) for t in treatment_list]
    df = pd.DataFrame(np.column_stack(pred_list))
    df.columns = treatment_list
    df.index = X.index
    df.index.name = 'ID'
    df['Algorithm'] = algorithm
    df.set_index('Algorithm', append = True, inplace = True)
    return df


def algorithm_prediction_evaluation(X, Z, y, treatment_list, algorithm, dataset, SEED = 1, prediction = 'DEATH', split = 'bycountry'):
        
    #Create a list of all the treatment predictions for a given algorithm
    df = algorithm_predictions(X, algorithm = algorithm, dataset = dataset, treatment_list = treatment_list)       
    
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
        auc_t = metrics.roc_auc_score(y_t, probs_t)
        df_results.append(auc_t)
                
    return df_results
    

def algorithms_pred_evaluation(X, Z, y, treatment_list, algorithm_list, dataset, SEED = 1, prediction = 'DEATH', split = 'bycountry'):

     #creates a new dataframe that's empty where we will store all the results
    auc_results = pd.DataFrame(columns = treatment_list)

    #Retrieve the AUCs for every algorithm
    for alg in algorithm_list:
       res_alg = algorithm_prediction_evaluation(X, Z, y, treatment_list, alg, dataset, SEED = 1, prediction = 'DEATH', split = 'bycountry')
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
             temp_res = filter_by(result, {'ID' : i})
             #Reset the index to access easily the algorithm name
             temp_res.reset_index(inplace=True)
             
             #Create a score for the each option
             list_score = list()
             for t in prescriptions:
             
                 #Add an AUC for that recommendation
                 #Find what are the algorithms for this recommendation
                 algs = temp_res[temp_res['Prescribe']==t]['Algorithm']                
                 list_score.append(pred_results[pred_results.index.isin(algs)][t].mean())
                    
             best_treatment_idx = list_score.index(min(list_score))    
             best_treatment = prescriptions[best_treatment_idx]
             
             summary.at[i,'Prescribe']=best_treatment
    return summary


def retrieve_proba_per_prescription(result, summary, pred_results):

    result_red = result[{'Prescribe','Prescribe_Prediction'}]
    result_red.reset_index(inplace=True)        

    merged_summary = pd.merge(result_red, summary, left_on='ID', right_index=True)
    merged_summary = merged_summary[merged_summary['Prescribe_y']==merged_summary['Prescribe_x']]

    merged_summary.drop(columns=['Prescribe_y'], inplace=True)
    merged_summary.rename(columns={"Prescribe_x":"Prescribe"}, inplace=True)

    #Add the AUC for each treatment and algorithm
    merged_summary['AUC'] = 0
    for i, row in merged_summary.iterrows():    
        merged_summary.loc[i,'AUC'] = pred_results.loc[row['Algorithm'],row['Prescribe']]
    
    return merged_summary

def prescription_effectiveness(result_df, summary, pred_results,algorithm_list):
    
    
    result_df = result_df.reset_index()        
    #Add the prescription decision and the outcome for every patient
    merged_summary = pd.merge(result_df, summary[{'Prescribe','DEATH'}], left_on='ID', right_index=True)
    
    merged_summary.drop(columns={'Prescribe_x'}, inplace=True)
    merged_summary.rename(columns={"Prescribe_y":"Prescribe"}, inplace=True)
    
    merged_summary = merged_summary.melt(id_vars=['ID', 'Algorithm','Prescribe_Prediction', 'DEATH', 'Prescribe'])
    
    pe_list = list()
    for alg in algorithm_list: 
        #Filter to the appropriate ground truth
        # Convert to long format
        res = merged_summary[(merged_summary['Algorithm']==alg) & (merged_summary['Prescribe']==merged_summary['variable'])]

        pe = res.value.mean() - res.DEATH.mean()
        pe_list.append(pe)
        
    pe_list = pd.Series(pe_list, index = algorithm_list)

    return pe_list

def prescription_robustness_a(result, summary, pred_results,algorithm_list):
    
    result_df = result
    
    result_df = result_df.reset_index()        
    #Add the prescription decision and the outcome for every patient
    merged_summary = pd.merge(result_df, summary[{'Prescribe','DEATH','REGIMEN'}], left_on='ID', right_index=True)
    
    merged_summary.drop(columns={'Prescribe_x'}, inplace=True)
    merged_summary.rename(columns={"Prescribe_y":"Prescribe"}, inplace=True)
    
    merged_summary = merged_summary.melt(id_vars=['ID', 'Algorithm','Prescribe_Prediction', 'DEATH', 'REGIMEN','Prescribe'])

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


def algorithm_prescription_robustness(result, n_summary, pred_results,algorithm_list):
    
    result_df = result
    
    result_df = result_df.reset_index()        
    #Add the prescription decision and the outcome for every patient
    merged_summary = pd.merge(result_df, n_summary[{'Prescribe','DEATH','REGIMEN','AverageProbability'}], left_on='ID', right_index=True)
    
    merged_summary.drop(columns={'Prescribe_x'}, inplace=True)
    merged_summary.rename(columns={"Prescribe_y":"Prescribe"}, inplace=True)
    
    merged_summary = merged_summary.melt(id_vars=['ID', 'Algorithm','Prescribe_Prediction', 'DEATH', 'REGIMEN','Prescribe','AverageProbability'])

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




def get_prescription_AUC(n_summary):
    
    y_t = n_summary[n_summary['Match']==True]['DEATH']
    pred_t = n_summary[n_summary['Match']==True]['AverageProbability']
    
    auc_res = metrics.roc_auc_score(y_t, pred_t)
    
    return auc_res



