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