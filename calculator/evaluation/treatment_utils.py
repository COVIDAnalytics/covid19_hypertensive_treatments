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
import shap
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats
import math
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split


import analyzer.dataset as ds

from math import sqrt


col_mapping = {'ACEI_ARBS': 'ACE Inhibitors or ARBs',
     'AF': 'Atrial Fibrillation',
     'AGE': 'Age',
     'ANTIBIOTICS': 'Antibiotics',
     'ANTICOAGULANTS':'Anticoagulants',
     'ANTIVIRAL':'Antivirals',
     'ANYCEREBROVASCULARDISEASE':'Cerebrovascular Disease',
     'ANYHEARTDISEASE':'Heart Disease',
     'ANYLUNGDISEASE':'Lung Disease',
     # 'BLOOD_PRESSURE_ABNORMAL_B':'Low systolic blood pressure (<100 mm Hg)',
     'BLOOD_PRESSURE_ABNORMAL_B':'Low systolic BP',
     'CANCER':'Cancer',
     'CLOROQUINE':'Hydroxychloroquine',
     'CONECTIVEDISEASE':'Connective Tissue Disease',
     'CREATININE':'Creatinine',
     # 'DDDIMER_B':'Elevated D-Dimer (>0.5 mg/L)',
     'DDDIMER_B':'Elevated D-Dimer',
     'DIABETES':'Diabetes',
     'DISLIPIDEMIA':'Dislipidemia',
     'GENDER_MALE':'Gender = Male',
     'HEMOGLOBIN':'Hemoglobin',
     'HYPERTENSION':'Hypertension',
     'INTERFERONOR':'Interferons',
     # 'LDL_B':'Elevated Lactic Acid Dehydrogenase (>480 U/L)',
     'LDL_B':'Elevated LDH',
     'LEUCOCYTES':'White Blood Cell Count',
     'LIVER_DISEASE':'Liver Disease',
     'LYMPHOCYTES':'Lymphocytes',
     'MAXTEMPERATURE_ADMISSION':'Temperature',
     'OBESITY':'Obesity',
     # 'PCR_B':'Elevated C-Reactive Protein (>10 mg/L)',
     'PCR_B':'Elevated CRP',
     'PLATELETS':'Platelets',
     'RACE_CAUC':'Race = Caucasian',
     'RACE_LATIN':'Race = Hispanic',
     'RACE_ORIENTAL':'Race = Asian',
     'RACE_OTHER':'Race = Other',
     'RENALINSUF':'Renal Insufficiency',
     # 'SAT02_BELOW92':'Low Oxygen Saturation (< 92)',
     'SAT02_BELOW92':'Low Oxygen Saturation',
     'SODIUM':'Blood Sodium',
     'CORTICOSTEROIDS':'Corticosteroids',
     'TOCILIZUMAB':'Tocilizumab',
     # 'TRANSAMINASES_B':'Elevated Transaminase (>40 U/L)',
     'TRANSAMINASES_B':'Elevated Transaminase',
     'VIH':'HIV',
     'HF':'Heart Failure',
     'ARF':'Acute Renal Failure',
     'SEPSIS':'Sepsis',
     'EMBOLIC':'Embolic Event',
     'OUTCOME_VENT':'Mechanical Ventilation',
     'COMORB_DEATH':'Mortality/Morbidity',
     'DEATH':'Death'}

def load_data(folder, train_name, split, matched, prediction = 'DEATH', 
    med_hx=False, other_tx = True, treatment=None, replace_na = None):
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
    
    if sum(X['REGIMEN'].isna() > 0):
        if replace_na != None:
            print("Warning: NA regimens - recode  as "+replace_na)
            X.loc[:,'REGIMEN'] =  X['REGIMEN'].replace(np.nan, replace_na)
    # elif handle_na = 'drop'
    #     print("Warning: NA outcomes - set to 0")
        else:
            print("Warning: NA regimens - must recode")

    
    Z = X['REGIMEN']
    X = X.drop('REGIMEN', axis = 1)
    
    ## Manually drop when one-hot encoding if it is training set
    X = pd.get_dummies(X, prefix_sep='_', drop_first=False)
    try:
        X.drop(['GENDER_FEMALE','RACE_OTHER'], axis=1, inplace = True)
    except: 
        X.drop(['GENDER_FEMALE'], axis=1, inplace = True)
    
    return X, Z, y


def recalibrate_models(treatment, algorithm, matched, result_path, 
                   SEED = 1, prediction = 'DEATH'): 
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
    else:
        with open(file_name, 'rb') as file:
             model_file = pickle.load(file)
        ## Match data to dummy variables for this dataframe
        train = model_file['train'].drop(prediction, axis=1)
        train_y = model_file['train'][prediction]
        if algorithm  != 'oct':
            model = model_file['model']
            model_cv = CalibratedClassifierCV(model, method = "sigmoid")
            model_cv.fit(train, train_y)
            model_file['model_original'] = model
            model_file['model'] = model_cv
            ## Save calibrated model
            with open(file_name, 'wb') as handle:
                pickle.dump(model_file, handle, protocol=4)
        else: 
            model_cv = None
    return model_cv


def generate_preds(X, treatment, algorithm, matched, result_path, 
                   SEED = 1, prediction = 'DEATH'): 
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
        train_y = model_file['train'][prediction]
        X = X.reindex(labels = train.columns,  axis = 1).replace(np.nan,0)
        
                    
        if algorithm  == 'oct':
            from julia import Julia           
            jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
            from interpretableai import iai

            model = iai.read_json(file_name+'.json')
            prob_pos = model.predict_proba(X).iloc[:,1]
        # elif algorithm == 'xgboost':
        else:
            model = model_file['model']
            model_cv = CalibratedClassifierCV(model, method = "sigmoid")
            model_cv.fit(train, train_y)
            prob_pos = model_cv.predict_proba(X)[:, 1]
        # else: 
        #     model = model_file['model']
        #     prob_pos = model.predict_proba(X)[:, 1]
        
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
                 list_score.append(pred_results[pred_results.index.isin(algs)].values.mean())
                 # list_score.append(pred_results[pred_results.index.isin(algs)][t].mean())
                    
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

# def prescription_effectiveness(result_df, summary, pred_results,algorithm_list, prediction = 'DEATH'):
#     result_df = result_df.reset_index()        
#     #Add the prescription decision and the outcome for every patient
#     merged_summary = pd.merge(result_df, summary[{'Prescribe',prediction}], left_on='ID', right_index=True)
    
#     merged_summary.drop(columns={'Prescribe_x'}, inplace=True)
#     merged_summary.rename(columns={"Prescribe_y":"Prescribe"}, inplace=True)
    
#     merged_summary = merged_summary.melt(id_vars=['ID', 'Algorithm','Prescribe_Prediction', prediction, 'Prescribe'])
    
#     pe_list = list()
#     for alg in algorithm_list: 
#         #Filter to the appropriate ground truth
#         # Convert to long format
#         res = merged_summary[(merged_summary['Algorithm']==alg) & (merged_summary['Prescribe']==merged_summary['variable'])]

#         pe = res.value.mean() - res[prediction].mean()
#         pe_list.append(pe)
        
#     pe_list = pd.Series(pe_list, index = algorithm_list)

#     return pe_list

def prescription_effectiveness(result_df, summary, pred_results,algorithm_list, y_train, calibration=False, prediction = 'DEATH'):
    
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
        if calibration:
            pe = res.value.mean()*(res[prediction].mean()/y_train.mean()) - res[prediction].mean()
        else:
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

def simple_calibration_plot(n_summary, outcome, save_path, data_version, match_status, weighted_status, threshold):
    #Reduce it to where is match
    rd_df = n_summary.loc[n_summary['Match']==True]
    fig1 = pyplot.gcf()
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    fop, mpv = calibration_curve(rd_df[outcome], rd_df['AverageProbability'], n_bins=10)
    # plot model reliability
    pyplot.ylabel("Fraction of positives")
    pyplot.ylim([-0.05, 1.05])
    pyplot.legend(loc="lower right")
    pyplot.title('Calibration plots (reliability curve)')
    
    pyplot.xlabel("Mean predicted value")
    
    pyplot.plot(mpv, fop, marker='.')
    
    pyplot.show()
    pyplot.draw()
    fig1.savefig(save_path+data_version+'_'+match_status+'_calibration_plot_with_agreement_'+weighted_status+'_t'+str(threshold)+'.png', dpi=100)

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.
    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    # NB (bart): default font-size in latex is 11. This should exactly match
    # the font size in the text if the figwidth is set appropriately.
    # Note that this does not hold if you put two figures next to each other using
    # minipage. You need to use subplots.
    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 10, # fontsize for x and y labels (was 12 and before 10)
              'axes.titlesize': 14,
              'font.size': 10, # was 12 and before 10
              'legend.fontsize': 10, # was 12 and before 10
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)

def calculate_average_shap(file_name, treatment, outcome,algorithm, top_features, plot_file = ''):
    
    if file_name == '':
        print("Invalid treatment/outcome combination (" + str(treatment) + ", " + str(outcome)+ ")")
    else:
        with open(file_name, 'rb') as file:
             model_file = pickle.load(file)
          
    model = model_file['model']
    data = model_file['train']
    data_test = model_file['test']

    X = data.drop(["COMORB_DEATH"], axis=1, inplace = False)
    y = data["COMORB_DEATH"]
    X_test = data_test.drop(["COMORB_DEATH"], axis=1, inplace = False)
    y_test = data_test["COMORB_DEATH"]

    ## Calculate SHAP values (for each observation x feature)
    if algorithm in ['rf','cart','xgboost']:
        explainer = shap.TreeExplainer(model,
                                   data=X_test,
                                   model_output="probability",
                                   );
        shap_values = explainer.shap_values(X_test);
        
        ## only save plot for tree models
        if plot_file != '':
            plt.close()
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[1], X_test, show=False,
                          max_display=10,
                          plot_size=(10, 5),
                          plot_type="violin")  
            else:             
                shap.summary_plot(shap_values, X_test, show=False,
                      max_display=10,
                      plot_size=(10, 5),
                      plot_type="violin")  
                
            f = plt.gcf()
            ax = plt.gca()
            
            plt.xlabel('SHAP value (impact on model output)')   
        
            f.savefig(plot_file, bbox_inches='tight')
            
            plt.close()
        
    else:
        X_train_summary = shap.kmeans(X, 50)
        explainer = shap.KernelExplainer(model.predict_proba,
                                   data=X_train_summary,
                                   model_output="logit",
                                   );
        shap_values = explainer.shap_values(X_test);
        
        if plot_file != '':
            print('Cannot plot summary plot for non-tree models')


    df = pd.DataFrame(columns = ['Risk Factor', 'Mean Absolute SHAP Value']) 
    
    for i in range(0,len(X.columns)):
        if isinstance(shap_values, list):
            df = df.append({'Risk Factor' : X.columns[i], 'Mean Absolute SHAP Value' : pd.Series(shap_values[1][:,i]).abs().mean()},  
                ignore_index = True) 
        else:
            df = df.append({'Risk Factor' : X.columns[i], 'Mean Absolute SHAP Value' : pd.Series(shap_values[:,i]).abs().mean()},  
                ignore_index = True)
    
    df = df.sort_values(by='Mean Absolute SHAP Value', ascending=False)    
    df = df.head(top_features)
    
    return df