#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:58:08 2020

@author: agni
"""

import numpy as np
import pandas as pd
import os
import pickle

# Other packages
import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json, change_SaO2, top_features, remove_dir, impute_missing
import analyzer.dataset as ds

from sklearn.metrics import (brier_score_loss, precision_score, recall_score,accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
from sklearn import metrics

#Get data function
def get_dataset(model_type, model_lab, columns, imputer):
    
    ## Variables determined by input
    prediction = 'Outcome'if model_type == 'mortality' else 'Swab'
    swabs_data = False if model_type == 'mortality' else True
    comorbidities_data = True if model_type == 'mortality' else False
    discharge_data = True if model_type == 'mortality' else False 
    lab_tests = True if model_lab == 'with_lab' else False

    # ## Constant variables
    extra_data = False
    demographics_data = True
    vitals_data = True

    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])

    ## Load Cremona data
    data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
    X_cremona, y_cremona = ds.create_dataset(data, discharge_data, comorbidities_data, vitals_data,
                                      lab_tests, demographics_data, swabs_data, prediction = prediction)

    if model_type == "mortality":
        ## Load Spain data
        data_spain = hmfundacion.load_fundacionhm('../data/spain/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)
        X_spain, y_spain =  ds.create_dataset(data_spain, discharge_data, comorbidities_data, vitals_data,
                                             lab_tests, demographics_data, swabs_data, prediction = prediction)

        # Merge datasets, filter outliers, match format of stored model
        X0 = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
        y = pd.concat([y_cremona, y_spain], ignore_index=True)
    else: 
        X0, y = X_cremona, y_cremona

    X0, bounds_dict = ds.filter_outliers(X0)
    X0 = X0[columns] 

    X = pd.DataFrame(imputer.transform(X0))
    X.columns =  X0.columns
        
    return X, y


def get_model_outcomes(model_type, model_lab, website_path):
        
    print(model_type)
    print(model_lab)
        
    #Load model corresponding to model_type and lab
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    #Extract the inputs of the model    
    model = model_file['model']
    features = model_file['json']
    columns = model_file['columns']
    imputer= model_file['imputer']
    
    X, y = get_dataset(model_type, model_lab, columns, imputer)
        
    y_pred = model.predict(X)
    prob_pos = model.predict_proba(X)[:, 1]

    return y, y_pred, prob_pos 

def plot_calibration_curve(model_types, model_labs, website_path):
    """Plot calibration curve for est w/o and with calibration. """
    fig_index=1

    
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
 
    #Get the data 
    for model_type in model_types:
        for model_lab in model_labs:           
        
            name = "Calibration Plot of "+model_type+" "+model_lab
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path)
       
            model_score = brier_score_loss(y, prob_pos, pos_label=y.max())

            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y, prob_pos, n_bins=10)

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s (%1.3f)" % (name, model_score))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                     histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+' calibration_plot.png', bbox_inches='tight')

def plot_auc_curve(model_types, model_labs, website_path):
    fig = plt.figure(1, figsize=(10, 10))

    #Get the data 
    for model_type in model_types:
        for model_lab in model_labs:           
        
            name = "AUC of "+model_type+" "+model_lab+" "
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path)
            fpr, tpr, _ = metrics.roc_curve(y,  prob_pos)
            auc = metrics.roc_auc_score(y, prob_pos)
            plt.plot(fpr,tpr,label=name+str(round(auc,3)))
            plt.legend(loc=4)

    plt.ylabel("Sensitivity")
    plt.xlabel("1 - Specificity")
    plt.tight_layout()
    plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+' auc_plot.png', bbox_inches='tight')

def plot_precision_recall_curve(model_types, model_labs, website_path):
    fig = plt.figure(1, figsize=(10, 10))

    #Get the data 
    for model_type in model_types:
        for model_lab in model_labs:           
        
            name = "Precision Recall of "+model_type+" "+model_lab+" "
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path)
            
            precision, recall, _ = precision_recall_curve(y,prob_pos)
            
            plt.plot(recall,precision,label=name)
            plt.legend(loc=4)

            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # show the legend
            plt.legend()
            # show the plot
            plt.tight_layout()
            plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+' precision_recall_plot.png', bbox_inches='tight')


def get_scores(y, y_pred, threshold):
   CM = confusion_matrix(y, y_pred)
   TN = CM[0][0]
   FN = CM[1][0] 
   TP = CM[1][1]
   FP = CM[0][1]
   # Sensitivity, hit rate, recall, or true positive rate
   TPR = TP/(TP+FN)
   # Specificity or true negative rate
   TNR = TN/(TN+FP) 
   # Precision or positive predictive value
   PPV = TP/(TP+FP)
   # Negative predictive value
   NPV = TN/(TN+FN)
   # Fall out or false positive rate
   FPR = FP/(FP+TN)
   # False negative rate
   FNR = FN/(TP+FN)
   # False discovery rate
   FDR = FP/(TP+FP)

   # Overall accuracy
   ACC = (TP+TN)/(TP+FP+FN+TN)

   colnames = ['Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
   mat_met = pd.DataFrame(columns = colnames) 
   newrow =[threshold, ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR]
   mat_met.loc[0] = newrow
   return mat_met

def classification_report_table(model_types, model_labs, website_path, sensitivity_threshold):
    
    #Get the data ''
    
    cols = ['Model Type','Model Labs','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
    tab = pd.DataFrame(columns = cols) 
    
    for model_type in model_types:
        for model_lab in model_labs:           
        
            name = "Precision Recall of "+model_type+" "+model_lab+" "
            target_names = ['no outcome', 'outcome']
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path)
            
            is_fpr, is_tpr, thresh = precision_recall_curve(y, prob_pos)

            colnames = ['Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
            sum_table = pd.DataFrame(columns = colnames) 

            for t in thresh:
                y_pred = [1 if m > t else 0 for m in prob_pos]
                sum_table.loc[len(sum_table)] = get_scores(y, y_pred, t).loc[0]
            
            sum_table.to_csv('../results/performance_evaluation/performance_tables/'+model_type+'_'+model_lab+'_detailed_perforamance.csv', index=False)
            
            df = sum_table[sum_table['Sensitivity'] > sensitivity_threshold]
            x = df[df['Sensitivity'] == df['Sensitivity'].min()].iloc[0]
            x2 = pd.Series({'Model Type': model_type, 'Model Labs': model_lab})
                     
            tab.loc[len(tab)] = x2.append(x)
    
    tab.to_csv('../results/performance_evaluation/performance_tables/'+'_'.join(model_types)+'_'.join(model_labs)+'_'+str(sensitivity_threshold)+'_summary_perforamance.csv', index=False)
    return tab

#Paths for data access
website_path = '/Users/agni/Packages/website/'

#Select the model type
model_types = ['mortality','infection']
model_labs = ['with_lab','without_lab']

#Extract the seed
SEED = 1

# Plot calibration curve for all models
plot_calibration_curve(model_types, model_labs, website_path)
plt.show()

#Plot AUC curve for all models
plot_auc_curve(model_types, model_labs, website_path)
plt.show()

#Plot Precision Recall curve for all models
plot_precision_recall_curve(model_types, model_labs, website_path)
plt.show()

sensitivity_threshold=0.9
df = classification_report_table(model_types, model_labs, website_path, sensitivity_threshold)




