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
import xgboost as xgb

# Other packages
import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json, change_SaO2, top_features, remove_dir, impute_missing
import analyzer.dataset as ds

from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt


import shap

#Get data function
def get_dataset(model_type, model_lab, columns):
    
    ## Variables determined by input
    prediction = 'Outcome'if model_type == 'mortality' else 'Swab'
    swabs_data = False if model_type == 'mortality' else True
    comorbidities_data = True if model_type == 'mortality' else False
    lab_tests = True if model_lab == 'with_lab' else False

    # ## Constant variables
    extra_data = False
    demographics_data = True
    discharge_data = True
    vitals_data = True

    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])

    ## Load Cremona data
    data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
    X_cremona, y_cremona = ds.create_dataset(data, 
                                             discharge_data, 
                                             comorbidities_data, 
                                             vitals_data,
                                             lab_tests, 
                                             demographics_data, 
                                             swabs_data, 
                                             prediction = prediction)

    if model_type == "mortality":
        ## Load Spain data
        data_spain = hmfundacion.load_fundacionhm('../data/spain/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)
        X_spain, y_spain =  ds.create_dataset(data_spain, discharge_data, comorbidities_data, vitals_data,
                                          lab_tests, demographics_data, swabs_data, prediction = prediction)

        # Merge datasets, filter outliers, match format of stored model
        X = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
        y = pd.concat([y_cremona, y_spain], ignore_index=True)
    else: 
        X, y = X_cremona, y_cremona

    X, bounds_dict = ds.filter_outliers(X)
    X = X[columns] 
        
    return X, y


SEED = 1
prediction = 'Swab'
folder_name = 'swab_prediction_seed' + str(SEED) + '_' + prediction.lower()
output_folder = 'predictors/swab'

discharge_data = False
comorbidities_data = False
vitals_data = True
lab_tests = True
demographics_data = True
swabs_data = True
icu_data = False

# Load cremona data
data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)

# Create dataset
X, y = ds.create_dataset(data,
                         discharge_data,
                         comorbidities_data,
                         vitals_data,
                         lab_tests,
                         demographics_data,
                         swabs_data,
                         prediction=prediction)




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
    
    X, y = get_dataset(model_type, model_lab, columns)
    X = impute_missing(X)
        
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
                     label="%s (%1.3f)" % (name, clf_score))

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


#Paths for data access
website_path = '/Users/agni/Packages/website/'
path_cremona = '/Users/agni/Dropbox (MIT)/COVID_clinical/covid19_cremona/data/'
path_hm = '/Users/agni/Dropbox (MIT)/COVID_clinical/covid19_hmfoundation/'

#Select the model type
model_types = ['mortality','infection']
model_labs = ['with_lab','without_lab']

#Extract the seed
SEED = 1

# Plot calibration curve for all models
plot_calibration_curve(model_types, model_labs, website_path)
plt.show()












