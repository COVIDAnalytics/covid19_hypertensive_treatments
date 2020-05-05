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
def get_dataset(model_lab):
    
    if model_lab == 'with_lab':
        lab_tests = True
    elif model_lab == 'without_lab':
        lab_tests = False
    
    extra_data = False
    demographics_data = True
    discharge_data = True
    comorbidities_data = True
    vitals_data = True
    swabs_data = False
    
    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])
    
    ## Load Cremona data
    data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
    X_cremona, y_cremona = ds.create_dataset(data, discharge_data, comorbidities_data, vitals_data,
                                             lab_tests, demographics_data, swabs_data, prediction = prediction)

    ## Load Spain data
    data_spain = hmfundacion.load_fundacionhm('../data/spain/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)
    X_spain, y_spain =  ds.create_dataset(data_spain, discharge_data, comorbidities_data, vitals_data,
                                      lab_tests, demographics_data, swabs_data, prediction = prediction)
    
    # Merge datasets, filter outliers, match format of stored model
    X = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
    y = pd.concat([y_cremona, y_spain], ignore_index=True)
    
    X, bounds_dict = ds.filter_outliers(X)
    X = X[columns] 
    
    return X, y


#Paths for data access
website_path = '/Users/agni/Packages/website/'
path_cremona = '/Users/agni/Dropbox (MIT)/COVID_clinical/covid19_cremona/data/'
path_hm = '/Users/agni/Dropbox (MIT)/COVID_clinical/covid19_hmfoundation/'

#Select the model type
model_type = 'mortality'
prediction = 'Outcome'
model_lab = 'without_lab'

#Load model corresponding to model_type and lab
with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
    model_file = pickle.load(file)

#Extract the inputs of the model    
model = model_file['model']
features = model_file['json']
columns = model_file['columns']

#Extract the seed
SEED = 1

#Get the data 
X, y = get_dataset(model_lab)
X = impute_missing(X)

fig_index=1
name = "Calibration Plot"

def plot_calibration_curve(X, y, model, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    y_pred = model.predict(X)
    prob_pos = model.predict_proba(X)[:, 1]
    
    
    clf_score = brier_score_loss(y, prob_pos, pos_label=y.max())
    print("\tBrier: %1.3f" % (clf_score))
    print("\tPrecision: %1.3f" % precision_score(y, y_pred))
    print("\tRecall: %1.3f" % recall_score(y, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y, y_pred))

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

# Plot calibration curve for Gaussian Naive Bayes
plot_calibration_curve(X, y, model, name, fig_index)

plt.show()


scoring = ['accuracy', 'precision']

y_pred = model.predict_proba(X)
logreg_y, logreg_x = calibration_curve(y, y_pred[:,1], n_bins=10)













