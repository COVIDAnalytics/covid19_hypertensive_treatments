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
import csv

import evaluation_utils as u
import matplotlib.pyplot as plt


#Paths for data access
website_path = '/Users/agni/Packages/website/'
results_path = '/Users/agni/Dropbox (Personal)/COVID_clinical/covid19_clean_data/xgboost/'
validation_path = '/Users/agni/Dropbox (Personal)/COVID_clinical/covid19_greece/'



#Select the model type
model_types = ['mortality','infection']
model_labs = ['with_lab','without_lab']
seeds = list(range(1, 41))

#Extract the seed
SEED = 1
SPINE_COLOR = 'gray'


model_type = 'mortality'
model_lab = 'with_lab'
model_lab = 'without_lab'










# Bootstrapped Results
# Calibration
u.plot_calibration_curve_bootstrap(model_types, model_labs, results_path, seeds)
plt.show()

# AUC performance
u. plot_auc_curve_bootstrap(model_types, model_labs, results_path, seeds)
plt.show()

#Precision Recall
u.plot_precision_recall_curve_bootstrap(model_types, model_labs, results_path, seeds)
plt.show()

#Table of average results
sensitivity_threshold = 0.9
confidence_level = 0.95
tab = u.classification_report_table_bootstrap(model_types, model_labs, results_path, sensitivity_threshold, confidence_level)

####### Best Seed #########
# Plot calibration curve for all models
u.plot_calibration_curve(model_types, model_labs, website_path, results_path)
plt.show()

#Plot AUC curve for all models
u.plot_auc_curve(model_types, model_labs, website_path, results_path)
plt.show()

#Plot Precision Recall curve for all models
u.plot_precision_recall_curve(model_types, model_labs, website_path, results_path)
plt.show()

#Get performance metrics evaluations
sensitivity_threshold=0.9
df = u.classification_report_table(model_types, model_labs, website_path, sensitivity_threshold, results_path)




