#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:58:08 2020

@author: agni
"""

import os

import itertools
import evaluation_utils as u
import evaluation.importance as imp
import matplotlib.pyplot as plt

#Paths for data access
website_path = '../../../website/'
results_path = '../../covid19_clean_data/'
output_path = '../results'
validation_path_greece = '../../covid19_greece/general_greek_registry.csv'
validation_paths=[validation_path_greece]

#Select the model type
model_types = ['mortality','infection']
model_labs = ['with_lab','without_lab']
seeds = list(range(1, 41))

#Extract the seed
SEED = 1
SPINE_COLOR = 'gray'
model_type = 'mortality'
sensitivity_threshold = 0.9
confidence_level = 0.95


PAPER_TYPE = 'MORTALITY'  # OR PNAS

if PAPER_TYPE == 'MORTALITY':

    ## Tables
    # AUC and Sensitivity results with confidence intervals
    tab = u.classification_report_table_validation(model_type, website_path, model_labs,
                                                   results_path + "xgboost/", validation_paths,
                                                   sensitivity_threshold,
                                                   output_path=output_path)

    # AUC and Sensitivity results with confidence intervals across models
    tab_models = u.classification_report_table_mlmodels(seeds, model_type, model_labs, results_path,
                                                        sensitivity_threshold, confidence_level,
                                                        output_path=output_path)

    ### PLOTS ###

    # AUC Curves
    u.plot_auc_curve_validation(model_type, website_path,
                                model_labs, results_path + 'xgboost/',
                                validation_paths,
                                output_path=output_path)

    # SHAP Features plots
    for model_type, model_lab in itertools.product(model_types, model_labs):
        # for model_type, model_lab in itertools.product(['mortality'],['without_lab']):
        print("Model: %s, %s" % (model_type, model_lab))
        save_path = os.path.join(output_path, model_type, 'model_'+ model_lab)
        imp.feature_importance(model_type, model_lab, website_path, results_path + "xgboost/",
                               save_path, latex=True, feature_limit=10, dependence_plot=True)


    # No longer used
    # Plot 2:
    #  u.plot_precision_recall_curve_validation(model_type, website_path, model_labs, results_path,
    #                                           validation_paths)
    #  plt.show()

    # Plot 3
    #  u.plot_calibration_curve_validation(model_type,website_path, model_labs, results_path,
    #                                      validation_paths)
    #  plt.show()

elif PAPER_TYPE == 'PNAS':

    #PNAS PAPER

    # Bootstrapped Results
    # Calibration
    u.plot_calibration_curve_bootstrap(model_types, model_labs, results_path, seeds)
    plt.show()

    # AUC performance
    u.plot_auc_curve_bootstrap(model_types, model_labs, results_path, seeds)
    plt.show()

    # Precision Recall
    u.plot_precision_recall_curve_bootstrap(model_types, model_labs, results_path, seeds)
    plt.show()

    # Table of average results
    tab = u.classification_report_table_bootstrap(model_types, model_labs, results_path, sensitivity_threshold, confidence_level)

    ####### Best Seed #########
    # Plot calibration curve for all models
    u.plot_calibration_curve(model_types, model_labs, website_path, results_path)
    plt.show()

    # Plot AUC curve for all models
    u.plot_auc_curve(model_types, model_labs, website_path, results_path)
    plt.show()

    # Plot Precision Recall curve for all models
    u.plot_precision_recall_curve(model_types, model_labs, website_path, results_path)
    plt.show()

    # Get performance metrics evaluations
    df = u.classification_report_table(model_types, model_labs, website_path, sensitivity_threshold, results_path)




