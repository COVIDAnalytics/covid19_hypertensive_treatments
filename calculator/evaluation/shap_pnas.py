#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:44:11 2020

@author: hollywiberg
"""

import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


from math import sqrt
import matplotlib


import evaluation.descriptive_utils as u

#%% Latex-style image printing

SPINE_COLOR = 'gray'

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
              'axes.labelsize': 12, # fontsize for x and y labels (was 12 and before 10)
              'axes.titlesize': 12,
              'font.size': 12, # was 12 and before 10
              'legend.fontsize': 12, # was 12 and before 10
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

def get_model_data(model_type, model_lab, website_path, data_path):

    print(model_type)
    print(model_lab)

    #Load model corresponding to model_type and lab
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    seedID = model_file['best seed']

    #Load model corresponding to model_type and lab
    with open(data_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    #Extract the inputs of the model
    model = model_file['model']
    features = model_file['json']
    columns = model_file['columns']
    imputer= model_file['imputer']
    test = model_file['test']

    if model_type == 'mortality':
        y = test['Outcome']
    else:
        y=test['Swab']

    X = test.iloc[:,0:(len(test.columns)-1)]



#%% Run evaluation
model_type = "mortality"
model_lab = "with_lab"
data_path = '../../covid19_clean_data/pnas_models/xgboost/'
feature_limit = 100
# Mortality with lab: 30
# Mortality without lab: 26
# Infection with lab: 30
# Infection without lab: 6
SEED_ID = 6


assert model_type in('mortality','infection'), "Invalid outcome"
assert model_lab in('with_lab','without_lab'), "Invalid lab specification"

#Load model corresponding to model_type and lab
with open(data_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
    model_file = pickle.load(file)

model = model_file['model']
data = model_file['train']
data_test = model_file['test']

## Load data: to be replaced once we store X_train and X_test. Currently no imputation 
if model_type == "mortality":
    X = data.drop(["Outcome"], axis=1, inplace = False)
    y = data["Outcome"]
    X_test = data_test.drop(["Outcome"], axis=1, inplace = False)
    y_test = data_test["Outcome"]

else:
    X = data.drop(["Swab"], axis=1, inplace = False)
    y = data["Swab"]
    X_test = data_test.drop(["Swab"], axis=1, inplace = False)
    y_test = data_test["Swab"]

## Calculate SHAP values (for each observation x feature)
explainer = shap.TreeExplainer(model,
                               model_output="raw",
                               );
shap_values = explainer.shap_values(X);

plt.close()
shap.summary_plot(shap_values, X, show=False,
                  max_display=10,
                  plot_size=(10, 5),
                  feature_names=[c[:c.find("(")] if c.find("(") != -1 else c for c in X.columns],
                  plot_type="violin")
f = plt.gcf()
ax = plt.gca()
ax.text(0, 1, "(a)",
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        weight='bold',
        )
plt.xlabel('SHAP value (impact on model output)')
f
# f.savefig(os.path.join(save_path, 'summary_plot' + suffix_filter + '.pdf'),
#         bbox_inches='tight'
#         )
# plt.clf()

#%% descriptive
model_type = "mortality"
model_lab = "with_lab"
seedID = 30

with open(data_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
    model_file = pickle.load(file)

#Extract the inputs of the model
model = model_file['model']
features = model_file['json']
columns = model_file['columns']
imputer= model_file['imputer']
test = model_file['test']

X, y = u.get_dataset_preload(model_type, model_lab, hartford = False, filter_type = 'old')

data = X.drop(['Location'], axis = 1).copy()
data['Outcome'] = y

data_a= data.query('Outcome == 1')
data_b = data.query('Outcome == 0')
summary_table = u.pairwise_compare(data_a, data_b, features, 
                                   title_mapping = u.title_mapping_summary, row_order = u.row_order,
                                 filter_A = 'Non-Survivor', filter_B = 'Survivor', digits = 1)


summary_table.to_csv('../results/PNAS_v1/descriptive_mortality.csv',
                      index = False)

#%% Load from infection.cluster

model_type = "infection"
model_lab = "with_lab"
data_path = '../../covid19_clean_data/pnas_models/xgboost/'
# Mortality with lab: 30
# Mortality without lab: 26
# Infection with lab: 30
# Infection without lab: 6
SEED_ID = 30

with open(data_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
    model_file = pickle.load(file)

model = model_file['model']
columns = model_file['columns']
features = model_file['json']

data = X.copy()
data['Outcome'] = y

data_a= data.query('Outcome == 1')
data_b = data.query('Outcome == 0')

row_order_infec = ['Filter', 'Patient Count', 
     'Age',
    'Gender',
    'Cardiac Frequency',
    'Respiratory Frequency',
    'Body Temperature',
    'ABG: Oxygen Saturation (SaO2)',
    'Alanine Aminotransferase (ALT)',
    'Aspartate Aminotransferase (AST)',
    'Blood Creatinine',
    'Blood Urea Nitrogen (BUN)',
    'Blood Calcium',
    'C-Reactive Protein (CRP)',
    'CBC: Hemoglobin',
    'CBC: Mean Corpuscular Volume (MCV)',
    'CBC: Platelets',
    'CBC: Red cell Distribution Width (RDW)',
    'Blood Sodium',
    'Prothrombin Time (INR)',
    'CBC: Leukocytes',
    'Total Bilirubin']

summary_table = u.pairwise_compare(data_a, data_b, features, 
                                   title_mapping = u.title_mapping_summary, 
                                   row_order = row_order_infec,
                                   digits = 1,
                                 filter_A = 'Infection', filter_B = 'No Infection')

summary_table.to_csv('../results/PNAS_v1/descriptive_infection.csv',
                      index = False)

