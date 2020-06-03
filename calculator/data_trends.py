#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 09:59:21 2020

@author: agni
"""

import os

import itertools
import evaluation_utils as u
import evaluation.importance as imp
import matplotlib.pyplot as plt

#Paths for data access
website_path = '../../website/'
results_path = '../../COVID_clinical/covid19_clean_data/'
output_path = '../results'
validation_path_greece = '../../COVID_clinical/covid19_greece/general_greek_registry.csv'
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

train_option = True

#Load model corresponding to model_type and lab
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    seedID = model_file['best seed']



#Load model corresponding to model_type and lab
with open(results_path+'/xgboost/'+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
    model_file = pickle.load(file)

    #Extract the inputs of the model
    model = model_file['model']
    features = model_file['json']
    columns = model_file['columns']
    imputer= model_file['imputer']
    seedID = model_file['best seed']

    if train_option:
        train = model_file['train']
        if model_type == 'mortality':
            y = train['Outcome']
        else:
            y=train['Swab']
        X = train.iloc[:,0:(len(train.columns)-1)]
    else:
        test = model_file['test']
        if model_type == 'mortality':
            y = test['Outcome']
        else:
            y=test['Swab']
        X = test.iloc[:,0:(len(test.columns)-1)]
  
    y_pred = model.predict(X)
    prob_pos = model.predict_proba(X)[:, 1]

        
    X['Outcome'] = y    
    X['Predicted'] = prob_pos





X_h = X.loc[X['ABG: Oxygen Saturation (SaO2)']>99]    
X_h['Predicted'].mean()
X_h['Outcome'].mean()
    
#Training set: 0.36, 0.5   

from math import sqrt 
import matplotlib

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


latexify()
# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()
plt.show()













































    
    
    