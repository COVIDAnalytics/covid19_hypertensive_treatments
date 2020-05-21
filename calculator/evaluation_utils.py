#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:50:02 2020

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

import numpy as np
import scipy.stats
from math import sqrt
import matplotlib
import subprocess
# import latexcodec


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
              'axes.labelsize': 20, # fontsize for x and y labels (was 12 and before 10)
              'axes.titlesize': 12,
              'font.size': 20, # was 12 and before 10
              'legend.fontsize': 20, # was 12 and before 10
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'text.usetex': False,
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


def get_model_outcomes(model_type, model_lab, website_path, results_path):

    print(model_type)
    print(model_lab)

    #Load model corresponding to model_type and lab
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    seedID = model_file['best seed']

    #Load model corresponding to model_type and lab
    with open(results_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
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

    y_pred = model.predict(X)
    prob_pos = model.predict_proba(X)[:, 1]

    return y, y_pred, prob_pos

def plot_calibration_curve(model_types, model_labs, website_path, results_path):
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
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path, results_path)

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
    plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+'_best_calibration_plot.png', bbox_inches='tight')

def plot_auc_curve(model_types, model_labs, website_path, results_path):
    fig = plt.figure(1, figsize=(10, 10))

    #Get the data
    for model_type in model_types:
        for model_lab in model_labs:

            name = "AUC of "+model_type+" "+model_lab+" "
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path, results_path)
            fpr, tpr, _ = metrics.roc_curve(y,  prob_pos)
            auc = metrics.roc_auc_score(y, prob_pos)
            plt.plot(fpr,tpr,label=name+str(round(auc,3)))
            plt.legend(loc=4)

    plt.ylabel("Sensitivity")
    plt.xlabel("1 - Specificity")
    plt.tight_layout()
    plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+'_best_auc_plot.png', bbox_inches='tight')

def plot_precision_recall_curve(model_types, model_labs, website_path, results_path):
    fig = plt.figure(1, figsize=(10, 10))

    #Get the data
    for model_type in model_types:
        for model_lab in model_labs:

            name = "Precision Recall of "+model_type+" "+model_lab+" "
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path, results_path)

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
            plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+'_best_precision_recall_plot.png', bbox_inches='tight')

def get_scores(y, y_pred, threshold, prob_pos):
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

   auc = metrics.roc_auc_score(y, prob_pos)


   colnames = ['AUC','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
   mat_met = pd.DataFrame(columns = colnames)
   newrow =[auc, threshold, ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR]
   mat_met.loc[0] = newrow
   return mat_met

def classification_report_table(model_types, model_labs, website_path, sensitivity_threshold, results_path):

    #Get the data ''

    cols = ['Model Type','Model Labs','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
    tab = pd.DataFrame(columns = cols)

    for model_type in model_types:
        for model_lab in model_labs:

            name = "Precision Recall of "+model_type+" "+model_lab+" "
            target_names = ['no outcome', 'outcome']
            y, y_pred, prob_pos = get_model_outcomes(model_type, model_lab, website_path, results_path)

            is_fpr, is_tpr, thresh = precision_recall_curve(y, prob_pos)

            colnames = ['Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
            sum_table = pd.DataFrame(columns = colnames)

            for t in thresh:
                y_pred = [1 if m > t else 0 for m in prob_pos]
                sum_table.loc[len(sum_table)] = get_scores(y, y_pred, t).loc[0]

            sum_table.to_csv('../results/performance_evaluation/performance_tables/final_model/'+model_type+'_'+model_lab+'_best_detailed_perforamance.csv', index=False)

            df = sum_table[sum_table['Sensitivity'] > sensitivity_threshold]
            x = df[df['Sensitivity'] == df['Sensitivity'].min()].iloc[0]
            x2 = pd.Series({'Model Type': model_type, 'Model Labs': model_lab})

            tab.loc[len(tab)] = x2.append(x)

    tab.to_csv('../results/performance_evaluation/performance_tables/final_model/'+'_'.join(model_types)+'_'.join(model_labs)+'_'+str(sensitivity_threshold)+'_best_summary_perforamance.csv', index=False)
    return tab

def get_model_outcomes_pickle(model_type, model_lab, results_path, seedID):

    #Load model corresponding to model_type and lab
    with open(results_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
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

    y_pred = model.predict(X)
    prob_pos = model.predict_proba(X)[:, 1]

    return y, y_pred, prob_pos


def plot_calibration_curve_bootstrap(model_types, model_labs, results_path, seeds):
    """Plot calibration curve for est w/o and with calibration. """
    fig_index=1

    latexify()
    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    #Get the data
    for model_type in model_types:
        for model_lab in model_labs:

            name = "Calibration Plot of "+model_type+" "+model_lab

            y=[]
            y_pred = []
            prob_pos = []

            for seedID in seeds:
                y_temp, y_pred_temp, prob_pos_temp = get_model_outcomes_pickle(model_type, model_lab, results_path, seedID)
                y.extend(list(y_temp))
                y_pred.extend(list(y_pred_temp))
                prob_pos.extend(list(prob_pos_temp))

            y = pd.Series(y)
            y_pred = pd.Series(y_pred)
            prob_pos = pd.Series(prob_pos)

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
    plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+'_'+str(seeds[0])+'_'+str(seeds[len(seeds)-1])+'_calibration_plot.pdf', bbox_inches='tight')


def plot_auc_curve_bootstrap(model_types, model_labs, results_path, seeds):
    latexify()
    fig = plt.figure(1, figsize=(10, 10))

    #Get the data
    for model_type in model_types:
        for model_lab in model_labs:
            name = "AUC of "+model_type+" "+model_lab+" "

            y=[]
            y_pred = []
            prob_pos = []

            for seedID in seeds:
                y_temp, y_pred_temp, prob_pos_temp = get_model_outcomes_pickle(model_type, model_lab, results_path, seedID)
                y.extend(list(y_temp))
                y_pred.extend(list(y_pred_temp))
                prob_pos.extend(list(prob_pos_temp))

            y = pd.Series(y)
            y_pred = pd.Series(y_pred)
            prob_pos = pd.Series(prob_pos)

            fpr, tpr, _ = metrics.roc_curve(y,  prob_pos)
            auc = metrics.roc_auc_score(y, prob_pos)
            plt.plot(fpr,tpr,linewidth=4.0,label=name+str(round(auc,3)))
            plt.legend(loc=4)

    params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.ylabel("Sensitivity")
    plt.xlabel("1 - Specificity")
    plt.tight_layout()
    plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+'_'+str(seeds[0])+'_'+str(seeds[len(seeds)-1])+'_auc_plot.pdf', bbox_inches='tight')


def plot_precision_recall_curve_bootstrap(model_types, model_labs, results_path, seeds):
    latexify()
    fig = plt.figure(1, figsize=(10, 10))

    #Get the data
    for model_type in model_types:
        for model_lab in model_labs:

            name = "Precision Recall of "+model_type+" "+model_lab+" "

            y=[]
            y_pred = []
            prob_pos = []

            for seedID in seeds:
                y_temp, y_pred_temp, prob_pos_temp = get_model_outcomes_pickle(model_type, model_lab, results_path, seedID)
                y.extend(list(y_temp))
                y_pred.extend(list(y_pred_temp))
                prob_pos.extend(list(prob_pos_temp))

            y = pd.Series(y)
            y_pred = pd.Series(y_pred)
            prob_pos = pd.Series(prob_pos)

            precision, recall, _ = precision_recall_curve(y,prob_pos)

            plt.plot(recall,precision,linewidth=4.0,label=name)
            plt.legend(loc=4)
            params = {'legend.fontsize': 20,
                      'legend.handlelength': 2}
            plt.rcParams.update(params)

            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # show the legend
            plt.legend()
            # show the plot
            plt.tight_layout()
            plt.savefig('../results/performance_evaluation/'+'_'.join(model_types)+'_'.join(model_labs)+'_'+str(seeds[0])+'_'+str(seeds[len(seeds)-1])+'_precision_recall_plot.pdf', bbox_inches='tight')

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_table_confidence_interval(tab2, confidence_level, colnames):

    tab3 = pd.DataFrame(columns = colnames)

    for i in colnames:
       m, l, u = (round(100*v,2) for v in mean_confidence_interval(tab2[i], confidence=confidence_level))
       val = str(m)+' ('+str(l)+','+str(u)+')'
       tab3[i] = pd.Series(val)

    return tab3


def classification_report_table_bootstrap(seeds, model_types, model_labs, results_path, sensitivity_threshold, confidence_level):

    #Get the data ''

    cols = ['Model Type','Model Labs','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
    tab = pd.DataFrame(columns = cols)

    for model_type in model_types:
        for model_lab in model_labs:

            tab2 = pd.DataFrame(columns = cols)
            name = "Precision Recall of "+model_type+" "+model_lab+" "

            for seedID in seeds:
                y, y_pred, prob_pos = get_model_outcomes_pickle(model_type, model_lab, results_path, seedID)
                is_fpr, is_tpr, thresh = precision_recall_curve(y, prob_pos)

                colnames = ['Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
                sum_table = pd.DataFrame(columns = colnames)

                for t in thresh:
                    y_pred = [1 if m > t else 0 for m in prob_pos]
                    sum_table.loc[len(sum_table)] = get_scores(y, y_pred, t).loc[0]


                df = sum_table[sum_table['Sensitivity'] > sensitivity_threshold]
                x = df[df['Sensitivity'] == df['Sensitivity'].min()].iloc[0]
                x2 = pd.Series({'Model Type': model_type, 'Model Labs': model_lab})

                tab2.loc[len(tab2)] = x2.append(x)

            tab3 = get_table_confidence_interval(tab2, confidence_level, colnames)
            tab3['Model Type'] = model_type
            tab3['Model Labs'] = model_lab
            tab3 = tab3[tab.columns]
            tab = tab.append(tab3)

    tab.to_csv('../results/performance_evaluation/performance_tables/bootstrap/'+'_'.join(model_types)+'_'+'_'.join(model_labs)+'_'+str(sensitivity_threshold)+'_'+str(seeds[0])+'_'+str(seeds[len(seeds)-1])+'_summary_perforamance.csv', index=False)
    return tab


def get_model_outcomes_pickle_validation(model_type, model_lab, website_path, results_path, validation_path):

    #Load validation population
    val_df = pd.read_csv(validation_path , encoding= 'unicode_escape')
    #Filter to only patients for which the outcome is known.
    val_df = val_df.loc[val_df['Outcome'].isin([0,1])]

    if model_lab == 'without_lab':
        val_df = val_df.rename(columns={'ABG: Oxygen Saturation (SaO2)':'SaO2'})

    if val_df['Body Temperature'].mean() < 45:
        val_df['Body Temperature'] = ((val_df['Body Temperature']/5)*9)+32

    #Load model corresponding to model_type and lab
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    seedID = model_file['best seed']

    #Load model corresponding to model_type and lab
    with open(results_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
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

    #Select only the relevant columns
    val_X0 = val_df[X.columns]
    val_y =  val_df['Outcome']

    val_X = pd.DataFrame(imputer.transform(val_X0))
    val_X.columns = list(val_X0.columns.values)

    y_pred = model.predict(val_X)
    prob_pos = model.predict_proba(val_X)[:, 1]

    return val_y,y_pred, prob_pos

def get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option):

    #Load model corresponding to model_type and lab
    with open(results_path+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

    #Extract the inputs of the model
    model = model_file['model']
    features = model_file['json']
    columns = model_file['columns']
    imputer= model_file['imputer']
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

    return y, y_pred, prob_pos

def get_binomial_CI(metric, n):

    interval = 1.96 * sqrt( (metric * (1 - metric)) / n)
    return metric, metric-interval, metric+interval


def get_validation_table_confidence_interval(tab2, cols2,n):

    tab3 = pd.DataFrame(columns = cols2)

    for i in cols2:
       m, l, u = (round(100*v,2) for v in get_binomial_CI(tab2[i], n))
       val = str(m)+' ('+str(l)+','+str(u)+')'
       tab3[i] = pd.Series(val)
    return tab3

def create_metrics_table(cohort, cols, model_type, model_lab, results_path, seedID, train_option, sensitivity_threshold):

    tab3 = pd.DataFrame(columns = cols)

    #First we add the training set
    y, y_pred, prob_pos = get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option)
    is_fpr, is_tpr, thresh = precision_recall_curve(y, prob_pos)
    n = len(y)

    colnames = ['AUC','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
    sum_table = pd.DataFrame(columns = colnames)

    for t in thresh:
        y_pred = [1 if m > t else 0 for m in prob_pos]
        sum_table.loc[len(sum_table)] = get_scores(y, y_pred, t, prob_pos).loc[0]

    df = sum_table[sum_table['Sensitivity'] > sensitivity_threshold]
    x = df[df['Sensitivity'] == df['Sensitivity'].min()].iloc[0]

    cols2 =  set(cols).intersection(colnames)
    tab2 = get_validation_table_confidence_interval(x, cols2, n)


    x2 = pd.Series({'Model Type': model_type, 'Model Labs': model_lab, 'Cohort': cohort,'N':n})

    tab3.loc[len(tab3)] = x2.append(tab2.loc[0,:])

    return tab3

def create_metrics_table_validation(cohort, cols, model_type, model_lab, website_path, results_path, validation_path, sensitivity_threshold):

    tab3 = pd.DataFrame(columns = cols)

    #First we add the training set
    y, y_pred, prob_pos = get_model_outcomes_pickle_validation(model_type, model_lab, website_path, results_path, validation_path)
    is_fpr, is_tpr, thresh = precision_recall_curve(y, prob_pos)
    n = len(y)

    colnames = ['AUC','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
    sum_table = pd.DataFrame(columns = colnames)

    for t in thresh:
        y_pred = [1 if m > t else 0 for m in prob_pos]
        sum_table.loc[len(sum_table)] = get_scores(y, y_pred, t, prob_pos).loc[0]

    df = sum_table[sum_table['Sensitivity'] > sensitivity_threshold]
    x = df[df['Sensitivity'] == df['Sensitivity'].min()].iloc[0]

    cols2 =  set(cols).intersection(colnames)
    tab2 = get_validation_table_confidence_interval(x, cols2, n)


    x2 = pd.Series({'Model Type': model_type, 'Model Labs': model_lab, 'Cohort': cohort,'N':n})

    tab3.loc[len(tab3)] = x2.append(tab2.loc[0,:])

    return tab3



def classification_report_table_validation(model_type, website_path, model_labs, results_path, validation_paths, sensitivity_threshold, output_path='.'):

    #Get the data
    cols = ['Model Type','Model Labs','Cohort','N','AUC','Threshold','Accuracy','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
    tab = pd.DataFrame(columns = cols)

    for model_lab in model_labs:

        #Load model corresponding to model_type and lab
        with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
            model_file = pickle.load(file)

        seedID = model_file['best seed']
        tab1 = create_metrics_table('Training Set', cols, model_type, model_lab, results_path, seedID, True, sensitivity_threshold)
        tab = tab.append(tab1)

        tab2 = create_metrics_table('Testing Set', cols, model_type, model_lab, results_path, seedID, False, sensitivity_threshold)
        tab = tab.append(tab2)

        tab3 = create_metrics_table_validation('Greek HC', cols, model_type, model_lab, website_path, results_path, validation_path = validation_paths[0], sensitivity_threshold=sensitivity_threshold)
        tab = tab.append(tab3)
        
        tab4 = create_metrics_table_validation('Sevilla', cols, model_type, model_lab, website_path, results_path, validation_path = validation_paths[1], sensitivity_threshold=sensitivity_threshold)
        tab = tab.append(tab4)

    tab.to_csv(os.path.join(output_path, model_type, 'summary_performance.csv'), index=False)

    return tab


def get_model_outcomes_algorithm_pickle(alg, model_type, model_lab, results_path, seedID):

    #Load model corresponding to model_type and lab
    with open(results_path+alg+'/'+model_type+'_'+model_lab+'/seed'+str(seedID)+'.pkl', 'rb') as file:
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

    y_pred = model.predict(X)
    prob_pos = model.predict_proba(X)[:, 1]

    return y, y_pred, prob_pos


def classification_report_table_mlmodels_bootstrapping(seeds, model_type, model_labs, results_path, sensitivity_threshold, confidence_level, output_path='.'):
    #Get the data
    cols = ['Algorithm','Model Labs','AUC','Threshold','Accuracy','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']

    algs_list = ['xgboost','lr','cart']
    tab = pd.DataFrame(columns = cols)

    for model_lab in model_labs:
        for alg in algs_list:

            tab2 = pd.DataFrame(columns = cols)

            for seedID in seeds:
                y, y_pred, prob_pos = get_model_outcomes_algorithm_pickle(alg, model_type, model_lab, results_path, seedID)
                is_fpr, is_tpr, thresh = precision_recall_curve(y, prob_pos)

                colnames = ['AUC','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
                sum_table = pd.DataFrame(columns = colnames)

                for t in thresh:
                    y_pred = [1 if m > t else 0 for m in prob_pos]
                    sum_table.loc[len(sum_table)] = get_scores(y, y_pred, t, prob_pos).loc[0]


                df = sum_table[sum_table['Sensitivity'] > sensitivity_threshold]
                x = df[df['Sensitivity'] == df['Sensitivity'].min()].iloc[0]
                x2 = pd.Series({'Algorithm':alg, 'Model Labs': model_lab})

                tab2.loc[len(tab2)] = x2.append(x)

            cols2 =  set(cols).intersection(colnames)

            tab3 = get_table_confidence_interval(tab2, confidence_level, cols2)
            tab3['Algorithm'] = alg
            tab3['Model Type'] = model_type
            tab3['Model Labs'] = model_lab
            tab3 = tab3[tab.columns]
            tab = tab.append(tab3)

    tab.to_csv(os.path.join(output_path, model_type, 'performance_comparison.csv'))
    return tab


def classification_report_table_mlmodels(seeds, model_type, model_labs, results_path, sensitivity_threshold, confidence_level, output_path='.'):
    #Get the data
    cols = ['Algorithm','Model Labs','AUC','Threshold','Accuracy','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']

    algs_list = ['xgboost','lr','cart']
    tab = pd.DataFrame(columns = cols)

    for model_lab in model_labs:
        for alg in algs_list:

            tab2 = pd.DataFrame(columns = cols)

            for seedID in seeds:
                y, y_pred, prob_pos = get_model_outcomes_algorithm_pickle(alg, model_type, model_lab, results_path, seedID)
                is_fpr, is_tpr, thresh = precision_recall_curve(y, prob_pos)
                n = len(y)

                colnames = ['AUC','Threshold','Accuracy','Sensitivity','Specificity','Precision','Negative predictive value','False positive rate','False negative rate','False discovery rate']
                sum_table = pd.DataFrame(columns = colnames)

                for t in thresh:
                    y_pred = [1 if m > t else 0 for m in prob_pos]
                    sum_table.loc[len(sum_table)] = get_scores(y, y_pred, t, prob_pos).loc[0]


                df = sum_table[sum_table['Sensitivity'] > sensitivity_threshold]
                x = df[df['Sensitivity'] == df['Sensitivity'].min()].iloc[0]
                x2 = pd.Series({'Algorithm':alg, 'Model Labs': model_lab})

                tab2.loc[len(tab2)] = x2.append(x)

            cols2 =  set(cols).intersection(colnames)

            tab3 = get_validation_table_confidence_interval(x, cols2, n)
            tab3['Algorithm'] = alg
            tab3['Model Type'] = model_type
            tab3['Model Labs'] = model_lab
            tab3 = tab3[tab.columns]
            tab = tab.append(tab3)

    tab.to_csv(os.path.join(output_path, model_type, 'performance_comparison.csv'))
    return tab



def plot_auc_curve_validation(model_type,website_path, model_labs, results_path,
                              validation_paths, output_path='.'):

    # TODO: Create 2 subplots: ROCs (123) with lab + without lab.

    latexify(columns=1)
    fig, axs = plt.subplots(1, len(model_labs), figsize=(10, 5))
    axs = {model_lab: axs[i] for i, model_lab in enumerate(model_labs)}

    for model_lab in model_labs:
        ax = axs[model_lab]

        #Then we add the validation set 1
        y1, y_pred1, prob_pos1 = get_model_outcomes_pickle_validation(model_type, model_lab, website_path, results_path, validation_paths[0])
        fpr1, tpr1, _ = metrics.roc_curve(y1,  prob_pos1)
        auc1 = metrics.roc_auc_score(y1, prob_pos1)
        name1 = "AUC of Greek HC "
        #  plt.close()

        #Load model corresponding to model_type and lab
        with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
            model_file = pickle.load(file)

        seedID = model_file['best seed']
        #  plt.close()

        #First we add the training set
        name2 = "AUC of Training Set "
        y2, y_pred2, prob_pos2 = get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option=True)
        fpr2, tpr2, _ = metrics.roc_curve(y2,  prob_pos2)
        auc2 = metrics.roc_auc_score(y2, prob_pos2)
        plt.close()

        #Then we add the testing set
        name3 = "AUC of Testing Set "
        y3, y_pred3, prob_pos3 = get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option=False)
        fpr3, tpr3, _ = metrics.roc_curve(y3,  prob_pos3)
        auc3 = metrics.roc_auc_score(y3, prob_pos3)
        
        plt.close()
        
        #Then we add the validation set 1
        y4, y_pred4, prob_pos4 = get_model_outcomes_pickle_validation(model_type, model_lab, website_path, results_path, validation_paths[1])
        fpr4, tpr4, _ = metrics.roc_curve(y4,  prob_pos4)
        auc4 = metrics.roc_auc_score(y4, prob_pos4)
        name4 = "AUC of Sevilla"



        ax.plot(fpr1, tpr1, label=name1+str(round(auc1, 3)))
        ax.plot(fpr2, tpr2, label=name2+str(round(auc2, 3)))
        ax.plot(fpr3, tpr3, label=name3+str(round(auc3, 3)))
        ax.plot(fpr4, tpr4, label=name4+str(round(auc4, 3)))
        ax.legend(loc=4)
        ax.set_title("Model " + model_lab.replace("_", " "))
        ax.set_ylabel("Sensitivity")
        ax.set_xlabel("1 - Specificity")

    fig.savefig(os.path.join(output_path, model_type, "auc_curves.pdf"),
                bbox_inches='tight')

def plot_precision_recall_curve_validation(model_type,website_path, model_labs, results_path, validation_paths):

    for model_lab in model_labs:

        #Then we add the validation set 1
        y1, y_pred1, prob_pos1 = get_model_outcomes_pickle_validation(model_type, model_lab, website_path, results_path, validation_paths[0])
        precision1, recall1, _ = precision_recall_curve(y1,prob_pos1)
        name1 = "Greek HC"

        plt.close()

        #Load model corresponding to model_type and lab
        with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
            model_file = pickle.load(file)

        seedID = model_file['best seed']

        plt.close()

        #First we add the training set
        name2 = "Training Set "
        y2, y_pred2, prob_pos2 = get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option=True)
        precision2, recall2, _ = precision_recall_curve(y2,prob_pos2)

        plt.close()

        #Then we add the testing set
        name3 = "Testing Set "
        y3, y_pred3, prob_pos3 = get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option=False)
        precision3, recall3, _ = precision_recall_curve(y3,prob_pos3)


        plt.close()
        fig = plt.figure(1, figsize=(10, 10))
        plt.plot(precision1, recall1,label=name1)
        plt.legend(loc=4)
        plt.plot(precision2, recall2,label=name2)
        plt.legend(loc=4)
        plt.plot(precision3, recall3,label=name3)
        plt.legend(loc=4)

        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.tight_layout()

        with open('../results/mortality/paper_plots/pr'+'_'+(model_lab)+'_plot.pkl','wb') as fid:
            pickle.dump(fig, fid)

        plt.savefig('../results/mortality/paper_plots/pr'+'_'+(model_lab)+'_plot.png', bbox_inches='tight')
        plt.close()

def plot_calibration_curve_validation(model_type,website_path, model_labs, results_path, validation_paths):

    for model_lab in model_labs:

        #Then we add the validation set 1
        y1, y_pred1, prob_pos1 = get_model_outcomes_pickle_validation(model_type, model_lab, website_path, results_path, validation_paths[0])
        fraction_of_positives1, mean_predicted_value1 = \
                calibration_curve(y1, prob_pos1, n_bins=10, strategy = 'quantile')

        name1 = "Greek HC"

        plt.close()

        #Load model corresponding to model_type and lab
        with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
            model_file = pickle.load(file)

        seedID = model_file['best seed']

        plt.close()

        #First we add the training set
        name2 = "Training Set "
        y2, y_pred2, prob_pos2 = get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option=True)
        fraction_of_positives2, mean_predicted_value2 = \
                calibration_curve(y2, prob_pos2, n_bins=10, strategy = 'quantile')

        plt.close()

        #Then we add the testing set
        name3 = "Testing Set "
        y3, y_pred3, prob_pos3 = get_model_outcomes_pickle_flexible(model_type, model_lab, results_path, seedID, train_option=False)
        fraction_of_positives3, mean_predicted_value3 = \
                calibration_curve(y3, prob_pos3, n_bins=10, strategy = 'quantile')


        plt.close()
        fig = plt.figure(1, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.plot(mean_predicted_value1, fraction_of_positives1, "s-",
                     label=(name1))
        ax2.hist(prob_pos1, range=(0, 1), bins=10, label=name1,
                     histtype="step", lw=2)

        ax1.plot(mean_predicted_value2, fraction_of_positives2, "s-",
                     label=(name2))
        ax2.hist(prob_pos2, range=(0, 1), bins=10, label=name2,
                     histtype="step", lw=2)

        ax1.plot(mean_predicted_value3, fraction_of_positives3, "s-",
                     label=(name3))
        ax2.hist(prob_pos3, range=(0, 1), bins=10, label=name3,
                     histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()

        with open('../results/mortality/paper_plots/calibration'+'_'+(model_lab)+'_plot.pkl','wb') as fid:
            pickle.dump(fig, fid)

        plt.savefig('../results/mortality/paper_plots/calibration'+'_'+(model_lab)+'_plot.png', bbox_inches='tight')
        plt.close()



