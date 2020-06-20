"""
Script for aggregating and evaluating performance of predictive models across treatments

Created on Wed Jun 17 13:57:19 2020

@author: hollywiberg
"""

#%% Prepare environment
import os

os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
# import evaluation.importance as imp
# import matplotlib.pyplot as plt
# import analyzer.optuna as o
import pandas as pd

#%% Set Problem Parameters
#Paths for data access
        
data_path = '../../covid19_hope/hope_matched.csv'
save_path = '../../covid19_treatments_results/summary/'

dataset = "hope"
treatment_list = ['Chloroquine_Only', 'All', 'Chloroquine_and_Anticoagulants',
                              'Chloroquine_and_Antivirals', 'Non-Chloroquine']


# algorithm_list = range(0,len(o.algo_names))
algorithm_list = ['lr','rf','cart','xgboost','oct']


#%% Generate predictions

split = 'bycountry'
X_train, Z_train, y_train, X_test, Z_test, y_test = u.load_data(data_path, split = split)

data_version = 'train'

if data_version == 'train':
    X = X_train
    Z = Z_train
    y = y_train
else: 
    X = X_test
    Z = Z_test
    y = y_test

result = pd.concat([u.algorithm_predictions(X, algorithm = alg, dataset = dataset, treatment_list = treatment_list) 
                    for alg in algorithm_list], axis = 0)

# Find optimal prescription across methods
result['Prescribe'] = result.idxmin(axis=1)
result['Prescribe_Prediction'] = result.min(axis=1)

result.to_csv(save_path+data_version+'_bypatient_allmethods.csv')

#%% Evaluate Methods
# =============================================================================
# Summary contains:
# - Mean probability for each treatment (averaged across methods)
# - Prescribe: most common prescription (mode of prescriptions across methods), list if there are ties
# - REGIMEN: true prescribed treatment
# - Match: indicator of whether true regimen is (in list of) optimal prescription(s)
# =============================================================================
summary = pd.concat([result.groupby('ID')[treatment_list].agg({'mean'}),
          result.groupby('ID')['Prescribe'].apply(
              lambda x:  ', '.join(pd.Series.mode(x).sort_values())),  
          Z, y], axis=1)
summary['Match'] = [x.replace(' ','_') in(y) for x,y in zip(summary['REGIMEN'], summary['Prescribe'])]
summary.to_csv(save_path+data_version+'_bypatient_summary.csv')

# =============================================================================
# Prescription_summary contains:
# - Frequencies of prescriptions and how often they were actually prescribed
# =============================================================================
prescription_summary = pd.crosstab(index = summary.Prescribe, columns = summary.Match, 
                                   margins = True, margins_name = 'Total')
prescription_summary.columns = ['No Match', 'Match', 'Total']
prescription_summary.drop('Total',axis=0)
prescription_summary.sort_values('Total', ascending = False, inplace = True)
prescription_summary.to_csv(save_path+data_version+'_bytreatment_summary.csv')
