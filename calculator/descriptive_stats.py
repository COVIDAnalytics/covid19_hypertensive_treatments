import numpy as np
import pandas as pd
import os
import pickle

# Other packages
import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json, change_SaO2
import analyzer.dataset as ds

import shap
import matplotlib.pyplot as plt

#%% Load mapping (from website/risk_calculators/utils.py)

oxygen = 'Oxygen Saturation'

title_mapping = {
    'ABG: Oxygen Saturation (SaO2)': oxygen,
    'Alanine Aminotransferase (ALT)': 'Alanine Aminotransferase (ALT)',
    'Age': 'Age',
    'Aspartate Aminotransferase (AST)': 'Aspartate Aminotransferase',
    'Blood Creatinine': 'Creatinine',
    'Blood Sodium': 'Sodium',
    'Blood Urea Nitrogen (BUN)': 'Blood Urea Nitrogen (BUN)',
    'Body Temperature': 'Temperature',
    'C-Reactive Protein (CRP)':  'C-Reactive Protein',
    'CBC: Hemoglobin': 'Hemoglobin',
    'CBC: Leukocytes': 'Leukocytes',
    'CBC: Mean Corpuscular Volume (MCV)': 'Mean Corpuscular Volume',
    'CBC: Platelets': 'Platelets',
    'CBC: Red cell Distribution Width (RDW)': 'Red Cell Distribution Width (RDW)',
    'Cardiac Frequency': 'Heart Rate',
    'Cardiac dysrhythmias': 'Cardiac dysrhythmias',
    'Gender' : 'Gender',
    'Glycemia': 'Glycemia',
    'Potassium Blood Level': 'Potassium',
    'Prothrombin Time (INR)': 'Prothrombin Time',
    'Systolic Blood Pressure': 'Systolic Blood Pressure (SYS)',
    'SaO2': oxygen,
    'Blood Calcium': 'Calcium',
    'ABG: PaO2': 'Partial Pressure Oxygen (PaO2)',
    'ABG: pH': 'Arterial Blood Gas pH',
    'Cholinesterase': 'Cholinesterase',
    'Respiratory Frequency': 'Respiratory Frequency',
    'ABG: MetHb': 'Arterial Blood Gas Methemoglobinemia',
    'Total Bilirubin': 'Total Bilirubin',
    'Comorbidities':'Comorbidities'
}


# vitals = ['Age'
# 'Body Temperature',
# 'Cardiac Frequency',
# 'Gender',
# 'SaO2',
# 'ABG: Oxygen Saturation (SaO2)',
# 'Systolic Blood Pressure',
# 'Respiratory Frequency']
        
    
# labs = ['Alanine Aminotransferase (ALT)',
#     'Aspartate Aminotransferase (AST)',
#     'Blood Creatinine',
#     'Blood Sodium',
#     'Blood Urea Nitrogen (BUN)',
#     'C-Reactive Protein (CRP)',
#     'CBC: Hemoglobin',
#     'CBC: Leukocytes',
#     'CBC: Mean Corpuscular Volume (MCV)',
#     'CBC: Platelets',
#     'CBC: Red cell Distribution Width (RDW)',
#     'Glycemia',
#     'Potassium Blood Level',
#     'Prothrombin Time (INR)',
#     'Blood Calcium',
#     'Cholinesterase',
#     'Total Bilirubin']

#%% Distribution by Feature

cols_numeric = [i['name'] for i in features['numeric']]
cols_categoric = X.columns.difference(cols_numeric)
summary_numeric = np.transpose(X[cols_numeric].describe())
summary_numeric['Type'] = 'Numeric'
summary_categoric = np.transpose(X[cols_categoric].describe())
summary_categoric['Type'] = 'Categoric'

summary_full = summary_numeric.append(summary_categoric)
summary_full.drop(["count"], axis = 1, inplace = True)
summary_full.columns = ['Mean', 'Standard Deviation', 
                        'Minimum', '25th Percentile', '50th Percentile',
                        '75th Percentile', 'Maximum', 'Type']
summary_full["Feature"] = summary_full.index
summary_full["Feature_Recoded"] = summary_full["Feature"].replace(title_mapping, inplace=False)

final_cols = ['Feature_Recoded', 'Type', 'Mean', 'Standard Deviation',
              'Minimum', '25th Percentile', '50th Percentile','75th Percentile', 'Maximum']
summary_full[final_cols].sort_values(by = ['Type', 'Feature_Recoded']).to_csv('../results/'+model_type+'/model_'+model_lab+'/descriptive_statistics.csv',
                                index = False)
