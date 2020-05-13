import os
os.chdir('git/covid19_calculator/calculator/')

#%% Load packages
import itertools

import evaluation.descriptive as ev
import evaluation.importance as imp

#%% Load mapping (from website/risk_calculators/utils.py)

title_mapping = {
    'ABG: Oxygen Saturation (SaO2)': 'Oxygen Saturation',
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
    'Glycemia': 'Blood Glucose',
    'Potassium Blood Level': 'Potassium',
    'Prothrombin Time (INR)': 'Prothrombin Time',
    'Systolic Blood Pressure': 'Systolic Blood Pressure (SYS)',
    'SaO2': 'Oxygen Saturation',
    'Blood Calcium': 'Calcium',
    'ABG: PaO2': 'Partial Pressure Oxygen (PaO2)',
    'ABG: pH': 'Arterial Blood Gas pH',
    'Cholinesterase': 'Cholinesterase',
    'Respiratory Frequency': 'Respiratory Frequency',
    'ABG: MetHb': 'Arterial Blood Gas Methemoglobinemia',
    'Total Bilirubin': 'Total Bilirubin',
    'Comorbidities':'Comorbidities',
    'Diabetes': 'Diabetes',
    'Chronic kidney disease': 'Chronic kidney disease',
    'Cardiac dysrhythmias': 'Cardiac dysrhythmias',
    'Coronary atherosclerosis and other heart disease': 'Coronary atherosclerosis and other heart disease'
}

#%% Distribution by Feature

## Set paths
website_path = '/Users/hollywiberg/git/website/'
data_path = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_clean_data/'

subgroups = {'Age < 55': 'Age_below55',
    'Age >= 55 & Age < 80': 'Age_55to79',
    'Age >= 80': 'Age_atleast80',
    'Gender ==  0': 'Male',
    'Gender == 1': 'Female'}

for model_type, model_lab in itertools.product(['infection','mortality'],['with_lab','without_lab']):
# for model_type, model_lab in itertools.product(['mortality'],['without_lab']):
    save_path = '../results/'+model_type+'/model_'+model_lab+'/'
    imp.feature_importance_website(model_type, model_lab, website_path, data_path, save_path, title_mapping, 
                               feature_limit = 10)
    print("Model: %s, %s" %(model_type, model_lab))
    for s in subgroups.keys():
        print("Subgroup: %s" % s)
        imp.feature_importance(model_type, model_lab, website_path, data_path, save_path, title_mapping, 
                               latex = True, feature_limit = 10, dependence_plot = False,
                               data_filter = s, suffix_filter = subgroups[s])
