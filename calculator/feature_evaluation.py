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

for model_type, model_lab in itertools.product(['infection','mortality'],['with_lab','without_lab']):
# for model_type, model_lab in itertools.product(['mortality'],['without_lab']):
    print("Model: %s, %s" %(model_type, model_lab))
    save_path = '../results/'+model_type+'/model_'+model_lab+'/'
    imp.feature_importance(model_type, model_lab, website_path, data_path, save_path, title_mapping, 
                           latex = True, feature_limit = 10)
    # summary = ev.generate_summary(model_type, model_lab, website_path, title_mapping)
    # summary.to_csv(save_path+'descriptive_statistics.csv', index = False)
    
# model_type = 'mortality'; model_lab = 'without_lab'
# with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
#         model_file = pickle.load(file)


# ft_recode = []
# for i in X.columns:
#     ft_recode.append(title_mapping[i])
    