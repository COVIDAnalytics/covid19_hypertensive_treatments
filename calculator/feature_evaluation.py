import os
#  os.chdir('git/covid19_calculator/calculator/')

#%% Load packages
import itertools

import evaluation.descriptive as ev
import evaluation.importance as imp

#%% Load mapping (from website/risk_calculators/utils.py)

title_mapping = {
    'ABG: Oxygen Saturation (SaO2)': 'Oxygen Saturation (\%)',
    'Alanine Aminotransferase (ALT)': 'Alanine Aminotransferase (U/L)',
    'Age': 'Age',
    'Aspartate Aminotransferase (AST)': 'Aspartate Aminotransferase (U/L)',
    'Blood Creatinine': 'Blood Creatinine (mg/dL)',
    'Blood Sodium': 'Sodium (mmol/L)',
    'Blood Urea Nitrogen (BUN)': 'Blood Urea Nitrogen (mg/dL)',
    'Body Temperature': 'Temperature (F)',
    'C-Reactive Protein (CRP)':  'C-Reactive Protein (mg/L)',
    'CBC: Hemoglobin': 'Hemoglobin (g/dL)',
    'CBC: Leukocytes': 'White Blood Cells (1000/muL)',
    'CBC: Mean Corpuscular Volume (MCV)': 'Mean Corpuscular Volume (fL)',
    'CBC: Platelets': 'Platelets (1000/muL)',
    'CBC: Red cell Distribution Width (RDW)': 'Red Cell Distribution Width (%)',
    'Cardiac Frequency': 'Heart Rate (bpm)',
    'Cardiac dysrhythmias': 'Cardiac dysrhythmias',
    'Gender': 'Gender (M/F)',
    'Glycemia': 'Blood Glucose',
    'Potassium Blood Level': 'Potassium',
    'Prothrombin Time (INR)': 'Prothrombin Time (INR)',
    'Systolic Blood Pressure': 'Systolic Blood Pressure (mmHg)',
    'SaO2': 'Oxygen Saturation (\%)',
    'Blood Calcium': 'Calcium (mg/dL)',
    'ABG: PaO2': 'Partial Pressure Oxygen (PaO2)',
    'ABG: pH': 'Arterial Blood Gas pH',
    'Cholinesterase': 'Cholinesterase',
    'Respiratory Frequency': 'Respiratory Frequency (bpm)',
    'ABG: MetHb': 'Arterial Blood Gas Methemoglobinemia',
    'Total Bilirubin': 'Total Bilirubin (mg/dL)',
    'Comorbidities': 'Comorbidities',
    'Diabetes': 'Diabetes',
    'Chronic kidney disease': 'Chronic kidney disease',
    'Cardiac dysrhythmias': 'Cardiac dysrhythmias',
    'Coronary atherosclerosis and other heart disease': 'Coronary atherosclerosis and other heart disease'
}

title_mapping_summary = {
    'ABG: Oxygen Saturation (SaO2)': 'Oxygen\n Saturation (\%)',
    'Alanine Aminotransferase (ALT)': 'ALT (U/L)',
    'Age': 'Age',
    'Aspartate Aminotransferase (AST)': 'AST (U/L)',
    'Blood Creatinine': 'Creatinine (mg/dL)',
    'Blood Sodium': 'Sodium (mmol/L)',
    'Blood Urea Nitrogen (BUN)': 'BUN (mg/dL)',
    'Body Temperature': 'Temperature (F)',
    'C-Reactive Protein (CRP)':  'CRP (mg/L)',
    'CBC: Hemoglobin': 'Hemoglobin (g/dL)',
    'CBC: Leukocytes': 'WBC (1000/muL)',
    'CBC: Mean Corpuscular Volume (MCV)': 'MCV (fL)',
    'CBC: Platelets': 'Platelets (1000/muL)',
    'CBC: Red cell Distribution Width (RDW)': 'RDW (\%)',
    'Cardiac Frequency': 'Heart Rate (bpm)',
    'Cardiac dysrhythmias': 'Cardiac\n dysrhythmias',
    'Gender': 'Gender (M/F)',
    'Glycemia': 'Blood\n Glucose',
    'Potassium Blood Level': 'Potassium',
    'Prothrombin Time (INR)': 'INR',
    'Systolic Blood Pressure': 'Systolic BP (mmHg)',
    'SaO2': 'Oxygen\n Saturation (\%)',
    'Blood Calcium': 'Calcium',
    # 'ABG: PaO2': 'Partial Pressure\n Oxygen (PaO2)',
    # 'ABG: pH': 'Arterial Blood Gas pH',
    'Cholinesterase': 'Cholinesterase',
    'Respiratory Frequency': 'Respiratory\n Frequency',
    # 'ABG: MetHb': 'Arterial Blood Gas Methemoglobinemia',
    'Total Bilirubin': 'Total\n Bilirubin (mg/dL)',
    'Comorbidities': 'Comorbidities',
    'Diabetes': 'Diabetes',
    'Chronic kidney disease': 'Chronic\n kidney disease',
    'Cardiac dysrhythmias': 'Cardiac\ndysrhythmias',
    'Coronary atherosclerosis and other heart disease': 'Coronary atherosclerosis\n and other heart disease'
}

#%% Distribution by Feature

## Set paths
#  website_path = '/Users/hollywiberg/git/website/'
#  data_path = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_clean_data/xgboost/'
website_path = '../../../website/'
data_path = '../../covid19_clean_data/xgboost/'

# subgroups = {'Age < 55': 'Age_below55',
#     'Age >= 55 & Age < 80': 'Age_55to79',
#     'Age >= 80': 'Age_atleast80',
#     'Gender ==  0': 'Male',
#     'Gender == 1': 'Female'}

for model_type, model_lab in itertools.product(['infection', 'mortality'], ['with_lab', 'without_lab']):
    # for model_type, model_lab in itertools.product(['mortality'],['without_lab']):
    print("Model: %s, %s" % (model_type, model_lab))
    save_path = '../results/'+model_type+'/model_'+model_lab+'/'
    imp.feature_importance(model_type, model_lab, website_path, data_path, save_path, title_mapping_summary,
                           latex=True, feature_limit=10, dependence_plot=True)


    #  imp.feature_importance_website(model_type, model_lab, website_path, data_path, save_path, title_mapping_summary,
    #                                 feature_limit=10)





    # for s in subgroups.keys():
    #     print("Subgroup: %s" % s)
    #     imp.feature_importance(model_type, model_lab, website_path, data_path, save_path, title_mapping_summary,
    #                            latex = True, feature_limit = 10, dependence_plot = False,
    #                            data_filter = s, suffix_filter = subgroups[s])
