import numpy as np
import pandas as pd
import pickle

import analyzer.dataset as ds
import analyzer.loaders.hartford.hartford as hartford

from scipy import stats
#%%
row_order = ['Filter', 'Patient Count', 
    'Age',
    'Gender',
    'Cardiac Frequency',
    'ABG: Oxygen Saturation (SaO2)',
    'Body Temperature',
    'Alanine Aminotransferase (ALT)',
    'Aspartate Aminotransferase (AST)',
    'Blood Creatinine',
    'Glycemia',
    'Blood Urea Nitrogen (BUN)',
    'C-Reactive Protein (CRP)',
    'CBC: Hemoglobin',
    'CBC: Mean Corpuscular Volume (MCV)',
    'CBC: Platelets',
    'Potassium Blood Level',
    'Blood Sodium',
    'Prothrombin Time (INR)',
    'CBC: Leukocytes',
    'Cardiac dysrhythmias',
    'Chronic kidney disease',
    'Coronary atherosclerosis and other heart disease',
    'Diabetes']

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
    'Glycemia': 'Blood Glucose (mg/dL)',
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
    'ABG: Oxygen Saturation (SaO2)': 'Oxygen Saturation (\%)',
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
    'Glycemia': 'Blood Glucose (mg/dL)',
    'Potassium Blood Level': 'Potassium',
    'Prothrombin Time (INR)': 'INR',
    'Systolic Blood Pressure': 'Systolic BP (mmHg)',
    'SaO2': 'Oxygen Saturation (\%)',
    'Blood Calcium': 'Calcium',
    # 'ABG: PaO2': 'Partial Pressure\n Oxygen (PaO2)',
    # 'ABG: pH': 'Arterial Blood Gas pH',
    'Cholinesterase': 'Cholinesterase',
    'Respiratory Frequency': 'Respiratory Frequency',
    # 'ABG: MetHb': 'Arterial Blood Gas Methemoglobinemia',
    'Total Bilirubin': 'Total Bilirubin (mg/dL)',
    'Comorbidities': 'Comorbidities',
    'Diabetes': 'Diabetes',
    'Chronic kidney disease': 'Chronic\n kidney disease',
    'Cardiac dysrhythmias': 'Cardiac\ndysrhythmias',
    'Coronary atherosclerosis and other heart disease': 'Coronary atherosclerosis\n and other heart disease'
}


#%% Load function

def get_dataset_preload(model_type, model_lab, columns, imputer, impute = False, 
                combine = True):
    data_cremona = pd.read_csv('../../covid19_clean_data/clean_data/cremona_'+model_type+'_'+model_lab+'.csv')
    X_cremona = data_cremona.drop('Outcome', axis = 1); y_cremona = data_cremona['Outcome']
    X_cremona['Location'] = 'Cremona'
    data_spain = pd.read_csv('../../covid19_clean_data/clean_data/spain_'+model_type+'_'+model_lab+'.csv')
    X_spain = data_spain.drop('Outcome', axis = 1); y_spain = data_spain['Outcome']
    X_spain['Location'] = 'Spain'
    
    prediction = 'Outcome' if model_type == 'mortality' else 'Swab'
        
    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
        
    extra_data = False
    demographics_data = True
    
    if model_lab:
        o2_col = 'ABG: Oxygen Saturation (SaO2)'
        discharge_data = True
        comorbidities_data = True
        vitals_data = True
        lab_tests = True
        swabs_data = False
        mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
        print(name_datasets[mask])
    else:
        o2_col = 'SaO2'
        discharge_data = True
        comorbidities_data = True
        vitals_data = True
        lab_tests = False
        swabs_data = False
        mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
        print(name_datasets[mask])
    
    data_hartford = hartford.load_hartford('/nfs/sloanlab003/projects/cov19_calc_proj/hartford/hhc_inpatient_other.csv', 
      discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
    
    X_hartford, y_hartford =  ds.create_dataset(data_hartford,
                                          discharge_data,
                                          comorbidities_data,
                                          vitals_data,
                                          lab_tests,
                                          demographics_data,
                                          swabs_data,
                                          prediction = prediction)
    
    X_hartford['Location'] = 'Hartford'
    # Merge dataset
    
    X = pd.concat([X_cremona, X_spain, X_hartford], join='inner', ignore_index=True)
    y = pd.concat([y_cremona, y_spain, y_hartford], ignore_index=True)

    X, bounds_dict = ds.filter_outliers(X, filter_lb = 1.0, filter_ub = 99.0, o2 = o2_col)

    return X, y

def descriptive_table(data, features, short_version = False):

    cols_numeric = [i['name'] for i in features['numeric']]
    cols_categoric = [i['name'] for i in features['categorical']] + features['multidrop'][0]['vals']
    
    summary_numeric = np.transpose(data[cols_numeric].describe())
    summary_numeric['Type'] = 'Numeric'
    summary_numeric['output'] = round(summary_numeric['50%'],2).map(str) + " (" + round(summary_numeric['25%'],2).map(str) + "-" + round(summary_numeric['75%'],2).map(str) + ")"
    
    summary_categoric = np.transpose(data[cols_categoric].describe())
    summary_categoric['Type'] = 'Categoric'
    summary_categoric['output'] = round(summary_categoric['count']*summary_categoric['mean']).map(str) + " (" + round(summary_categoric['mean']*100,2).map(str) + "%)"
    
    summary_full = summary_numeric.append(summary_categoric)
    summary_full['Missing_Pct'] = round((1 - summary_full['count']/data.shape[0])*100,2).map(str)+'%'
    # summary_full.drop(["count"], axis = 1, inplace = True)
    summary_full.columns = ['Count', 'Mean', 'Standard Deviation', 
                            'Minimum', '25th Percentile', '50th Percentile',
                            '75th Percentile', 'Maximum', 'Type',  'Output', 'Percent Missing']
    final_cols = ['Type', 'Output', 'Percent Missing', 'Count', 'Mean', 'Standard Deviation',
              'Minimum', '25th Percentile', '50th Percentile','75th Percentile', 'Maximum']
    summary_full.loc['Patient Count'] = np.nan
    summary_full.loc['Patient Count','Output'] = data.shape[0] 

    if short_version:
        final_cols = ['Output', 'Percent Missing']
        
    return summary_full[final_cols]

# def generate_summary(data, features, title_mapping = None):
    
#     resAll = descriptive_table(data, features, title_mapping)
#     resAll['Outcome'] = 'All'
#     res1 = descriptive_table(data.query('Outcome == 1'), features)
#     res1['Outcome'] = 'Non-survivor' if model_type == 'mortality' else 'Infection'
#     res0 = descriptive_table(data.query('Outcome == 0'), features)
#     res0['Outcome'] = 'Survivor' if model_type == 'mortality' else 'No Infection'
    
#     summary_full = pd.concat([resAll, res1, res0])
#     summary_full.reset_index(inplace = True)
#     summary_full['index'] = summary_full['index'].replace(title_mapping, inplace=False)
    
#     return summary_full

def pairwise_compare(data_a, data_b, features, title_mapping = None, row_order = None,
                     filter_A = 'Group A', filter_B = 'Group B'):
    
    data = pd.concat([data_a, data_b])
    
    describe_all  = descriptive_table(data, features, short_version = True)
    describe_a = descriptive_table(data_a, features, short_version = True)
    describe_b = descriptive_table(data_b, features, short_version = True)
    
    describe_subgroups = describe_a.merge(describe_b, how = 'left', 
                     left_index = True, right_index = True,
                     suffixes = ('_A','_B'))
    
    describe_all = describe_all.merge(describe_subgroups, how = 'left',
                                       left_index = True, right_index = True)
    
    columns = describe_all.index.drop('Patient Count')
    
    sig_test = stats.ttest_ind(data_a[columns], data_b[columns], 
                               axis = 0, equal_var = False, nan_policy = 'omit')
    df_sig = pd.DataFrame(np.transpose(sig_test)[:,1], columns = ['p-Value'])
    df_sig.index = columns
     
    describe_all = describe_all.merge(df_sig, how = 'left',
                                       left_index = True, right_index = True)
    
    describe_all.loc['Filter'] = np.nan
    describe_all.loc['Filter',['Output', 'Output_A', 'Output_B']] = ['All', filter_A, filter_B]
    
    if row_order != None:
        describe_all = describe_all.reindex(row_order)
    
    describe_all.reset_index(inplace = True)
    
    if title_mapping != None:
        describe_all['index'] = describe_all['index'].replace(title_mapping, inplace=False)
    
    return describe_all