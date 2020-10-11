#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:27:13 2020

@author: hollywiberg
"""


import os

# os.chdir('/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_calculator/calculator')

import evaluation.treatment_utils as u
import evaluation.descriptive_utils as d
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
from pathlib import Path


from sklearn.metrics import (brier_score_loss, precision_score, recall_score,accuracy_score,
                             f1_score, confusion_matrix)
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
from sklearn import metrics

#%% Version-specific parameters

# version = 'matched_single_treatments_hope_bwh/'
# train_file = '_hope_matched_all_treatments_train.csv'
# data_list = ['train','test','validation_all','validation_partners']

version = 'matched_single_treatments_hypertension/'
train_file = '_hope_hm_cremona_matched_all_treatments_train.csv'
data_list = ['train','test','validation_all','validation_partners',
              'validation_hope','validation_hope_italy']

threshold = 0.05
weighted_status = 'no_weights'
#%% General parameters

data_path = '../../covid19_treatments_data/'+version
        
preload = True
matched = True
match_status = 'matched' if matched else 'unmatched'

SEEDS = range(1, 2)
algorithm_list = ['rf','cart','oct','xgboost','qda','gb']
# prediction_list = ['COMORB_DEATH','OUTCOME_VENT','DEATH','HF','ARF','SEPSIS']
outcome = 'COMORB_DEATH'

treatment = 'ACEI_ARBS'
treatment_list = [treatment, 'NO_'+treatment]

training_set_name = treatment+train_file

results_path = '../../covid19_treatments_results/'
version_folder = version+str(treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

training_set_name = treatment+train_file

#%% Predictive Methods table

columns = pd.MultiIndex.from_product([data_list,treatment_list])
index = algorithm_list

res = pd.DataFrame(columns = pd.MultiIndex.from_product([data_list,treatment_list]), 
             index = algorithm_list)

for data_version in data_list:
    pred_results = pd.read_csv(save_path+data_version+'_'+match_status+'_performance_allmethods.csv')
    pred_results.set_index('Algorithm', inplace = True)
    assert np.all(pred_results.index == algorithm_list)
    res[data_version] = pred_results
  
res.index = res.index.str.upper()
res.loc['Average AUC',:] = res.mean(axis=0)

col_dict = dict({'train':'Training Data', 'test':'Testing Data', 'validation_all':'Validation Data',
                 'ACEI_ARBS':'ACEI/ARBs','NO_ACEI_ARBS':'No ACEI/ARBs'})
res.rename(col_dict, axis = 1, inplace = True)

res.to_latex(buf = save_path+'latex_predictive_table.txt', 
             column_format = 'l'+'c'*res.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c')

#%% Predictive with LR
algorithm_list_lr = ['lr'] + algorithm_list 
columns = pd.MultiIndex.from_product([data_list,treatment_list])
index = algorithm_list_lr

res = pd.DataFrame(columns = pd.MultiIndex.from_product([data_list,treatment_list]), 
              index = algorithm_list_lr)

for data_version in data_list:
    pred_results = pd.read_csv(save_path+data_version+'_'+match_status+'_performance_allmethods_withlr.csv')
    pred_results.set_index('Algorithm', inplace = True)
    assert np.all(pred_results.index == algorithm_list_lr)
    res[data_version] = pred_results
  
res.index = res.index.str.upper()
res.loc['Average AUC',:] = res.mean(axis=0)

col_dict = dict({'train':'Training Data', 'test':'Testing Data', 'validation_all':'Validation Data',
                  'ACEI_ARBS':'ACEI/ARBs','NO_ACEI_ARBS':'No ACEI/ARBs'})
res.rename(col_dict, axis = 1, inplace = True)

res.to_latex(buf = save_path+'latex_predictive_table_withlr.txt', 
              column_format = 'l'+'c'*res.shape[1],
              float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c')


#%% Prescriptive Methods table

summ = pd.read_csv(save_path+'matched_metrics_summary.csv')

summ = summ.query('threshold == %s & weighted_status == "%s"' % (threshold, weighted_status))
summ = summ.loc[summ.data_version.isin(data_list),:]
summ = summ[['data_version', 'match_rate', 'presc_count', 
             'average_auc', 'PE', 'CPE', 'pr_low', 'pr_high']]

# summ.data_version = summ.data_version.str.capitalize()
summ.rename({'data_version':'Data Version', 'match_rate':'Match Rate',
            'presc_count':'Presc. Count', 'average_auc':'Avg. AUC', 
            'pr_low':'PR (Low)', 'pr_high':'PR (High)'}, axis=1, inplace = True)

summ.set_index('Data Version', inplace = True)
summ = summ.rename(index = col_dict)

summ.to_latex(buf = save_path+'latex_prescriptive_table_t'+str(threshold)+'.txt', 
             column_format = 'l'+'c'*summ.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c',
             index_names = False)

#%% Comparison of threshholds on the test set

summ = pd.read_csv(save_path+'matched_metrics_summary.csv')

summ = summ.query('data_version == "%s" & weighted_status == "%s"' % ('test', weighted_status))
# summ = summ.loc[summ.data_version.isin(data_list),:]
summ = summ[['threshold', 'match_rate', 'presc_count', 
             'average_auc', 'PE', 'CPE', 'pr_low', 'pr_high']]

summ.rename({'threshold':'Threshold (%)', 'match_rate':'Match Rate',
            'presc_count':'Presc. Count', 'average_auc':'Avg. AUC', 
            'pr_low':'PR (Low)', 'pr_high':'PR (High)'}, axis=1, inplace = True)

summ.set_index('Threshold (%)', inplace = True)
summ.to_latex(buf = save_path+'latex_threshold_table.txt', 
             column_format = 'l'+'c'*summ.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c',
             index_names = False)

#%% Appendix - voting of individual methods

data_version = 'test'
agr = pd.read_csv(save_path+data_version+'_'+match_status+'_t'+str(threshold)+'_'+'agreement_byalgorithm.csv')

agr.set_index('algorithm', inplace = True)
agr = agr[['prescription_count','agreement_no_weights']]
agr['agreement_no_weights'] = ["{:.1%}".format(x) for x in agr['agreement_no_weights']]
agr.rename({'prescription_count':'Prescription Count', 'agreement_no_weights':'Agreement with Prescription'}, 
           axis=1, inplace = True)
agr.index = agr.index.str.upper()
agr.rename({'VOTE':'Prescription'}, axis = 0, inplace = True)

agr.to_latex(buf = save_path+'latex_agreement_table_t'+str(threshold)+'.txt', 
             column_format = 'l'+'c'*agr.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c',
             index_names = False)
# preed_reseults
                  
#%% Descriptive analysis: pre- vs. post-matching

data_mgb = pd.read_csv('../../covid19_treatments_data/'+version+treatment+'_hope_hm_cremona_all_treatments_validation_partners.csv')
data_mgb['ACEI_ARBS'] = data_mgb['REGIMEN'].apply(lambda x: 1 if x == treatment else 0)

data_pre = pd.read_csv('../../covid19_treatments_data/hope_hm_cremona_data_clean_imputed_addl_outcomes.csv')
data_pre = data_pre.query('HYPERTENSION == 1')
data_pre['REGIMEN'] = data_pre[treatment].apply(lambda x: treatment if x == 1 else 'NO_'+treatment)
data_pre = pd.concat([data_pre, data_mgb], axis=0, ignore_index = False)

data_pre = pd.get_dummies(data_pre, columns = ['GENDER','RACE'], drop_first = True)
data_pre['Version'] = 'Pre-Match'

# data_post0 = pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_matched.csv')
data_post_train = pd.read_csv('../../covid19_treatments_data/'+version+treatment+train_file)
data_post_test = pd.read_csv('../../covid19_treatments_data/'+version+treatment+train_file.replace('train','test'))
data_post = pd.concat([data_post_train, data_post_test], axis=0)
# data_pre['REGIMEN'] = data_pre[treatment].apply(lambda x: treatment if x == 1 else 'NO_'+treatment)
data_post = pd.get_dummies(data_post, columns = ['GENDER','RACE'], drop_first = True)
data_post['Version'] = 'Post-Match'

features = {'categorical':['GENDER_MALE', 'RACE_CAUC', 'RACE_LATIN', 'RACE_ORIENTAL', 'RACE_OTHER',
       'SAT02_BELOW92',
       'BLOOD_PRESSURE_ABNORMAL_B', 'DDDIMER_B', 'PCR_B', 'TRANSAMINASES_B',
       'LDL_B', 
       'DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
       'RENALINSUF', 'ANYLUNGDISEASE', 'AF', 'VIH', 'ANYHEARTDISEASE',
       'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE', 'LIVER_DISEASE',
       'CANCER',
       'CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS',
       'DEATH','COMORB_DEATH','HF', 'ARF', 'SEPSIS', 'EMBOLIC', 'OUTCOME_VENT'],
                'numeric':['AGE','MAXTEMPERATURE_ADMISSION','CREATININE', 'SODIUM', 
                           'LEUCOCYTES', 'LYMPHOCYTES','HEMOGLOBIN', 'PLATELETS'],
                'multidrop':[]}

col_order = ['Patient Count', 'AGE','GENDER_MALE', 
             'RACE_CAUC', 'RACE_LATIN', 'RACE_ORIENTAL', 'RACE_OTHER', 
             'MAXTEMPERATURE_ADMISSION', 'CREATININE', 'SODIUM', 'LEUCOCYTES', 'LYMPHOCYTES', 'HEMOGLOBIN', 'PLATELETS',
             'SAT02_BELOW92', 'BLOOD_PRESSURE_ABNORMAL_B', 'DDDIMER_B', 'PCR_B', 'TRANSAMINASES_B', 'LDL_B', 
             'DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY', 'RENALINSUF', 'ANYLUNGDISEASE', 'AF', 'VIH', 'ANYHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE', 'LIVER_DISEASE', 'CANCER', 
             'CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS', 
             'DEATH','COMORB_DEATH','HF', 'ARF', 'SEPSIS', 'EMBOLIC', 'OUTCOME_VENT']

col_mapping = {'ACEI_ARBS': 'ACE Inhibitors or ARBs',
     'AF': 'Atrial Fibrillation',
     'AGE': 'Age',
     'ANTIBIOTICS': 'Antibiotics',
     'ANTICOAGULANTS':'Anticoagulants',
     'ANTIVIRAL':'Antivirals',
     'ANYCEREBROVASCULARDISEASE':'Cerebrovascular Disease',
     'ANYHEARTDISEASE':'Heart Disease',
     'ANYLUNGDISEASE':'Lung Disease',
     # 'BLOOD_PRESSURE_ABNORMAL_B':'Low systolic blood pressure (<100 mm Hg)',
     'BLOOD_PRESSURE_ABNORMAL_B':'Low systolic BP',
     'CANCER':'Cancer',
     'CLOROQUINE':'Hydroxychloroquine',
     'CONECTIVEDISEASE':'Connective Tissue Disease',
     'CREATININE':'Creatinine',
     # 'DDDIMER_B':'Elevated D-Dimer (>0.5 mg/L)',
     'DDDIMER_B':'Elevated D-Dimer',
     'DIABETES':'Diabetes',
     'DISLIPIDEMIA':'Dislipidemia',
     'GENDER_MALE':'Gender = Male',
     'HEMOGLOBIN':'Hemoglobin',
     'HYPERTENSION':'Hypertension',
     'INTERFERONOR':'Interferons',
     # 'LDL_B':'Elevated Lactic Acid Dehydrogenase (>480 U/L)',
     'LDL_B':'Elevated LDH',
     'LEUCOCYTES':'White Blood Cell Count',
     'LIVER_DISEASE':'Liver Disease',
     'LYMPHOCYTES':'Lymphocytes',
     'MAXTEMPERATURE_ADMISSION':'Temperature',
     'OBESITY':'Obesity',
     # 'PCR_B':'Elevated C-Reactive Protein (>10 mg/L)',
     'PCR_B':'Elevated CRP',
     'PLATELETS':'Platelets',
     'RACE_CAUC':'Race = Caucasian',
     'RACE_LATIN':'Race = Hispanic',
     'RACE_ORIENTAL':'Race = Asian',
     'RACE_OTHER':'Race = Other',
     'RENALINSUF':'Renal Insufficiency',
     # 'SAT02_BELOW92':'Low Oxygen Saturation (< 92)',
     'SAT02_BELOW92':'Low Oxygen Saturation',
     'SODIUM':'Blood Sodium',
     'CORTICOSTEROIDS':'Corticosteroids',
     'TOCILIZUMAB':'Tocilizumab',
     # 'TRANSAMINASES_B':'Elevated Transaminase (>40 U/L)',
     'TRANSAMINASES_B':'Elevated Transaminase',
     'VIH':'HIV',
     'HF':'Heart Failure',
     'ARF':'Acute Renal Failure',
     'SEPSIS':'Sepsis',
     'EMBOLIC':'Embolic Event',
     'OUTCOME_VENT':'Mechanical Ventilation',
     'COMORB_DEATH':'Mortality/Morbidity',
     'DEATH':'Death'}

columns = pd.MultiIndex.from_product([['Pre-Match','Post-Match'],treatment_list])
index = col_order


#%% Create pre/post data comparison
# df = pd.concat([data_pre[col_order], data_post[col_order]], axis=0)
df = data_pre.query('COUNTRY == "Spain"') ## restrict to data in derivation cohort

desc_list = []
for p in treatment_list:
    df_sub = df.loc[df['REGIMEN']==p,:]
    desc = d.descriptive_table_treatments(df_sub, features, short_version = True)
    desc = desc.drop('Percent Missing', axis=1)
    desc.loc['Treatment'] = p
    # desc = desc.add_suffix('_'+p)
    desc = desc.rename({'Output':p}, axis=1) 
    desc_list.append(desc)
desc_pre = pd.concat(desc_list, axis=1).reindex(col_order, axis = 0)
# desc_pre.rename({'index':'Feature'}, inplace = True, axis = 1)


df = data_post
desc_list = []
for p in treatment_list:
    df_sub = df.loc[df['REGIMEN']==p,:]
    desc = d.descriptive_table_treatments(df_sub, features, short_version = True)
    desc = desc.drop('Percent Missing', axis=1)
    desc.loc['Treatment'] = p
    # desc = desc.add_suffix('_'+p)
    desc = desc.rename({'Output':p}, axis=1) 
    desc_list.append(desc)
desc_post = pd.concat(desc_list, axis=1).reindex(col_order, axis = 0)
# desc_post.rename({'index':'Feature'}, inplace = True, axis = 1)


res = pd.DataFrame(columns = columns, 
             index = index)

res['Pre-Match'] = desc_pre
res['Post-Match'] = desc_post

res = res.rename(index = col_mapping, columns =  {'ACEI_ARBS':'ACEI/ARBs','NO_ACEI_ARBS':'No ACEI/ARBs'})
res.index.name = 'Feature'

res.to_latex(buf = save_path+'latex_descriptive_prematch.txt', 
             column_format = 'l'+'c'*res.shape[1],
             float_format="%.3f", bold_rows = False, multicolumn = True, multicolumn_format = 'c',
             index_names = False)

#%% Break down pre-treatment data by site

# deriv = data_pre.query('COUNTRY == "Spain"')
# val = data_pre.query('COUNTRY != "Spain"')

df = data_pre
df.SOURCE_COUNTRY = df.SOURCE_COUNTRY.replace({'Hope-Italy':'Hope-Other',
                                               'Hope-Ecuador':'Hope-Other',
                                               'Hope-Germany':'Hope-Other',
                                               'USA':'MGB-USA'})
## Derivation Cohort
desc_list = []
for p in ['Hope-Spain','HM-Spain']:
    df_sub = df.loc[df['SOURCE_COUNTRY']==p,:]
    desc = d.descriptive_table_treatments(df_sub, features, short_version = True)
    desc = desc.drop('Percent Missing', axis=1)
    desc.loc['Treatment'] = p
    # desc = desc.add_suffix('_'+p)
    desc = desc.rename({'Output':p}, axis=1) 
    desc_list.append(desc)
desc_country = pd.concat(desc_list, axis=1).reindex(col_order, axis = 0)
# desc_pre.rename({'index':'Feature'}, inplace = True, axis = 1)

desc_country = desc_country.rename(index = col_mapping)

desc_country.to_latex(buf = save_path+'latex_descriptive_bycountry_derivation.txt', 
             column_format = 'l'+'c'*desc_country.shape[1],
             float_format="%.3f", bold_rows = False, multicolumn = True, multicolumn_format = 'c',
             index_names = False)

## Repeat for validation cohort
desc_list = []
for p in ['Cremona-Italy','Hope-Other','MGB-USA']:
    df_sub = df.loc[df['SOURCE_COUNTRY']==p,:]
    desc = d.descriptive_table_treatments(df_sub, features, short_version = True)
    desc = desc.drop('Percent Missing', axis=1)
    desc.loc['Treatment'] = p
    # desc = desc.add_suffix('_'+p)
    desc = desc.rename({'Output':p}, axis=1) 
    desc_list.append(desc)
desc_country = pd.concat(desc_list, axis=1).reindex(col_order, axis = 0)
# desc_pre.rename({'index':'Feature'}, inplace = True, axis = 1)

desc_country = desc_country.rename(index = col_mapping)

desc_country.to_latex(buf = save_path+'latex_descriptive_bycountry_validation.txt', 
             column_format = 'l'+'c'*desc_country.shape[1],
             float_format="%.3f", bold_rows = False, multicolumn = True, multicolumn_format = 'c',
             index_names = False)

#%% Compare validation and derivation populations by ACE/ARBs

deriv = data_post
val = data_pre.query('COUNTRY != "Spain"')

# df['Split'] = df['COUNTRY'].apply(lambda x: 'Derivation' if x == "Spain" else 'Validation')

res = pd.DataFrame(columns = pd.MultiIndex.from_product([['Derivation','Validation'],treatment_list]), 
             index = index)

for split in ['Derivation','Validation']:
    df = deriv if split == 'Derivation' else val
    desc_list = []
    for p in treatment_list:
        df_sub = df.loc[(df['REGIMEN']==p),:]
        desc = d.descriptive_table_treatments(df_sub, features, short_version = True)
        desc = desc.drop('Percent Missing', axis=1)
        desc.loc['Treatment'] = p
        # desc = desc.add_suffix('_'+p)
        desc = desc.rename({'Output':p}, axis=1) 
        desc_list.append(desc)
    desc_table = pd.concat(desc_list, axis=1).reindex(col_order, axis = 0)
    # desc_post.rename({'index':'Feature'}, inplace = True, axis = 1)
    res[split] = desc_table


res = res.rename(index = col_mapping, columns =  {'ACEI_ARBS':'ACEI/ARBs','NO_ACEI_ARBS':'No ACEI/ARBs'})
res.index.name = 'Feature'


res.to_latex(buf = save_path+'latex_descriptive_derivation_validation.txt', 
             column_format = 'l'+'c'*res.shape[1],
             float_format="%.3f", bold_rows = False, multicolumn = True, multicolumn_format = 'c',
             index_names = False)


#%% Save % changes 
data_version = 'validation_all'
summ_all = pd.read_csv(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_'+'prescription_rate_by_feature.csv')

summ_all['Feature_Value'] = summ_all['Feature_Value'].replace({'[0, 40)':'L','[40, 55)':'M','[55, 70)':'H','[70, 110)':'VH',
                                                               '0.0':'L','1.0':'H',
                                                               '0':'L','1':'H'})
summ_all['Label'] = summ_all.apply(lambda row: 'ValChange_'+row['Feature']+'_'+str(row['Feature_Value']), axis=1)

def convert_to_latex(command_name, val):   
    command_strip = command_name.replace('_','')
    val_pct = str(np.round(val*100,1))+'\%'
    com = '\\newcommand{\\' + command_strip +'}{'+val_pct+'}'  
    return com

shortcuts = list()

for i, row in summ_all.iterrows():
    shortcuts.append(convert_to_latex(row['Label'],row['Change_Relative']))

#Save the shortcuts in a txt
results_shortcuts_path = save_path+'latex_clinical_validation_shortcuts_t'+str(threshold)+'.txt'
with open(results_shortcuts_path, 'w') as f:
    for item in shortcuts:
        f.write("%s\n" % item)
    
#%% Results shortcuts from summary metrics

metric_summ = pd.read_csv(save_path+'matched_metrics_summary.csv')

metric_summ = metric_summ[(metric_summ['weighted_status']==weighted_status) &(metric_summ['threshold']==threshold) ]
def convert_to_latex(command_name, val):   
    command_strip = command_name.replace('_','')
    val_pct = str(np.round(val*100,1))+'\%'
    com = '\\newcommand{\\' + command_strip +'}{'+val_pct+'}'  
    return com

shortcuts = list()

PE = np.round(metric_summ[metric_summ['data_version']=='test']['PE'].values[0]*(-1),3)
shortcuts.append(convert_to_latex('prescriptioneffectivenessAbstract', PE))

shortcuts.append(convert_to_latex('improvementThreshold', threshold))

validation_size = len(pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_all_treatments_validation_all.csv'))
testing_size = len(pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_matched_all_treatments_test.csv'))
training_size = len(pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'))


TrainProp = metric_summ[metric_summ['data_version']=='train']['presc_count'].values[0]/training_size
shortcuts.append(convert_to_latex('proportionTrainingPrescription', TrainProp))

TestProp = metric_summ[metric_summ['data_version']=='test']['presc_count'].values[0]/testing_size
shortcuts.append(convert_to_latex('proportionTestingPrescription', TestProp))

ValidationProp = metric_summ[metric_summ['data_version']=='validation_all']['presc_count'].values[0]/validation_size
shortcuts.append(convert_to_latex('proportionValidationPrescription', ValidationProp))

minimumPE = np.round(metric_summ[metric_summ['data_version']=='validation_all']['PE'].values[0]*(-1),3)
shortcuts.append(convert_to_latex('minimumPE', minimumPE))

minimumPRTest = np.round(metric_summ[metric_summ['data_version']=='test']['pr_low'].values[0]*(-1),3)
shortcuts.append(convert_to_latex('minimumPRTest', minimumPRTest))

maximumPRTest = np.round(metric_summ[metric_summ['data_version']=='test']['pr_high'].values[0]*(-1),3)
shortcuts.append(convert_to_latex('maximumPRTest', maximumPRTest))

minimumPRValidation = np.round(metric_summ[metric_summ['data_version']=='validation_all']['pr_low'].values[0]*(-1),3)
shortcuts.append(convert_to_latex('minimumPRValidation', minimumPRValidation))

maximumPRValidation = np.round(metric_summ[metric_summ['data_version']=='validation_all']['pr_high'].values[0]*(-1),3)
shortcuts.append(convert_to_latex('maximumPRValidation', maximumPRValidation))

matchrateTest = np.round(metric_summ[metric_summ['data_version']=='test']['match_rate'].values[0],3)
shortcuts.append(convert_to_latex('matchrateTest', matchrateTest))

matchrateValidation = np.round(metric_summ[metric_summ['data_version']=='validation_all']['match_rate'].values[0],3)
shortcuts.append(convert_to_latex('matchrateValidation', matchrateValidation))

prescriptionAUCTest = np.round(metric_summ[metric_summ['data_version']=='test']['average_auc'].values[0],3)
shortcuts.append(convert_to_latex('prescriptionAUCTest', prescriptionAUCTest))

prescriptionAUCValidation = np.round(metric_summ[metric_summ['data_version']=='validation_all']['average_auc'].values[0],3)
shortcuts.append(convert_to_latex('prescriptionAUCValidation', prescriptionAUCValidation))

shortcuts

#Save the shortcuts in a txt
results_shortcuts_path = save_path+'latex_prescription_metrics_summary_shortcuts_t'+str(threshold)+'.txt'
with open(results_shortcuts_path, 'w') as f:
    for item in shortcuts:
        f.write("%s\n" % item)



