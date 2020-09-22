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

#%% Version information

version = 'matched_single_treatments_der_val_addl_outcomes/'
data_path = '../../covid19_treatments_data/'+version
results_path = '../../covid19_treatments_results/'+version
        
threshold = 0.01
match_status = 'matched' 
weighted_status = 'no_weights'

SEEDS = range(1, 2)
algorithm_list = ['rf','cart','oct','xgboost','qda','gb']
data_list = ['train','test','validation']

outcome = 'COMORB_DEATH'
treatment = 'ACEI_ARBS'
treatment_list = [treatment, 'NO_'+treatment]

training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'

version_folder = str(treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

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

col_dict = dict({'train':'Training Data', 'test':'Testing Data', 'validation':'Validation Data',
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

col_dict = dict({'train':'Training Data', 'test':'Testing Data', 'validation':'Validation Data',
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

summ.data_version = summ.data_version.str.capitalize()
summ.rename({'data_version':'Data Version', 'match_rate':'Match Rate',
            'presc_count':'Presc. Count', 'average_auc':'Avg. AUC', 
            'pr_low':'PR (Low)', 'pr_high':'PR (High)'}, axis=1, inplace = True)

summ.set_index('Data Version', inplace = True)
summ.to_latex(buf = save_path+'latex_prescriptive_table.txt', 
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

agr.to_latex(buf = save_path+'latex_agreement_table.txt', 
             column_format = 'l'+'c'*agr.shape[1],
             float_format="%.3f", bold_rows = True, multicolumn = True, multicolumn_format = 'c',
             index_names = False)
# preed_reseults
                  
#%% Descriptive analysis: pre- vs. post-matching

data_pre = pd.read_csv('../../covid19_treatments_data/hope_hm_cremona_data_clean_imputed_addl_outcomes.csv')
data_pre['REGIMEN'] = data_pre[treatment].apply(lambda x: treatment if x == 1 else 'NO_'+treatment)
data_pre = pd.get_dummies(data_pre, columns = ['GENDER','RACE'], drop_first = True)
data_pre['Version'] = 'Pre-Match'

# data_post0 = pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_matched.csv')
data_post_train = pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_matched_all_treatments_train.csv')
data_post_test = pd.read_csv('../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'+treatment+'_hope_hm_cremona_matched_all_treatments_test.csv')
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

# df = pd.concat([data_pre[col_order], data_post[col_order]], axis=0)
df = data_pre

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
             column_format = 'l'+'c'*desc_all.shape[1],
             float_format="%.3f", bold_rows = False, multicolumn = True, multicolumn_format = 'c',
             index_names = False)

#%% Break down pre-treatment data by site
df = data_pre
df.SOURCE_COUNTRY = df.SOURCE_COUNTRY.replace({'Hope-Italy':'Hope-Other',
                                               'Hope-Ecuador':'Hope-Other',
                                               'Hope-Germany':'Hope-Other'})

desc_list = []
for p in ['Hope-Spain','HM-Spain','Cremona-Italy','Hope-Other']:
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

desc_country.to_latex(buf = save_path+'latex_descriptive_bycountry.txt', 
             column_format = 'l'+'c'*desc_all.shape[1],
             float_format="%.3f", bold_rows = False, multicolumn = True, multicolumn_format = 'c',
             index_names = False)


