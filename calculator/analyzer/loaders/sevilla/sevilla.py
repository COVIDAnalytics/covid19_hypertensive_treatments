#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:24:13 2020

@author: agni
"""

import pandas as pd
import datetime
import numpy as np
import pickle
import analyzer.dataset as ds


#path = '../../../Dropbox (Personal)/COVID_clinical/covid19_sevilla'

RENAMED_COLUMNS = {
    'ï»¿ID':'PATIENT ID',
    'Age (Age)':'Age',
    'Sex (Sex)':'Gender',
    'Did the patient die prior to discharge? (Did_the_patient_die_prior_to_discharge)':'Outcome',
    'Diabetes (Diabetes)':'Diabetes',
    'Hypertension (Hypertension)':'Essential hypertension',    
    'Coronary heart disease (Coronary_heart_disease)':'Coronary atherosclerosis and other heart disease',
    'Chronic kidney/renal disease (Chronic_kidney_renal_disease)':'Chronic renal disfunction',
    'Date/Time of Hospital Admission (Date_Time_of_Hospital_Admission)':'Date_Admission',
    'Death':'Outcome',
    'Oxygen Saturation':'ABG: Oxygen Saturation (SaO2)',
    'Max temperature (celsius) (Max_temperature_celsius)':'Body Temperature',
    'Pulse >=125 beats per min (Pulse_greq125_beats_per_min)':'Cardiac Frequency',
    'Chronic obstructive lung (Chronic_obstructive_lung)':'COPD',
    'Current smoker (Current_smoker)':'Smoking_history',
    'Current drinker (Current_drinker)':'alcohol_abuse',
    'Cancer (Any) (Cancer_Any)':'Cancer', 
    'Systolic blood pressure <90mm Hg (Systolic_blood_pressure_lt_90mm_Hg)':'Systolic Blood Pressure'}
  
# LABS_RENAMED_COLUMNS = {
#     'Glucosa(mg/dL)'
    
    
    
    
#     }  


#     {'HF':'Cardiac dysrhythmias',
#     'Liver_dis':'Liver disfunction',
#     'RR_adm':'Respiratory Frequency',
#     'GLU_PRIMERA/FIRST_URG/EMERG':'Glycemia',
#     'SBP_adm':'Systolic Blood Pressure',
#     'Tbil_adm':'Total Bilirubin',
#     'WBC_adm':'CBC: Leukocytes', #needs to be divided by 1000
#     'AST_adm':'Aspartate Aminotransferase (AST)',
#     'ALT_adm':'Alanine Aminotransferase (ALT)',
#     'CRP_adm':'C-Reactive Protein (CRP)',
#     'INR_adm':'Prothrombin Time (INR)',
#     'PLT_adm':'CBC: Platelets',
#     'Creat_adm':'Blood Creatinine',
#     'DD_adm':'D-Dimer',
#     'Urea_adm':'Urea',
#     'lympho_adm':'CBC: Lymphocytes',
#     'Hb_adm':'CBC: Hemoglobin',
#     'PCT_adm':'Procalcitonin (PCT)',
#     'FERR_adm':'Ferritinin',
#     'LDH_adm':'LDH',
#     'Creat_adm':'Blood Creatinine',
#     'fibr_adm':'fibrinogen',
#     'CPK_adm':'Creatinine Kinase',
#     'UricAcid_adm':'Uric acid',
#     'trop_adm':'troponin'}



DEMOGRAPHICS_COLUMNS = ['PATIENT ID','Age','Gender']

ADMISSION_COLUMNS = ['PATIENT ID','Date_Admission','Outcome']


COMORBIDITIES_COLUMNS = ['PATIENT ID','Essential hypertension', 'Diabetes', 
                 'Coronary atherosclerosis and other heart disease',
                 'Chronic renal disfunction',
                 'COPD','Smoking_history',
                 'alcohol_abuse',
                 'Cancer']

VITALS_COLUMNS = ['PATIENT ID','Body Temperature', 'Cardiac Frequency']


EXTRA_COLUMNS = ['PATIENT ID',
                 'Cardiac dysrhythmias',
                 'Liver disfunction']
                 
                 
                 
                 # 'Cholinesterase',              
                 # 'Blood Calcium',            
                 # 'Blood Amylase',            
                 # 'Activated Partial Thromboplastin Time (aPTT)',            
                 # 'CBC: Mean Corpuscular Volume (MCV)',
                 # 'Blood Sodium',
                 # 'Potassium Blood Level',
                 # 'Glycemia']


LAB_METRICS = ['PATIENT ID',
               'ABG: Oxygen Saturation (SaO2)',
               'CBC: Leukocytes',
               'Aspartate Aminotransferase (AST)',
               'Alanine Aminotransferase (ALT)',
               'Prothrombin Time (INR)',
               'CBC: Platelets',
               'C-Reactive Protein (CRP)',
               'Blood Creatinine',
               'CBC: Hemoglobin',
               'CBC: Lymphocytes',
               'Procalcitonin (PCT)',
               'Ferritinin',
               'LDH',
               'Blood Creatinine',
               'Total Bilirubin',
               'D-Dimer',
               'Creatinine Kinase',
               'troponin',
               'Urea']

def create_dataset_v2(data_dict, discharge_data = True,
                        comorbidities_data = True,
                        vitals_data = True,
                        lab_tests=True,
                        demographics_data = False,
                        swabs_data = False,
                        extra_data = True,
                        prediction='Outcome'):

    if demographics_data:
        data = data_dict['demographics']

    if discharge_data:
        data = data.join(data_dict['discharge'])

    if comorbidities_data:
        data = data.join(data_dict['comorbidities'])

    if vitals_data:
        data = data.join(data_dict['vitals'])

    if lab_tests:
        data = data.join(data_dict['lab'])

    if swabs_data:
        data = data.join(data_dict['swab'])
   
    if extra_data:
        data = data.join(data_dict['extras'])

    X = data.loc[:, data.columns.difference(['Outcome', 'ICU', 'Swab'])]
    y = data.loc[:, prediction]

    #X = X.astype(np.float32)
    y = y.astype(int)

    return X, y


def load_sevilla(path, discharge_data = True, comorbidities_data = True, vitals_data = True, lab_tests=False, demographics_data = True, extra_data = True):

    # Load admission info
    df = pd.read_csv('%s/COVID19_SASpatients_v1.csv' % path, sep=',' , encoding= 'unicode_escape')
    
    # Filter to only patients for which we know the endpoint  
    df = df[df['Discharged? (Discharged)'].notnull()]
    df = df[~(df['Discharged? (Discharged)']=='No')]
    
    #Drop rows for which the outcome of interest is not known
    df = df[df['Did the patient die prior to discharge? (Did_the_patient_die_prior_to_discharge)'].notna()]
    
    #Drop columns that only contain NA values
    df = df.dropna(axis=1, how='all')
    
    #Rename the relevant columns
    df = df.rename(columns=RENAMED_COLUMNS)
    
    #Keep relevant columns for that are in the calculator already
    dataset_admissions = df[ADMISSION_COLUMNS]
   
    # Filter to relevant patients
    patients = df['PATIENT ID']
    
    #Demographic information
    dataset_demographics = df[DEMOGRAPHICS_COLUMNS]
    
    #Vital values
    dataset_vitals = df[VITALS_COLUMNS]
    
    #Dictionary for cardiac frequency values
    dataset_vitals[['Cardiac Frequency']] = dataset_vitals[['Cardiac Frequency']].replace(['Yes','No'], [26,19])   
    
    #Dictionary for binary values
    binary_dict = {'Yes': 1,'No': 0}
    
    #Comorbidities
    dataset_comorbidities = df[COMORBIDITIES_COLUMNS]
    dataset_comorbidities[COMORBIDITIES_COLUMNS] = dataset_comorbidities[COMORBIDITIES_COLUMNS].replace(['Yes','No','Past'], [1,0,0])   
    
    #Add SaO2 in Vitals
    if not lab_tests:
        dataset_vitals['SaO2'] = df['ABG: Oxygen Saturation (SaO2)']
 
    #Change gender decoding    
    dataset_demographics[['Gender']] = dataset_demographics[['Gender']].replace(['Female','Male'], [1,0])   

    
    dataset_admissions[['Outcome']] = dataset_admissions[['Outcome']].replace(['Yes','No'], [1,0])   
     
    #Read in lab metrics
    labs = pd.read_csv('%s/Analiticas_SASpatientesv1.csv' % path, sep=',' , encoding= 'unicode_escape')

    #Drop columns that only contain NA values
    labs = labs.dropna(axis=1, how='all')

    #Filter to only patients that are in the inclusion criteria
    #df3 = pd.merge(dataset_admissions, labs, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['ï»¿ID', 'FECHA'])
    #dataset_admissions['PATIENT ID'][dataset_admissions['PATIENT ID'].isin(labs['ï»¿ID'])]
           
    # Lab values
    #dataset_labs = df[LAB_METRICS]
    dataset_extra = dataset_admissions['PATIENT ID'].to_frame()

    for i in EXTRA_COLUMNS:
        print(i)
        if not i =='PATIENT ID':
            dataset_extra[i] = np.nan

    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals','demographics', 'extras']) #'lab', 
    dataset_array = np.asarray([discharge_data, comorbidities_data, vitals_data, demographics_data, extra_data])#lab_tests

    # Set index
    dataset_admissions.set_index('PATIENT ID', inplace=True)
    dataset_comorbidities.set_index('PATIENT ID', inplace=True)
    dataset_vitals.set_index('PATIENT ID', inplace=True)
    #dataset_labs.set_index('PATIENT ID', inplace=True)
    dataset_demographics.set_index('PATIENT ID', inplace=True)
    dataset_extra.set_index('PATIENT ID', inplace=True)

    datasets = []


    # Create final dataset
    if discharge_data:
        datasets.append(dataset_admissions)
    
    if comorbidities_data:
        datasets.append(dataset_comorbidities)
    
    if vitals_data:
        datasets.append(dataset_vitals)
    
    #if lab_tests:
     #   datasets.append(dataset_labs)

    if demographics_data:
        datasets.append(dataset_demographics)

    if extra_data:
        datasets.append(dataset_extra)

    datasets = np.asarray(datasets)

    data = dict(zip(name_datasets[dataset_array], datasets))

    return data

##############

lab_tests = False
swabs_data = False
discharge_data = True
comorbidities_data = True
vitals_data = True
demographics_data = True
icu_data = False
extra_data = True

prediction = 'Outcome'

path = '../../../Dropbox (Personal)/COVID_clinical/covid19_sevilla'

data = load_sevilla(path, discharge_data = True, comorbidities_data = True, vitals_data = True, lab_tests=False, demographics_data = True, extra_data = True)

X_sevilla, y_sevilla = create_dataset_v2(data,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      swabs_data,
                                      extra_data,
                                      prediction = prediction)

sevilla = X_sevilla
sevilla['Outcome'] = y_sevilla

sevilla.to_csv(path+'/clean_data_no_labs.csv')




