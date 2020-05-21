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
from datetime import datetime


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
  

LABS_RENAMED_COLUMNS = {
    'Glucosa':'Glycemia',
    'Alanina transaminasa':'Alanine Aminotransferase (ALT)',
    'Aspartato transaminasa':'Aspartate Aminotransferase (AST)',
    'Potasio':'Potassium Blood Level',
    'Plaquetas (recuento)':'CBC: Platelets',
    'Urea':'Urea',
    'Creatinina':'Blood Creatinine',
    'Creatina quinasa':'Creatinine Kinase',
    'Sodio':'Blood Sodium',
    'Calcio':'Blood Calcium',
    'Hemoglobina':'CBC: Hemoglobin',
    'ProteÃ\xadna C reactiva':'C-Reactive Protein (CRP)',
    'Volumen corpuscular medio':'CBC: Mean Corpuscular Volume (MCV)',
    'Tiempo de protrombina normalizado (INR)':'Prothrombin Time (INR)',    
    'Leucocitos (recuento)':'CBC: Leukocytes',
    'Linfocitos (recuento)':'CBC: Lymphocytes',
    'Colinesterasa':'Cholinesterase',
    'DÃ\xadmero-D':'D-Dimer'}  


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

LAB_METRICS = ['PATIENT ID',
               'Glycemia',
               'Potassium Blood Level',
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
               'Total Bilirubin',
               'Blood Sodium',
               'Blood Calcium',
               'D-Dimer',
               'Creatinine Kinase',
               'CBC: Mean Corpuscular Volume (MCV)',
               'Cholinesterase',
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

def fahrenheit_covert(temp_celsius):
    temp_fahrenheit = ((temp_celsius * 9)/5)+ 32
    return temp_fahrenheit


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
    
    
    #Reformatting the vital values at the emergency department
    dataset_vitals['Body Temperature'] = dataset_vitals['Body Temperature'].replace('0',np.nan).astype(float)
    #Convert to Fahrenheit
    dataset_vitals['Body Temperature'] = fahrenheit_covert(dataset_vitals['Body Temperature'])

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
    labs = pd.read_csv('%s/Sara Gonzalez Garcia - Analiticas v3.csv' % path, sep=';' , encoding= 'unicode_escape')

    labs['fecha'] = pd.to_datetime(labs['fecha'])
    dataset_admissions['Date_Admission'] = pd.to_datetime(dataset_admissions['Date_Admission'])

       
    labs['nombre'] = labs['nombre'].replace(LABS_RENAMED_COLUMNS)

    #Limit to only rows for the ones known in Italy
    labs = labs.loc[labs['nombre'].isin(LAB_METRICS)]
    labs['valor'] = labs['valor'].str.extract('(\d+(?:\.\d+)?)', expand=False).astype(float)

    #Convert to wide format
    df2 = labs.pivot_table(index=['id_paciente','fecha'], columns='nombre', values='valor')
    df2=pd.DataFrame(df2.to_records())
    
    #Createe Blood Urea Nitrogen from Urea
    df2['Blood Urea Nitrogen (BUN)'] = df2['Urea']/2.14
    
    #Lymphocytes to higher scale
    df2['CBC: Lymphocytes'] = df2['CBC: Lymphocytes']*1000
     
    #Drop Urea
    df2 = df2.drop(columns=['Urea'])

    dfa = pd.merge(dataset_admissions, df2, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['id_paciente', 'fecha'])   
    
    dataset_admissions['Date_Admission2'] = dataset_admissions['Date_Admission']+ pd.DateOffset(1)
    dfb = pd.merge(dataset_admissions, df2, how='inner', left_on=['PATIENT ID','Date_Admission2'], right_on=['id_paciente', 'fecha'])   

    dataset_admissions['Date_Admission3'] = dataset_admissions['Date_Admission']+ pd.DateOffset(-1)
    dfc = pd.merge(dataset_admissions, df2, how='inner', left_on=['PATIENT ID','Date_Admission3'], right_on=['id_paciente', 'fecha'])   

    dataset_admissions['Date_Admission4'] = dataset_admissions['Date_Admission']+ pd.DateOffset(2)
    dfd = pd.merge(dataset_admissions, df2, how='inner', left_on=['PATIENT ID','Date_Admission4'], right_on=['id_paciente', 'fecha'])   

    s1 = pd.Series(dfa.id_paciente.unique())    

          




    
    #First we will do the entire extraction based on the day of admission and then we will try other dates
    df2 = pd.merge(dataset_admissions, labs, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['id_paciente', 'fecha'])   

 

    
    
    
    
    
    #Filter to only patients that are in the inclusion criteria
    #df3 = pd.merge(dataset_admissions, labs, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['ï»¿ID', 'FECHA'])
    #dataset_admissions['PATIENT ID'][dataset_admissions['PATIENT ID'].isin(labs['ï»¿ID'])]
    
    pats = dataset_admissions[dataset_admissions['PATIENT ID'].isin(labs['id_paciente'])]
    df3 = pd.merge(dataset_admissions, labs, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['id_paciente', 'fecha'])   
    len(df3.id_paciente.unique())
    
    dataset_admissions['Date_Admission2'] =     dataset_admissions['Date_Admission']+ pd.DateOffset(1)
    df4 = pd.merge(dataset_admissions, labs, how='inner', left_on=['PATIENT ID','Date_Admission2'], right_on=['id_paciente', 'fecha'])  
    len(df4.id_paciente.unique())
    
    dataset_admissions['Date_Admission3'] =     dataset_admissions['Date_Admission']+ pd.DateOffset(-1)
    df5 = pd.merge(dataset_admissions, labs, how='inner', left_on=['PATIENT ID','Date_Admission3'], right_on=['id_paciente', 'fecha'])  
    len(df5.id_paciente.unique())
    
    dataset_admissions['Date_Admission4'] =     dataset_admissions['Date_Admission']+ pd.DateOffset(+2)
    df6 = pd.merge(dataset_admissions, labs, how='inner', left_on=['PATIENT ID','Date_Admission4'], right_on=['id_paciente', 'fecha'])  
    len(df6.id_paciente.unique())

    
    s1 = pd.Series(df3.id_paciente.unique())    
    s2 = pd.Series(df4.id_paciente.unique())
    s3 = pd.Series(df5.id_paciente.unique())
    s4 = pd.Series(df5.id_paciente.unique())    
    s = s1.append(s2).append(s3).append(s4)
    
    len(s.unique())
    
    # 111 patients out of which 73 have lab values
    
    
    [datetime.strptime(x, '%d/%m/%Y') for x in labs['fecha']]
    
    labs['fecha']+ pd.DateOffset(1)
    


    x = 'Linfocitos (recuento)'
    len(df2['id_paciente'][df2['nombre']==x].unique())
    pd.to_numeric(df2['valor'][df2['nombre']==x]).describe()


    
    labs[labs["id_paciente"].isin(['31987'])]
    
    
    df4 = df[df['PATIENT ID'].isin(pats)]
           
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




