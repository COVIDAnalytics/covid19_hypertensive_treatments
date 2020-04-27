#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:18:10 2020

@author: agni
"""

import pandas as pd
#  import datetime
import numpy as np
import pickle

import analyzer.loaders.hmfundacion.utils as u


path = '../../../Dropbox (Personal)/COVID_clinical/covid19_hmfoundation'

### Admissions File
# Load admission info
admission = pd.read_csv('%s/admissions.csv' % path, sep=';' , encoding= 'unicode_escape')
#Maybe we can include 'Traslado a un Centro Sociosanitario'
admission = admission.rename(columns={'EDAD/AGE':'Age','SEXO/SEX':'Sex',
                                      'DIAG ING/INPAT':'DIAG_TYPE',
                                      'MOTIVO_ALTA/DESTINY_DISCHARGE_ING':'death',
                                      'F_INGRESO/ADMISSION_D_ING/INPAT':'Date_Admission',
                                      'F_INGRESO/ADMISSION_DATE_URG/EMERG':'Date_Emergency',
                                      'TEMP_PRIMERA/FIRST_URG/EMERG':'Temperature Celsius',
                                      'FC/HR_PRIMERA/FIRST_URG/EMERG':'Cardiac Frequency',
                                      'GLU_PRIMERA/FIRST_URG/EMERG':'Glycemia',
                                      'SAT_02_PRIMERA/FIRST_URG/EMERG':'ABG: Oxygen Saturation (SaO2)',
                                      'TA_MAX_PRIMERA/FIRST/EMERG_URG':'Systolic Blood Pressure'})

#Limit to only patients for whom we know the outcome
types = ['Fallecimiento', 'Domicilio']
admission = admission.loc[admission['death'].isin(types)]
#Dictionary for the outcome
death_dict = {'Fallecimiento': 1,'Domicilio': 0} 
admission.death = [death_dict[item] for item in admission.death] 

#Dictionary for the gender 
gender = {'MALE': 1,'FEMALE': 2} 
admission.Sex = [gender[item] for item in admission.Sex] 

#Dictionary for the vitas at admission
admission['Temperature'] = admission['Temperature Celsius'].replace('0',np.nan).str.replace(',','.').astype(float)
admission['Cardiac Frequency']=admission['Cardiac Frequency'].replace(0,np.nan)
admission['SaO2']=admission['ABG: Oxygen Saturation (SaO2)'].replace(0,np.nan)
admission['Glycemia']=admission['Glycemia'].replace(0,np.nan)
admission['Systolic Blood Pressure']=admission['Systolic Blood Pressure'].replace(0,np.nan)


admission['Date_Emergency']= pd.to_datetime(admission['Date_Emergency']).dt.date 
admission['Date_Admission']= pd.to_datetime(admission['Date_Admission']).dt.date 


admission_cols = ['PATIENT ID','death','DIAG_TYPE','Date_Admission','Date_Emergency','Age','Sex','Temperature Celsius','Cardiac Frequency','ABG: Oxygen Saturation (SaO2)','Systolic Blood Pressure']
df1 = admission[admission_cols]

# Lab Test Results #

labs = pd.read_csv('%s/labs.csv' % path, encoding= 'unicode_escape')
#Renaming of variables

labs['DETERMINACION.ITEM_LAB'] = labs['DETERMINACION.ITEM_LAB'].replace({'BT -- BILIRRUBINA TOTAL                                                               ':'Total Bilirubin',
                            'GOT -- GOT (AST)':'Aspartate Aminotransferase (AST)',
                            'LIN% -- Linfocitos %':'CBC: Leukocytes',
                            'HGB -- Hemoglobina':'CBC: Hemoglobin',
                            'VCM -- Volumen Corpuscular Medio':'CBC: Mean Corpuscular Volume (MCV)',
                            'GPT -- GPT (ALT)':'Alanine Aminotransferase (ALT)',
                            'NA -- SODIO':'Blood Sodium',
                            'LIN -- Linfocitos':'CBC: Leukocytes2',
                            'INR -- INR':'Prothrombin Time (INR)',
                            'K -- POTASIO':'Potassium Blood Level',                            
                            'COL -- COLESTEROL TOTAL':'Cholinesterase',
                            # '':'CBC: Red cell Distribution Width (RDW) '
                            'VPM -- Volumen plaquetar medio':'CBC: Platelets',
                            'PCR -- PROTEINA C REACTIVA':'C-Reactive Protein (CRP)',
                            'U -- UREA':'Urea',
                            'CREA -- CREATININA':'Blood Creatinine',
                            'CA -- CALCIO                                                                          ':'Blood Calcium',
                            'AMI -- AMILASA':'Blood Amylase',
                            'APTT -- TIEMPO DE CEFALINA (APTT':'Activated Partial Thromboplastin Time (aPTT)',
                            'HCO3 -- HCO3-':'ABG: standard bicarbonate (sHCO3)',
                            'PH -- pH':'ABG: pH',
                            'PO2 -- pO2':'ABG: PaO2',
                            'PCO2 -- pCO2':'ABG: PaCO2',
                            # '':'ABG: MetHb',
                            'LAC -- LACTATO':'ABG: Lactic Acid', 
                            # '':'ABG: COHb',
                            'BE(b) -- BE(b)':'ABG: Base Excess',
                            'LEUC -- Leucocitos':'CBC: Leukocytes3',
                            'LEUORS -- Leucocitos':'CBC: Leukocytes4',                           
                            'DD -- DIMERO D':'D-Dimer',
                            'GLU -- GLUCOSA':'Glycemia'})

common_labs = ['Total Bilirubin','Aspartate Aminotransferase (AST)',
              'CBC: Leukocytes','CBC: Hemoglobin', 'CBC: Mean Corpuscular Volume (MCV)',
              'Alanine Aminotransferase (ALT)','Blood Sodium',
              'Prothrombin Time (INR)','Potassium Blood Level','Cholinesterase', 
              'CBC: Platelets','C-Reactive Protein (CRP)','Urea','Blood Creatinine',
              'Blood Calcium','Blood Amylase','Activated Partial Thromboplastin Time (aPTT)',
              'ABG: standard bicarbonate (sHCO3)','ABG: pH','ABG: PaO2','ABG: PaCO2','ABG: Lactic Acid',
              'ABG: Base Excess','D-Dimer','Glycemia']

#Limit to only rows for the ones known in Italy
labs = labs.loc[labs['DETERMINACION.ITEM_LAB'].isin(common_labs)]
#Reduce the number of columns
cols_to_keep = ['PATIENT.ID','FECHA_PETICION.LAB_DATE','DETERMINACION.ITEM_LAB','RESULTADO.VAL_RESULT']
labs = labs[cols_to_keep]

#Convert to Date the date column
labs['FECHA_PETICION.LAB_DATE']= pd.to_datetime(labs['FECHA_PETICION.LAB_DATE']).dt.date 
labs['RESULTADO.VAL_RESULT'] = labs['RESULTADO.VAL_RESULT'].str.extract('(\d+(?:\.\d+)?)', expand=False).astype(float)

#Convert to wide format
df2 = labs.pivot_table(index=['PATIENT.ID','FECHA_PETICION.LAB_DATE'], columns='DETERMINACION.ITEM_LAB', values='RESULTADO.VAL_RESULT')
df2=pd.DataFrame(df2.to_records())

df2['PATIENT.ID'] = df2['PATIENT.ID'].astype(int)

#Createe Blood Urea Nitrogen from Urea
df2['Blood Urea Nitrogen (BUN)'] = df2['Urea']/2.14
#Drop Urea
df2 = df2.drop(columns=['Urea'])


#Merge the two dataframes admissions and labs together.
df3 = pd.merge(df1, df2, how='inner', left_on=['PATIENT ID','Date_Emergency'], right_on=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])
#df4 = pd.merge(df1, df2, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])
df3 = df3.drop(columns=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])



#Check to see how many missing values we have at each column
#missing_values_table(df4)    
missing_values_table(df3)    

# Identify the comorbidities of the patient
comorbidities_emerg = pd.read_csv('%s/comorbidities_emerg.csv' % path, sep=';',encoding= 'unicode_escape')
comorbidities_inpatient = pd.read_csv('%s/comorbidities_inpatient.csv' % path, sep=';', encoding= 'unicode_escape')

#Convert them to a long format
comorb_long1=pd.melt(comorbidities_emerg,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')
comorb_long2=pd.melt(comorbidities_inpatient,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')

#Concatanate the two dataframes
comorb_long = pd.concat([comorb_long1,comorb_long2])

#We will treat all types of diagnoses the same
comorb_long = comorb_long.drop(['DiagOrdering'], axis=1)
#Remove NA values
comorb_long=comorb_long.dropna()
#Convert PATIENT ID to integer
comorb_long['PATIENT ID'] = comorb_long['PATIENT ID'].astype(int)
#Remove the dot from the values column
comorb_long['values']= comorb_long['values'].str.replace(".","")

#Load the diagnoses dict
icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd10.csv') 

#The codes that are not mapped are mostly procedure codes or codes that are not of interest
icd_descr = pd.merge(comorb_long, icd_dict, how='inner', left_on=['values'], right_on=['DIAGNOSIS_CODE'])

#Now we need to restrict to the categories for which we have italian data
# cardiac arrhythmia -> 106
# acute renal failure -> 145
# chronic kidney disease -> 146
# CAD, heart disease -> 90
# diabetes -> 49, 50
# hypertension -> 87

#Create a list with the categories that we want
comorb_list = [49,50,87,90,95,145,146]
comorb_descr = icd_descr.loc[icd_descr['HCUP_ORDER'].isin(comorb_list)]

#Limit only to the HCUP Description and drop the duplicates
comorb_descr = comorb_descr[['PATIENT ID','GROUP_HCUP']].drop_duplicates()

#Convert from long to wide format
comorb_descr = pd.get_dummies(comorb_descr, prefix=['GROUP_HCUP'])

#Now we will remove the GROUP_HCUP_ from the name of each column
comorb_descr = comorb_descr.rename(columns = lambda x: x.replace('GROUP_HCUP_', ''))

#Let's combine the diabetes columns to one
comorb_descr['Diabetes'] = comorb_descr['Diabetes mellitus with complications'] + comorb_descr['Diabetes mellitus without complication'] 

#Drop the other two columns
comorb_descr = comorb_descr.drop(columns=['Diabetes mellitus with complications', 'Diabetes mellitus without complication'])

comorb_descr = pd.DataFrame(comorb_descr.groupby(['PATIENT ID'], as_index=False).sum())
comorb_descr = comorb_descr[comorb_descr['PATIENT ID'].isin(df3['PATIENT ID'])]

# Combine the comorbidities with the main filees
df4 = pd.merge(df3, comorb_descr, how='left', left_on=['PATIENT ID'], right_on=['PATIENT ID'])

#These are the columns for which we do not have values
X.columns[~X.columns.isin(df4.columns)]

for i in X.columns[~X.columns.isin(df4.columns)]:
    df4[i] = np.nan

#Save the file to the folder
df4.to_csv('%s/fundacionhm_italy_adjusted_clean.csv' % path, index=False)











#Limit to patients that are only in df3
df4 = df4[df4['PATIENT ID'].isin(df3['PATIENT ID'])]



labs.pivot_table(index=['Area', 'Year'], columns='Variable Name', values='Value')


labs[labs['DETERMINACION.ITEM_LAB']=='Blood Calcium']

# For (BUN) = Urea/2.14 - blood urea nitrogen × 2.14 = blood urea
# 'Blood Urea Nitrogen (BUN)'



#List of features that I have not mapped
# CBC: Red cell Distribution Width (RDW), ABG: MetHb, ABG: COHb, Cholinesterase 
# Blood Calcium, Blood Amylase: too many missing values
# 'Activated Partial Thromboplastin Time (aPTT): not present in the sample
# Influenza: we will remove, no point having it
# NOSOLOGICO: not clear why we keep it
# I have joined the comorbidities of both the emergency and admissions
# cardiac arrhythmia -> 95
# acute renal failure -> 145
# chronic kidney disease -> 146
# CAD, heart disease -> 90
# diabetes -> 49, 50
# hypertension -> 87,88
#Columns are not in the right order but they are all there including those with missing values

