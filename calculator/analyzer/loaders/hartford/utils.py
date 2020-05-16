import numpy as np
import pandas as pd
import datetime

# explicitly require this experimental feature
# from sklearn.experimental import enable_iterative_imputer  # noqa
# from sklearn.impute import IterativeImputer, KNNImputer

# from analyzer.utils import remove_missing

RENAMED_ADMISSION_COLUMNS = {
'PAT_MRN_ID':'PATIENT ID', 
    'EDAD/AGE':'Age',
    'SEXO/SEX':'Gender',
'DIAG_TYPE'
    'MOTIVO_ALTA/DESTINY_DISCHARGE_ING':'Outcome',
    'HOSP_ADMSN_TIME':'Date_Admission',
    'F_INGRESO/ADMISSION_DATE_URG/EMERG':'Date_Emergency'}

VITAL_COLUMNS ={
    'PATIENT ID',
    'Body Temperature',
    'Cardiac Frequency'   
    }

DEMOGRAPHICS_COLUMNS={'PATIENT ID','Gender','Age'}

ADMISSION_COLUMNS = ['PATIENT ID','Outcome','Date_Admission','Date_Emergency']

TREATMENT_COLS = ['VENT_ORDER', 'O2_DEVICE', 'INV_FLOW', 'INV_O2_CONCENTRATION', 
    'ECOMO', 'MED_PER_DAY', 'MAR_PER_DAY']

RENAMED_VITALS_MEASUREMENTS  = [
    'MIN_SPO2':'SaO2', # approximate matchh
    'MAX_TEMP':, 
    'MAX_RR':'Respiratory Frequency', 
    'MAX_SYSTOLIC':'Systolic Blood Pressure']

RENAMED_LAB_MEASUREMENTS = [
    'PAO2':'ABG: PaO2', 
    'PH':'ABG: pH', 
    'AST':'Aspartate Aminotransferase (AST)', 
    'CALCIUM':'Blood Calcium', 
    'HGB':'CBC: Hemoglobin', 
    'MCV':'CBC: Mean Corpuscular Volume (MCV)', 
    'METHGB':'ABG: MetHb',
    'CRP':'C-Reactive Protein (CRP)',
    'LACTIC_ACID':'ABG: Lactic Acid', 
    'PACO2':'ABG: PaCO2', 
    'ALT':'Alanine Aminotransferase (ALT)', 
    'AMYLASE': 'Blood Amylase',
    'CREATININE': 'Blood Creatinine', 
    'SODIUM':'Blood Sodium',
    'POTASSIUM':'Potassium Blood Level',
    'CARBOXYHGB':'ABG: COHb',  
    'D_DIMER': 'D-Dimer', 
    'HIGH_SENSITIVITY_D_DIMER',
    'PLATELET':'CBC: Platelets']

HCUP_LIST = [49,50,87,90,95,146]

HHC_EXTRA = ['LYMPHS_ABS', 'LDH', 'PTT', 'INTERLEUKIN_6', 'HGBA1C', 'BASE_DEFICIENCY',  # deficiency = 1 - excess?
    'PAT_ENC_CSN_ID', 'HOSP_DISCH_TIME', 'SYMPTOM_ACTIVITY', 'GENERAL_MOBILITY']

CREMONA_EXTRA = [ 
    'DIAG_TYPE',  # options: COVID_RESEARCH_DX, ENC_DX, RSN_FOR_VISIT
    'Total Bilirubin', # in Order
    'Blood Urea Nitrogen (BUN)' # in Orders]
    'CBC: Leukocytes', # in Orders
    'CBC: Red cell Distribution Width (RDW)', # in Orders
    'Cholinesterase',
    'Glycemia',
    'Prothrombin Time (INR)',
    'ABG: standard bicarbonate (sHCO3)'
    'ABG: Base Excess',
    'Activated Partial Thromboplastin Time (aPTT)'
    ]


# def missing_values_table(df):
#         mis_val = df.isnull().sum()
#         mis_val_percent = 100 * df.isnull().sum() / len(df)
#         mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
#         mis_val_table_ren_columns = mis_val_table.rename(
#         columns = {0 : 'Missing Values', 1 : '% of Total Values'})
#         mis_val_table_ren_columns = mis_val_table_ren_columns[
#             mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
#         '% of Total Values', ascending=False).round(1)
#         print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
#             "There are " + str(mis_val_table_ren_columns.shape[0]) +
#               " columns that have missing values.")
#         return mis_val_table_ren_columns



# def create_dataset_admissions(admission):

#     #Rename the columns for the admission
#     admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)

#     #Limit to only patients for whom we know the outcome
#     types = ['Fallecimiento', 'Domicilio']
#     admission = admission.loc[admission['Outcome'].isin(types)]
#     #Dictionary for the outcome
#     death_dict = {'Fallecimiento': 1,'Domicilio': 0}
#     admission.Outcome = [death_dict[item] for item in admission.Outcome]

#     #Convert to Dates the appropriate information
#     admission['Date_Emergency']= pd.to_datetime(admission['Date_Emergency']).dt.date
#     admission['Date_Admission']= pd.to_datetime(admission['Date_Admission']).dt.date

#     df1 = admission[ADMISSION_COLUMNS]

#     return df1


# def create_dataset_demographics(admission):
#     #Rename the columns for the admission
#     admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)

#     #Dictionary for the gender
#     gender = {'MALE': 0,'FEMALE': 1}
#     admission.Gender = [gender[item] for item in admission.Gender]

#     df2 = admission[DEMOGRAPHICS_COLUMNS]
#     return df2


# def create_vitals_dataset(admission):
#     #Rename the columns for the admission
#     admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)
#     #Reformatting the vital values at the emergency department
#     admission['Body Temperature'] = admission['Body Temperature'].replace('0',np.nan).str.replace(',','.').astype(float)
#     #Convert to Fahrenheit
#     admission['Body Temperature'] = fahrenheit_covert(admission['Body Temperature'])

#     admission['Cardiac Frequency']=admission['Cardiac Frequency'].replace(0,np.nan)
#     admission['SaO2']=admission['SaO2'].replace(0,np.nan)
#     admission['Glycemia']=admission['Glycemia'].replace(0,np.nan)
#     admission['Systolic Blood Pressure']=admission['Systolic Blood Pressure'].replace(0,np.nan)

#     df3 = admission[VITAL_COLUMNS]

#     df3 = remove_missing(df3)

#     return df3


def create_lab_dataset(labs, dataset_admissions, dataset_vitals):

    labs['DETERMINACION.ITEM_LAB'] = labs['DETERMINACION.ITEM_LAB'].replace(RENAMED_LAB_MEASUREMENTS)

    #Limit to only rows for the ones known in Italy
    labs = labs.loc[labs['DETERMINACION.ITEM_LAB'].isin(LAB_METRICS)]
    #Reduce the number of columns
    labs = labs[LAB_COLS]

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

    #Oxygen levels
    df2['ABG: Oxygen Saturation (SaO2)'] = df2['ABG: Oxygen Saturation (SaO2)a']

    for (index_label, row_series) in df2.iterrows():
        if np.isnan(df2['ABG: Oxygen Saturation (SaO2)'].iloc[index_label]) and dataset_vitals['PATIENT ID'].isin([row_series['PATIENT.ID']]).any():
            df2['ABG: Oxygen Saturation (SaO2)'].iloc[index_label] = dataset_vitals[dataset_vitals['PATIENT ID']==row_series['PATIENT.ID']]['SaO2'].iloc[0]

    df2 = df2.drop(columns=['ABG: Oxygen Saturation (SaO2)a','ABG: Oxygen Saturation (SaO2)b'])


    #Merge the two dataframes admissions and labs together.
    df3 = pd.merge(dataset_admissions, df2, how='inner', left_on=['PATIENT ID','Date_Emergency'], right_on=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])
    #df4 = pd.merge(df1, df2, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])
    df3 = df3.drop(columns=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])

    dataset_lab_full = df3[LAB_FEATURES]
    dataset_lab_full = pd.DataFrame(dataset_lab_full.groupby(['PATIENT ID'], as_index=False).first())

    dataset_lab_full = remove_missing(dataset_lab_full)

    return dataset_lab_full


# def prepare_dataset_comorbidities(comorbidities_emerg, comorbidities_inpatient):
#     #Convert them to a long format
#     comorb_long1=pd.melt(comorbidities_emerg,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')
#     comorb_long2=pd.melt(comorbidities_inpatient,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')
#     #Concatanate the two dataframes
#     comorb_long = pd.concat([comorb_long1,comorb_long2])

#     #We will treat all types of diagnoses the same
#     comorb_long = comorb_long.drop(['DiagOrdering'], axis=1)
#     #Remove NA values
#     comorb_long=comorb_long.dropna()
#     #Convert PATIENT ID to integer
#     comorb_long['PATIENT ID'] = comorb_long['PATIENT ID'].astype(int)
#     #Remove the dot from the values column
#     comorb_long['values']= comorb_long['values'].str.replace(".","")
#     return comorb_long

# def create_dataset_comorbidities(comorb_long, icd_category, dataset_admissions):

#      #Load the diagnoses dict
#     if icd_category == 9:
#         icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd9.csv')
#     else:
#         icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd10.csv')

#     #The codes that are not mapped are mostly procedure codes or codes that are not of interest
#     icd_descr = pd.merge(comorb_long, icd_dict, how='inner', left_on=['values'], right_on=['DIAGNOSIS_CODE'])

#     #Now we need to restrict to the categories for which we have italian data
#     # cardiac arrhythmia -> 106
#     # acute renal failure -> 145
#     # chronic kidney disease -> 146
#     # CAD, heart disease -> 90
#     # diabetes -> 49, 50
#     # hypertension -> 87

#     #Create a list with the categories that we want
#     comorb_descr = icd_descr.loc[icd_descr['HCUP_ORDER'].isin(HCUP_LIST)]

#     #Limit only to the HCUP Description and drop the duplicates
#     comorb_descr = comorb_descr[['PATIENT ID','GROUP_HCUP']].drop_duplicates()

#     #Convert from long to wide format
#     comorb_descr = pd.get_dummies(comorb_descr, prefix=['GROUP_HCUP'])

#     #Now we will remove the GROUP_HCUP_ from the name of each column
#     comorb_descr = comorb_descr.rename(columns = lambda x: x.replace('GROUP_HCUP_', ''))

#     #Let's combine the diabetes columns to one
#     comorb_descr['Diabetes'] = comorb_descr[["Diabetes mellitus with complications", "Diabetes mellitus without complication"]].max(axis=1)

#     #Drop the other two columns
#     comorb_descr = comorb_descr.drop(columns=['Diabetes mellitus with complications', 'Diabetes mellitus without complication'])

#     dataset_comorbidities = pd.DataFrame(comorb_descr.groupby(['PATIENT ID'], as_index=False).max())

#     # Combine the comorbidities with the main filees
#     dataset_comorbidities = pd.merge(dataset_admissions['PATIENT ID'], dataset_comorbidities, how='left', left_on=['PATIENT ID'], right_on=['PATIENT ID'])
#     dataset_comorbidities = dataset_comorbidities.fillna(0)

#     return dataset_comorbidities


# def add_extra_features(dataset_admissions):

#     dataset_extra = dataset_admissions['PATIENT ID'].to_frame()

#     for i in CREMONA_EXTRA:
#         dataset_extra[i] = np.nan
#     return dataset_extra

# def filter_patients(datasets):
#     patients = datasets[0]['PATIENT ID'].astype(np.int64)

#     # Get common patients
#     for d in datasets[1:]:
#         patients = d[d['PATIENT ID'].astype(np.int64).isin(patients)]['PATIENT ID'].unique()

#     # Remove values not in patients (in place)
#     for d in datasets:
#         d.drop(d[~d['PATIENT ID'].astype(np.int64).isin(patients)].index, inplace=True)
#     return patients

# def fahrenheit_covert(temp_celsius):
#     temp_fahrenheit = ((temp_celsius * 9)/5)+ 32
#     return temp_fahrenheit

