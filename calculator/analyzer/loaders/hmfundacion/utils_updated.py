import numpy as np
import pandas as pd
import datetime

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

# Ali added
import analyzerutils
from analyzerutils import remove_missing
# Ali added

# Ali commented out
#from analyzer.utils import remove_missing
# Ali commented out

RENAMED_ADMISSION_COLUMNS = {
    'EDAD/AGE':'Age',
    'SEXO/SEX':'Gender',
    'DIAG ING/INPAT':'DIAG_TYPE',
    'MOTIVO_ALTA/DESTINY_DISCHARGE_ING':'Outcome',
    'F_INGRESO/ADMISSION_D_ING/INPAT':'Dt_Hospital_Admission',
    'F_INGRESO/ADMISSION_DATE_URG/EMERG':'Date_Emergency',
    'TEMP_PRIMERA/FIRST_URG/EMERG':'First_Body_Temperature',
    'TEMP_ULTIMA/LAST_URG/EMERG':'Last_Body_Temperature',
    'FC/HR_PRIMERA/FIRST_URG/EMERG':'First_Cardiac_Frequency',
    'FC/HR_ULTIMA/LAST_URG/EMERG':'Last_Cardiac_Frequency',
    'SAT_02_PRIMERA/FIRST_URG/EMERG':'First_SaO2',
    'SAT_02_ULTIMA/LAST_URG/EMERG':'Last_SaO2',
    'TA_MAX_PRIMERA/FIRST/EMERG_URG':'First_Max_Systolic_Blood_Pressure',
    'TA_MAX_ULTIMA/LAST_URGEMERG':'Last_Max_Systolic_Blood_Pressure',
    'TA_MIN_PRIMERA/FIRST_URG/EMERG':'First_Min_Systolic_Blood_Pressure',
    'TA_MIN_ULTIMA/LAST_URG/EMERG':'Last_Min_Systolic_Blood_Pressure'}

VITAL_COLUMNS = {
    'PATIENT ID',
    'First_Body_Temperature',
    'Last_Body_Temperature',
    'Maxtemperature_Admission',
    'First_Cardiac_Frequency',
    'Last_Cardiac_Frequency',
    'First_SaO2',
    'Last_SaO2',
    'Min_SaO2',
    'BLOOD_PRESSURE_ABNORMAL_B'
    }

DEMOGRAPHICS_COLUMNS = {'PATIENT ID','Gender','Age'}

ADMISSION_COLUMNS = ['PATIENT ID','Outcome','Dt_Hospital_Admission','Date_Emergency']

RENAMED_LAB_MEASUREMENTS = {'DD -- DIMERO D':'D-Dimer',
                            'PROCAL -- PROCALCITONINA':'Procalcitonin',
                            'PCR -- PROTEINA C REACTIVA':'C-Reactive Protein (CRP)',
                            'GPT -- GPT (ALT)':'Alanine Aminotransferase (ALT)',
                            'GOT -- GOT (AST)':'Aspartate Aminotransferase (AST)',
                            'LDH -- LDH': 'Lactate Dehydrogenase',
                            'CREA -- CREATININA':'Blood Creatinine',
                            'NA -- SODIO':'Blood Sodium',
                            'LEUC -- Leucocitos':'CBC: Leukocytes',
                            'LIN -- Linfocitos':'CBC: Lymphocytes',
                            'HGB -- Hemoglobina':'CBC: Hemoglobin',
                            'PLAQ -- Recuento de plaquetas':'CBC: Platelets',
                            'SO2C -- sO2c (Saturación de oxígeno)':'ABG: Oxygen Saturation (SaO2)a',
                            'SO2CV -- sO2c (Saturación de oxígeno)':'ABG: Oxygen Saturation (SaO2)b'}

LAB_METRICS = ['D-Dimer', 'Procalcitonin', 'C-Reactive Protein (CRP)', 'Alanine Aminotransferase (ALT)', 'Aspartate Aminotransferase (AST)', 'Lactate Dehydrogenase', 'Blood Creatinine', 'Blood Sodium', 'CBC: Leukocytes', 'CBC: Lymphocytes', 'CBC: Hemoglobin', 'CBC: Platelets', 'ABG: Oxygen Saturation (SaO2)a', 'ABG: Oxygen Saturation (SaO2)b']

LAB_COLS = ['PATIENT.ID','FECHA_PETICION.LAB_DATE','DETERMINACION.ITEM_LAB','RESULTADO.VAL_RESULT']

LAB_FEATURES = ['PATIENT ID', 'D-Dimer', 'Procalcitonin', 'C-Reactive Protein (CRP)', 'Alanine Aminotransferase (ALT)', 'Aspartate Aminotransferase (AST)', 'Lactate Dehydrogenase', 'Blood Creatinine', 'Blood Sodium', 'CBC: Leukocytes', 'CBC: Lymphocytes', 'CBC: Hemoglobin', 'CBC: Platelets','ABG: Oxygen Saturation (SaO2)']

FINAL_RENAMED_LAB_COLUMNS = {'Blood Creatinine':'CREATININE',
                            'Blood Sodium':'SODIUM',
                            'CBC: Leukocytes':'LEUCOCYTES',
                            'CBC: Lymphocytes':'LYMPHOCYTES',
                            'CBC: Hemoglobin':'HEMOGLOBIN',
                            'CBC: Platelets':'PLATELETS'}

RENAMED_COMORBID_COLUMNS = {
    'Disorders of lipid metabolism':'Dislipidemia',
    'Other nutritional; endocrine; and metabolic disorders':'Obesity',
    'Cardiac dysrhythmias':'AF',
    'HIV infection':'VIH'}

HCUP_LIST = [49, 50, 174,
                 87, 88, 171,
                 53,
                 58,
                 145, 146,
                 116, 117, 121, 122,
                 95,
                 5,
                 90, 92, 93, 95,
                 98, 100, 101, 102,
                 198, 199,
                 6, 139,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,                    38, 39, 40, 41, 42, 43]

HCUP_LIST_INPATIENT = [2, 145, 97, 105]

MED_COLS = ['PATIENT ID', 'INICIO_TRAT/DRUG_START_DATE', 'Treatment']

ATC_START_LIST_2 = ['B01','C09','N05','H02','J01']
    
ATC_START_LIST_3 = ['C07A','N06A']

ATC5_LIST = ['N02BA','B01AC','R03AB','R03AC','R03BA','A11CC','L03AB','P01BA','J05AR','B01AB']

ATC7_LIST = ['V03AN01','L04AC07']

def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns
    
    
def determine_regimen(dataset_medications):
    if dataset_medications['ANTICOAGULANTS'] == 1 and dataset_medications['ANTIVIRAL'] == 1 and dataset_medications['CLOROQUINE'] == 1:
        return 'All'
    elif dataset_medications['ANTIVIRAL'] == 1 and dataset_medications['CLOROQUINE'] == 1:
        return 'Chloroquine and Antivirals'
    elif dataset_medications['ANTICOAGULANTS'] == 1 and dataset_medications['CLOROQUINE'] == 1:
        return 'Chloroquine and Anticoagulants'
    elif dataset_medications['CLOROQUINE'] == 1 and dataset_medications['ANTICOAGULANTS'] == 0 and dataset_medications['ANTIVIRAL'] == 0:
        return 'Chloroquine Only'
    else:
        return 'Non-Chloroquine'


def create_dataset_admissions(admission):

    # Rename the columns for the admission
    admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)

    # Limit to only patients for whom we know the outcome
    types = ['Fallecimiento', 'Domicilio'] # Death, # Home
    admission = admission.loc[admission['Outcome'].isin(types)]
    
    # Dictionary for the outcome
    death_dict = {'Fallecimiento': 1,'Domicilio': 0}
    admission.Outcome = [death_dict[item] for item in admission.Outcome]

    # Convert to dates the appropriate information
    admission['Date_Emergency'] = pd.to_datetime(admission['Date_Emergency'], dayfirst=True).dt.date
    admission['Dt_Hospital_Admission'] = pd.to_datetime(admission['Dt_Hospital_Admission'], dayfirst=True).dt.date

    df1 = admission[ADMISSION_COLUMNS]
    df1['Hospital'] = 'HM Foundation'
    df1['Country'] = 'Spain'

    return df1


def create_dataset_demographics(admission):
    # Rename the columns for the admission
    admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)
    admission['Age'] = admission['Age'].replace('0',np.nan).astype(float)

    # Dictionary for the gender
    # gender = {'MALE': 0,'FEMALE': 1}
    # admission.Gender = [gender[item] for item in admission.Gender]

    df2 = admission[DEMOGRAPHICS_COLUMNS]
    
    return df2


def create_vitals_dataset(admission):
    # Rename the columns for the admission
    admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)
    
    # Reformatting the vital values at the emergency department
    # Get maximum temperature during admission (in Celsius)
    admission['First_Body_Temperature'] = admission['First_Body_Temperature'].replace('0',np.nan).str.replace(',','.').astype(float)
    admission['Last_Body_Temperature'] = admission['Last_Body_Temperature'].replace('0',np.nan).str.replace(',','.').astype(float)
    admission['Maxtemperature_Admission'] = admission[['First_Body_Temperature', 'Last_Body_Temperature']].max(axis=1)
    
    # Get minimum oxygen saturation at admission
    admission['First_SaO2'] = admission['First_SaO2'].replace(0,np.nan)
    admission['Last_SaO2'] = admission['Last_SaO2'].replace(0,np.nan)
    admission['Min_SaO2'] = admission[['First_SaO2', 'Last_SaO2']].min(axis=1)
    
    # Blood pressure abmornal limit
    admission['First_Max_Systolic_Blood_Pressure'] = admission['First_Max_Systolic_Blood_Pressure'].replace(0,np.nan)
    admission['Last_Max_Systolic_Blood_Pressure'] = admission['Last_Max_Systolic_Blood_Pressure'].replace(0,np.nan)
    admission['Max_Sys_BP_Admission'] = admission[['First_Max_Systolic_Blood_Pressure', 'Last_Max_Systolic_Blood_Pressure']].max(axis=1)
    
    admission['First_Min_Systolic_Blood_Pressure'] = admission['First_Min_Systolic_Blood_Pressure'].replace(0,np.nan)
    admission['Last_Min_Systolic_Blood_Pressure'] = admission['Last_Min_Systolic_Blood_Pressure'].replace(0,np.nan)
    admission['Min_Sys_BP_Admission'] = admission[['First_Min_Systolic_Blood_Pressure', 'Last_Min_Systolic_Blood_Pressure']].min(axis=1)
    
    admission['BLOOD_PRESSURE_ABNORMAL_B'] = admission['Min_Sys_BP_Admission'].apply(lambda x: 1 if x < 100 else np.nan if pd.isnull(x) else 0)

    df3 = admission[VITAL_COLUMNS]
    
    df3 = df3[['PATIENT ID',
    'First_Body_Temperature',
    'Last_Body_Temperature',
    'Maxtemperature_Admission',
    'First_Cardiac_Frequency',
    'Last_Cardiac_Frequency',
    'First_SaO2',
    'Last_SaO2',
    'Min_SaO2',
    'BLOOD_PRESSURE_ABNORMAL_B']]

    #df3 = remove_missing(df3)

    return df3


def create_lab_dataset(labs, dataset_admissions, dataset_vitals):

    labs['DETERMINACION.ITEM_LAB'] = labs['DETERMINACION.ITEM_LAB'].replace(RENAMED_LAB_MEASUREMENTS)

    # Limit to only rows we need
    labs = labs.loc[labs['DETERMINACION.ITEM_LAB'].isin(LAB_METRICS)]
    
    # Reduce the number of columns
    labs = labs[LAB_COLS]

    # Convert to Date the date column
    labs['FECHA_PETICION.LAB_DATE']= pd.to_datetime(labs['FECHA_PETICION.LAB_DATE'], dayfirst=True).dt.date
    
    # Convert lab result column to float
    labs['RESULTADO.VAL_RESULT'] = labs['RESULTADO.VAL_RESULT'].str.extract('(\d+(?:\.\d+)?)', expand=False).astype(float)

    # Convert to wide format
    df2 = labs.pivot_table(index=['PATIENT.ID','FECHA_PETICION.LAB_DATE'], columns='DETERMINACION.ITEM_LAB', values='RESULTADO.VAL_RESULT')
    df2 = pd.DataFrame(df2.to_records())

    df2['PATIENT.ID'] = df2['PATIENT.ID'].astype(int)

    # Oxygen levels
    df2['ABG: Oxygen Saturation (SaO2)'] = df2['ABG: Oxygen Saturation (SaO2)a']

    for (index_label, row_series) in df2.iterrows():
        if np.isnan(df2['ABG: Oxygen Saturation (SaO2)'].iloc[index_label]) and dataset_vitals['PATIENT ID'].isin([row_series['PATIENT.ID']]).any():
            df2['ABG: Oxygen Saturation (SaO2)'].iloc[index_label] = dataset_vitals[dataset_vitals['PATIENT ID']==row_series['PATIENT.ID']]['Min_SaO2'].iloc[0]

    df2 = df2.drop(columns=['ABG: Oxygen Saturation (SaO2)a','ABG: Oxygen Saturation (SaO2)b'])

    # Merge the two dataframes admissions and labs together
    df3 = pd.merge(dataset_admissions, df2, how='inner', left_on=['PATIENT ID','Date_Emergency'], right_on=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])
    #df4 = pd.merge(df1, df2, how='inner', left_on=['PATIENT ID','Date_Admission'], right_on=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])
    
    df3 = df3.drop(columns=['PATIENT.ID','FECHA_PETICION.LAB_DATE'])

    dataset_lab_full = df3[LAB_FEATURES]
    dataset_lab_full = pd.DataFrame(dataset_lab_full.groupby(['PATIENT ID'], as_index=False).first())
    
    # D-Dimer binary limit = >= 0.5 mg/L (need to convert ng/L)
    dataset_lab_full['DDDIMER_B'] = dataset_lab_full['D-Dimer'].apply(lambda x: 1 if x >= 500 else np.nan if pd.isnull(x) else 0)
    
    # Procalcitonin binary limit = >= 0.5 ng/L (already in ng/L)
    dataset_lab_full['PROCALCITONIN_B'] = dataset_lab_full['Procalcitonin'].apply(lambda x: 1 if x >= 0.5 else np.nan if pd.isnull(x) else 0)
    
    # C-Reactive protein binary limit = >= 10 mg/L (already in mg/L)
    dataset_lab_full['PCR_B'] = dataset_lab_full['C-Reactive Protein (CRP)'].apply(lambda x: 1 if x >= 10 else np.nan if pd.isnull(x) else 0)
    
    # Transaminase (ALT, AST) binary limit >= 40 U/L (already in U/L)
    dataset_lab_full['ALT_B'] = dataset_lab_full['Alanine Aminotransferase (ALT)'].apply(lambda x: 1 if x >= 40 else np.nan if pd.isnull(x) else 0)
    dataset_lab_full['AST_B'] = dataset_lab_full['Aspartate Aminotransferase (AST)'].apply(lambda x: 1 if x >= 40 else np.nan if pd.isnull(x) else 0)
    dataset_lab_full['TRANSAMINASES_B'] = dataset_lab_full[['ALT_B', 'AST_B']].max(axis=1)
    
    # LDH binary limit
    dataset_lab_full['LDL_B'] = dataset_lab_full['Lactate Dehydrogenase'].apply(lambda x: 1 if x > 222  else np.nan if pd.isnull(x) else 0)
    
    # Get binary for oxygen saturation at admission below 92% (yes = 1, no = 0)
    dataset_lab_full['SAT02_BELOW92'] = dataset_lab_full['ABG: Oxygen Saturation (SaO2)'].apply(lambda x: 1 if x < 92 else np.nan if pd.isnull(x) else 0)
    
    # Drop columns no longer needed
    dataset_lab_full = dataset_lab_full.drop(columns=["D-Dimer","Procalcitonin","C-Reactive Protein (CRP)","Alanine Aminotransferase (ALT)","Aspartate Aminotransferase (AST)","Lactate Dehydrogenase","ABG: Oxygen Saturation (SaO2)","ALT_B","AST_B"])
    
    # Rename the columns that need renaming
    dataset_lab_full = dataset_lab_full.rename(columns=FINAL_RENAMED_LAB_COLUMNS)

    # Convert continuous lab values to correct units where needed
    dataset_lab_full['LEUCOCYTES'] = dataset_lab_full['LEUCOCYTES'].apply(lambda x: x*1000)
    
    dataset_lab_full['LYMPHOCYTES'] = dataset_lab_full['LYMPHOCYTES'].apply(lambda x: x*1000)
        
    dataset_lab_full['PLATELETS'] = dataset_lab_full['PLATELETS'].apply(lambda x: x*1000)
    
    #dataset_lab_full = remove_missing(dataset_lab_full)

    return dataset_lab_full


def prepare_dataset_comorbidities(comorbidities_emerg):
    # Can be used to import either emergency comorbidities or inpatient comorbidities
    
    # Convert them to a long format
    comorb_long1 = pd.melt(comorbidities_emerg,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')
#     comorb_long2 = pd.melt(comorbidities_inpatient,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')
    
    # Concatenate the two dataframes
#     comorb_long = pd.concat([comorb_long1,comorb_long2])
    comorb_long = comorb_long1

    # We will treat all types of diagnoses the same
    comorb_long = comorb_long.drop(['DiagOrdering'], axis=1)
    
    # Remove NA values
    comorb_long = comorb_long.dropna()
    
    # Convert PATIENT ID to integer
    comorb_long['PATIENT ID'] = comorb_long['PATIENT ID'].astype(int)
    
    # Remove the dot from the values column
    comorb_long['values'] = comorb_long['values'].str.replace(".","")
    
    return comorb_long


def prepare_dataset_comorbidities_both(comorbidities_emerg, comorbidities_inpatient):
    # Used to import both comorbidities datasets and concatenate them
    
    # Convert them to a long format
    comorb_long1 = pd.melt(comorbidities_emerg,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')
    comorb_long2 = pd.melt(comorbidities_inpatient,id_vars=['PATIENT ID'],var_name='DiagOrdering', value_name='values')
    
    # Concatenate the two dataframes
    comorb_long = pd.concat([comorb_long1,comorb_long2])

    # We will treat all types of diagnoses the same
    comorb_long = comorb_long.drop(['DiagOrdering'], axis=1)
    
    # Remove NA values
    comorb_long = comorb_long.dropna()
    
    # Convert PATIENT ID to integer
    comorb_long['PATIENT ID'] = comorb_long['PATIENT ID'].astype(int)
    
    # Remove the dot from the values column
    comorb_long['values'] = comorb_long['values'].str.replace(".","")
    
    return comorb_long


def create_dataset_comorbidities_admissions(comorb_long, icd_category, dataset_admissions):
    # Use emergency (admissions) dataset only for renal insufficiency, anylungdisease, and anycerebrovascular disease
    # Use emergency (admissions) and inpatient datasets for other comorbidities
    
    # Load the diagnoses dictionary
    if icd_category == 9:
        icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd9.csv')
    else:
        icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd10.csv') # Note: HM Hospitals datasets use ICD 10

    # The codes that are not mapped are mostly procedure codes or codes that are not of interest
    icd_descr = pd.merge(comorb_long, icd_dict, how='inner', left_on=['values'], right_on=['DIAGNOSIS_CODE'])

    # Now we need to restrict to the categories for matching to hope data
    # * Indicates that no patients in the HM dataset have this diagnosis in the admissions data
    # DIAGNOSIS --> HCUP_ORDER(S)
    # ------------------------------
    # RENALINSUF -> 145, 146 (Need to be more specific here and use diagnosis codes)
    # ANYLUNGDISEASE -> 116, 117, 121, 122
    # ANYCEREBROVASCULARDISEASE -> 98, 100*, 101, 102*

    # Create a list with the categories that we want
    comorb_descr = icd_descr.loc[icd_descr['HCUP_ORDER'].isin(HCUP_LIST)]

    # Limit only to the HCUP Description and drop the duplicates
    comorb_descr = comorb_descr[['PATIENT ID','GROUP_HCUP']].drop_duplicates()

    # Convert from long to wide format
    comorb_descr = pd.get_dummies(comorb_descr, prefix=['GROUP_HCUP'])

    # Now we will remove the GROUP_HCUP_ from the name of each column
    comorb_descr = comorb_descr.rename(columns = lambda x: x.replace('GROUP_HCUP_', ''))

    # Combine columns into one where needed
    # Let's combine the renal failure columns to one
    comorb_descr['RenalInsuf'] = comorb_descr[["Acute and unspecified renal failure","Chronic kidney disease"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Acute and unspecified renal failure","Chronic kidney disease"])
    
    # Let's combine the lung disease columns to one
    comorb_descr['AnyLungDisease'] = comorb_descr[["Chronic obstructive pulmonary disease and bronchiectasis","Asthma","Lung disease due to external agents","Other lower respiratory disease"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Chronic obstructive pulmonary disease and bronchiectasis","Asthma","Lung disease due to external agents","Other lower respiratory disease"])
    
    # Let's combine the cerebrovascular disease columns to one
    comorb_descr['AnyCerebrovascularDisease'] = comorb_descr[["Acute cerebrovascular disease","Transient cerebral ischemia"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Acute cerebrovascular disease","Transient cerebral ischemia"])
    
    # Keep only one record (row) per patient
    dataset_comorbidities = pd.DataFrame(comorb_descr.groupby(['PATIENT ID'], as_index=False).max())

    # Combine the comorbidities with the main files
    dataset_comorbidities = pd.merge(dataset_admissions['PATIENT ID'], dataset_comorbidities, how='left', left_on=['PATIENT ID'], right_on=['PATIENT ID'])
    dataset_comorbidities = dataset_comorbidities.fillna(0)

    return dataset_comorbidities


def create_dataset_comorbidities_inpatient(comorb_long, icd_category, dataset_admissions):
    # Use inpatient datasets for comorbidities developed due to COVID

    # Load the diagnoses dictionary
    if icd_category == 9:
        icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd9.csv')
    else:
        icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd10.csv') # Note: HM Hospitals datasets use ICD 10

    # The codes that are not mapped are mostly procedure codes or codes that are not of interest
    icd_descr = pd.merge(comorb_long, icd_dict, how='inner', left_on=['values'], right_on=['DIAGNOSIS_CODE'])

    # Now we need to restrict to the categories for matching to hope data
    # DIAGNOSIS --> HCUP_ORDER(S)
    # ------------------------------
    # SEPSIS -> 2
    # ACUTE RENAL FAILURE -> 145
    # HEART FAILURE -> 97
    # EMBOLIC EVENT -> 105

    # Create a list with the categories that we want
    comorb_descr = icd_descr.loc[icd_descr['HCUP_ORDER'].isin(HCUP_LIST_INPATIENT)]

    # Limit only to the HCUP Description and drop the duplicates
    comorb_descr = comorb_descr[['PATIENT ID','GROUP_HCUP']].drop_duplicates()

    # Convert from long to wide format
    comorb_descr = pd.get_dummies(comorb_descr, prefix=['GROUP_HCUP'])

    # Now we will remove the GROUP_HCUP_ from the name of each column
    comorb_descr = comorb_descr.rename(columns = lambda x: x.replace('GROUP_HCUP_', ''))

    # Combine all columns into one
    # Let's combine the diabetes columns to one
    comorb_descr['Comorbid_In'] = comorb_descr[["Acute and unspecified renal failure", "Congestive heart failure; nonhypertensive","Septicemia (except in labor)"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Acute and unspecified renal failure", "Congestive heart failure; nonhypertensive","Septicemia (except in labor)"])
    
    # Rename columns
    dataset_comorbidities = pd.DataFrame(comorb_descr.groupby(['PATIENT ID'], as_index=False).max())

    # Combine the comorbidities with the main files
    dataset_comorbidities = pd.merge(dataset_admissions['PATIENT ID'], dataset_comorbidities, how='left', left_on=['PATIENT ID'], right_on=['PATIENT ID'])
    dataset_comorbidities = dataset_comorbidities.fillna(0)

    return dataset_comorbidities


def create_dataset_comorbidities_both(comorb_long, icd_category, dataset_admissions):
    # Use emergency (admissions) dataset only for renal insufficiency, anylungdisease, and anycerebrovascular disease
    # Use emergency (admissions) and inpatient datasets for other comorbidities
    
    # Load the diagnoses dictionary
    if icd_category == 9:
        icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd9.csv')
    else:
        icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd10.csv') # Note: HM Hospitals datasets use ICD 10

    # The codes that are not mapped are mostly procedure codes or codes that are not of interest
    icd_descr = pd.merge(comorb_long, icd_dict, how='inner', left_on=['values'], right_on=['DIAGNOSIS_CODE'])

    # Now we need to restrict to the categories for matching to hope data
    # DIAGNOSIS --> HCUP_ORDER(S)
    # ------------------------------
    # DIABETES -> 49, 50, 174
    # HYPERTENSION -> 87, 88, 171
    # DISLIPIDEMIA -> 53 (Disorders of lipid metabolism) 
    # OBESITY -> 58 (Need to be more specific here and used diagnosis codes)
    # SMOKING -> 
    # AF -> 95
    # VIH -> 5
    # ANYHEARTDISEASE -> 90, 92, 93, 95
    # MAINHEARTDISEASE -> DIAGNOSIS_CODE I255 Ischemic cardiomyopathy, I470, I498, I499 Arrythmias (In Hope: CORONARY/VALVE/HEARTFAILURE/MYOPATHY/ARRHYTHMIAS/COMBINED)
    # CONECTIVEDISEASE -> 198, 199
    # LIVER_DISEASE -> 6, 139
    # CANCER -> 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43

    # Create a list with the categories that we want
    comorb_descr = icd_descr.loc[icd_descr['HCUP_ORDER'].isin(HCUP_LIST)]

    # Limit only to the HCUP Description and drop the duplicates
    comorb_descr = comorb_descr[['PATIENT ID','GROUP_HCUP']].drop_duplicates()

    # Convert from long to wide format
    comorb_descr = pd.get_dummies(comorb_descr, prefix=['GROUP_HCUP'])

    # Now we will remove the GROUP_HCUP_ from the name of each column
    comorb_descr = comorb_descr.rename(columns = lambda x: x.replace('GROUP_HCUP_', ''))

    # Combine columns into one where needed
    # Let's combine the diabetes columns to one
    comorb_descr['Diabetes'] = comorb_descr[["Diabetes mellitus without complication", "Diabetes mellitus with complications","Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Diabetes mellitus without complication", "Diabetes mellitus with complications","Diabetes or abnormal glucose tolerance complicating pregnancy; childbirth; or the puerperium"])
    
    # Let's combine the hypertension columns to one
    comorb_descr['Hypertension'] = comorb_descr[["Essential hypertension","Hypertension with complications and secondary hypertension"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Essential hypertension","Hypertension with complications and secondary hypertension"])
    
    # Let's combine the heart disease columns to one
    comorb_descr['AnyHeartDisease'] = comorb_descr[["Coronary atherosclerosis and other heart disease","Pulmonary heart disease","Other and ill-defined heart disease","Cardiac dysrhythmias"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Coronary atherosclerosis and other heart disease","Pulmonary heart disease","Other and ill-defined heart disease"])
    
    # Let's combine the connective disease columns to one
    comorb_descr['ConectiveDisease'] = comorb_descr[["Systemic lupus erythematosus and connective tissue disorders","Other connective tissue disease"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Systemic lupus erythematosus and connective tissue disorders","Other connective tissue disease"])
    
    # Let's combine the liver disease columns to one
    comorb_descr['Liver_Disease'] = comorb_descr[["Hepatitis","Other liver diseases"]].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=["Hepatitis","Other liver diseases"])
    
    # Let's combine the cancer columns to one
    cancer_cols =[x for x in comorb_descr.columns[comorb_descr.columns.str.contains(pat='cancer',case=False)]]
    other_cancers = ["Melanomas of skin","Hodgkin`s disease","Non-Hodgkin`s lymphoma","Leukemias","Multiple myeloma","Secondary malignancies","Malignant neoplasm without specification of site"]
    cancer_cols.extend(other_cancers)
    comorb_descr['Cancer'] = comorb_descr[cancer_cols].max(axis=1)
    # Drop the other columns
    comorb_descr = comorb_descr.drop(columns=cancer_cols)
    
    # Rename columns
    comorb_descr = comorb_descr.rename(columns=RENAMED_COMORBID_COLUMNS)
    
    comorb_descr = comorb_descr.rename(columns={'HIV infection':'VIH'})

    dataset_comorbidities = pd.DataFrame(comorb_descr.groupby(['PATIENT ID'], as_index=False).max())

    # Combine the comorbidities with the main files
    dataset_comorbidities = pd.merge(dataset_admissions['PATIENT ID'], dataset_comorbidities, how='left', left_on=['PATIENT ID'], right_on=['PATIENT ID'])
    dataset_comorbidities = dataset_comorbidities.fillna(0)

    return dataset_comorbidities


def create_dataset_medications(medications):
# ATC_START_LIST_2 = ['B01','C09','N05','H02','J01']
    
# ATC_START_LIST_3 = ['C07A','N06A']

# ATC5_LIST = ['N02BA','B01AC','R03AB','R03AC','R03BA','A11CC','L03AB','P01BA','J05AR','B01AB']

# ATC7_LIST = ['V03AN01','L04AC07']
    
    medications['INICIO_TRAT/DRUG_START_DATE']= pd.to_datetime(medications['INICIO_TRAT/DRUG_START_DATE'], dayfirst=True).dt.date
    medications['FIN_TRAT/DRUG_END_DATE']= pd.to_datetime(medications['FIN_TRAT/DRUG_END_DATE'], dayfirst=True).dt.date

    # Create helper columns
    medications['ID_ATC2'] = medications['ID_ATC5'].str[:3]
    medications['ID_ATC3'] = medications['ID_ATC5'].str[:4]
    
    # Limit to only the rows we are interested in and create treatment column
    medications_atc2 = medications.loc[medications['ID_ATC2'].isin(ATC_START_LIST_2)]
    medications_atc2['Treatment'] = medications_atc2['ID_ATC2'].apply(lambda x: 'IN_ORALANTICOAGL' if x == 'B01' else 'IN_ACEI_ARB' if x == 'C09' else 'IN_BENZODIACEPINES' if x == 'N05' else 'CORTICOSTEROIDS' if x == 'H02' else 'ANTIBIOTICS' if x == 'J01' else None)
    print("Number of null treatment rows: ",medications_atc2.Treatment.isnull().sum())
    
    medications_atc3 = medications.loc[medications['ID_ATC3'].isin(ATC_START_LIST_3)]
    medications_atc3['Treatment'] = medications_atc3['ID_ATC3'].apply(lambda x: 'IN_BETABLOCKERS' if x == 'C07A' else 'IN_ANTIDEPRESSANT' if x == 'N06A' else None)
    print("Number of null treatment rows: ",medications_atc2.Treatment.isnull().sum())
    
    medications_atc5 = medications.loc[medications['ID_ATC5'].isin(ATC5_LIST)]
    medications_atc5['Treatment'] = medications_atc5['ID_ATC5'].apply(lambda x: 'IN_PREVIOUSASPIRIN' if x == 'N02BA' else 'IN_OTHERANTIPLATELET' if x == 'B01AC' else 'IN_BETAGONISTINHALED' if x == 'R03AB' else 'IN_BETAGONISTINHALED' if x == 'R03AC' else 'IN_GLUCORTICOIDSINHALED' if x == 'R03BA' else 'IN_DVITAMINSUPLEMENT' if x == 'A11CC' else 'INTERFERONOR' if x == 'L03AB' else 'CLOROQUINE' if x == 'P01BA' else 'ANTIVIRAL' if x == 'J05AR' else 'ANTICOAGULANTS' if x == 'B01AB' else None)
    print("Number of null treatment rows: ",medications_atc5.Treatment.isnull().sum())
    
    medications_atc7 = medications.loc[medications['ID_ATC7'].isin(ATC7_LIST)]
    medications_atc7['Treatment'] = medications_atc7['ID_ATC7'].apply(lambda x: 'HOME_OXIGEN_THERAPY' if x == 'V03AN01' else 'TOCILIZUMAB' if x == 'L04AC07' else None)
    print("Number of null treatment rows: ",medications_atc7.Treatment.isnull().sum())
    
    # Concatenate dataframes
    frames = [medications_atc2, medications_atc3, medications_atc5, medications_atc7]
    medications_merged = pd.concat(frames)
    
    # Limit only to the columns we want
    medications_new = medications_merged[MED_COLS]
    
    # Create features from treatment column
    medications_new = pd.get_dummies(medications_new, prefix=['Treatment'], columns=['Treatment'])

    # Now we will remove the Treatment_ from the name of each column
    medications_new = medications_new.rename(columns = lambda x: x.replace('Treatment_', ''))
    
    medications_new['ACEI_ARBS'] = medications_new['IN_ACEI_ARB']
    
    # Remove duplicates and drop date column
    dataset_medications = pd.DataFrame(medications_new.groupby(['PATIENT ID'], as_index=False).max())
    dataset_medications = dataset_medications.drop(columns=['INICIO_TRAT/DRUG_START_DATE'])
    
    # Determine regimen
    dataset_medications['Regimen'] = dataset_medications.apply(determine_regimen, axis=1)
    
    return dataset_medications

###### FUNCTION BELOW IS USED IN DATASET CREATION SCRIPT
def filter_patients(datasets):
    patients = datasets[0]['PATIENT ID'].astype(np.int64)

    # Get common patients
    for d in datasets[1:]:
        patients = d[d['PATIENT ID'].astype(np.int64).isin(patients)]['PATIENT ID'].unique()

    # Remove values not in patients (in place)
    for d in datasets:
        d.drop(d[~d['PATIENT ID'].astype(np.int64).isin(patients)].index, inplace=True)
    return patients


###### FUNCTIONS BELOW ARE NOT USED

def add_extra_features(dataset_admissions):

    dataset_extra = dataset_admissions['PATIENT ID'].to_frame()

    for i in CREMONA_EXTRA:
        dataset_extra[i] = np.nan
    return dataset_extra




def fahrenheit_covert(temp_celsius):
    temp_fahrenheit = ((temp_celsius * 9)/5)+ 32
    return temp_fahrenheit