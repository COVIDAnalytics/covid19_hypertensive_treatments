import numpy as np
import pandas as pd
import datetime

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

RENAMED_ADMISSION_COLUMNS = {
    'EDAD/AGE':'Age','SEXO/SEX':'Sex',
    'DIAG ING/INPAT':'DIAG_TYPE',
    'MOTIVO_ALTA/DESTINY_DISCHARGE_ING':'death',
    'F_INGRESO/ADMISSION_D_ING/INPAT':'Date_Admission',
    'F_INGRESO/ADMISSION_DATE_URG/EMERG':'Date_Emergency',
    'TEMP_PRIMERA/FIRST_URG/EMERG':'Temperature Celsius',
    'FC/HR_PRIMERA/FIRST_URG/EMERG':'Cardiac Frequency',
    'GLU_PRIMERA/FIRST_URG/EMERG':'Glycemia',
    'SAT_02_PRIMERA/FIRST_URG/EMERG':'ABG: Oxygen Saturation (SaO2)',
    'TA_MAX_PRIMERA/FIRST/EMERG_URG':'Systolic Blood Pressure'}

VITAL_COLUMNS ={
    'Temperature Celsius',
    'Cardiac Frequency',
    'ABG: Oxygen Saturation (SaO2)',
    'Systolic Blood Pressure'
    }

DEMOGRAPHICS_COLUMNS={'Sex','Age'}

ADMISSION_COLUMNS = ['PATIENT ID','death','DIAG_TYPE','Date_Admission','Date_Emergency']


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



def get_percentages(df, missing_type=np.nan):
    if np.isnan(missing_type):
        df = df.isnull()  # Check what is NaN
    elif missing_type is False:
        df = ~df  # Check what is False

    percent_missing = df.sum() * 100 / len(df)
    return pd.DataFrame({'percent_missing': percent_missing})


def remove_missing(df, missing_type=np.nan, nan_threashold=40, impute=True):
    missing_values = get_percentages(df, missing_type)
    df_features = missing_values[missing_values['percent_missing'] < nan_threashold].index.tolist()

    df = df[df_features]

    if impute:
        imp_mean = IterativeImputer(random_state=0)
        imp_mean.fit(df)
        imputed_df = imp_mean.transform(df)
        df = pd.DataFrame(imputed_df, index=df.index, columns=df.columns)

    return df


def create_dataset_admissions(admission):
    
    #Rename the columns for the admission
    admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)
    
    #Limit to only patients for whom we know the outcome
    types = ['Fallecimiento', 'Domicilio']
    admission = admission.loc[admission['death'].isin(types)]
    #Dictionary for the outcome
    death_dict = {'Fallecimiento': 1,'Domicilio': 0} 
    admission.death = [death_dict[item] for item in admission.death] 
    
    #Convert to Dates the appropriate information
    admission['Date_Emergency']= pd.to_datetime(admission['Date_Emergency']).dt.date 
    admission['Date_Admission']= pd.to_datetime(admission['Date_Admission']).dt.date 

    df1 = admission[ADMISSION_COLUMNS]
    
    return df1


def create_dataset_demographics(admission):
    #Rename the columns for the admission
    admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)

    #Dictionary for the gender 
    gender = {'MALE': 0,'FEMALE': 1} 
    admission.Sex = [gender[item] for item in admission.Sex] 
    
    df2 = admission[DEMOGRAPHICS_COLUMNS]
    return df2


def create_vitals_dataset(admission):
    #Rename the columns for the admission
    admission = admission.rename(columns=RENAMED_ADMISSION_COLUMNS)
    #Reformatting the vital values at the emergency department
    admission['Temperature Celsius'] = admission['Temperature Celsius'].replace('0',np.nan).str.replace(',','.').astype(float)
    admission['Cardiac Frequency']=admission['Cardiac Frequency'].replace(0,np.nan)
    admission['SaO2']=admission['ABG: Oxygen Saturation (SaO2)'].replace(0,np.nan)
    admission['Glycemia']=admission['Glycemia'].replace(0,np.nan)
    admission['Systolic Blood Pressure']=admission['Systolic Blood Pressure'].replace(0,np.nan)

    df3 = admission[VITAL_COLUMNS]
    return df3



def create_lab_dataset(lab, patients):

    return dataset_lab_full

def create_dataset_comorbidities(comorbidities, patients):

 

    return dataset_comorbidities.set_index('NOSOLOGICO')





def cleanup_discharge_info(discharge_info):

  
    return discharge_info



def filter_patients(datasets):

    patients = datasets[0]['NOSOLOGICO'].astype(np.int64)

    # Get common patients
    for d in datasets[1:]:
        patients = d[d['NOSOLOGICO'].astype(np.int64).isin(patients)]['NOSOLOGICO'].unique()


    # Remove values not in patients (in place)
    for d in datasets:
        d.drop(d[~d['NOSOLOGICO'].astype(np.int64).isin(patients)].index, inplace=True)

    return patients

