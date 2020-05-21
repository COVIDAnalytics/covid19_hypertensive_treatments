import numpy as np
import pandas as pd
import datetime
import re

RENAMED_COLUMNS = {'SEX':'Gender',
    'AGE':'Age',
    'PAT_MRN_ID':'PATIENT_ID',
    'PAT_ENC_CSN_ID':'ENCOUNTER_ID',
    'PAT_CLASS_CLEAN':'Patient_Class',
    'MORTALITY_CLEAN':'Outcome',
    'HOSP_ADMSN_TIME':'Date_Admission',
    'HOSP_DISCH_TIME':'Date_Discharge',
    'MIN_SPO2':'SaO2', # approximate match
    'MAX_TEMP':'Body Temperature', 
    'MAX_RR':'Respiratory Frequency', 
    'MAX_SYSTOLIC':'Systolic Blood Pressure',
    'max_pulse':'Cardiac Frequency',
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
    'PLATELET':'CBC: Platelets',
    'TOTAL_BILIRUBIN':'Total Bilirubin', 
    'RDW':'CBC: Red cell Distribution Width (RDW)', 
    'SERUM_CHOLINESTERASE':'Cholinesterase',
    'GLUCOSE':'Glycemia_1',
    'GLUCOSE_POC':'Glycemia_2',
    'INR':'Prothrombin Time (INR)',
    'UREA_NITROGEN':'Blood Urea Nitrogen (BUN)',
    'WHITE_BLOOD_CELL':'CBC: Leukocytes'}

DEMOGRAPHICS_COLUMNS= ['Gender','Age']

ADMISSIONS_COLUMNS = ['Outcome']

VITALS_COLUMNS  = ['SaO2', # approximate matchh
                    'Body Temperature', 
                    'Respiratory Frequency', 
                    'Systolic Blood Pressure',
                    'Cardiac Frequency']

LAB_COLUMNS = ['ABG: PaO2', 
    'ABG: pH', 
    'Aspartate Aminotransferase (AST)', 
    'Blood Calcium', 
    'CBC: Hemoglobin', 
    'CBC: Mean Corpuscular Volume (MCV)', 
    'ABG: MetHb',
    'C-Reactive Protein (CRP)',
    'ABG: Lactic Acid', 
    'ABG: PaCO2', 
    'Alanine Aminotransferase (ALT)', 
    'Blood Amylase',
    'Blood Creatinine', 
    'Blood Sodium',
    'Potassium Blood Level',
    'ABG: COHb',  
    'D-Dimer', 
    'CBC: Platelets',
    'Total Bilirubin', 
    'CBC: Red cell Distribution Width (RDW)', 
    'Cholinesterase',
    'Glycemia',
    'Prothrombin Time (INR)',
    'Blood Urea Nitrogen (BUN)',
    'CBC: Leukocytes']

COMORBIDITIES_COLUMNS = ['Essential hypertension', 'Diabetes', 
                         'Coronary atherosclerosis and other heart disease', 'Cardiac dysrhythmias',
                        'Chronic kidney disease']
                      
def get_lab_dates(t):
    # TODO: Find better way to do so. Nested try-except is not nice.
    if pd.isnull(t):
        return np.NaN
    t = t[0:10]
    try:
        date = datetime.datetime.strptime(t, '%m/%d/%Y')
    except ValueError:
        date = datetime.datetime.strptime(t, '%Y-%m-%d')
    return date

def clean_labs(l):
    if pd.isnull(l):
        return np.NaN
    elif bool(re.match("(no.* detect)|(negative)", l.lower())):
        return "Negative"
    elif bool(re.match("(^detect)|(positive)", l.lower())):
        return "Positive"
    else:
        return l
    
def clean_patient_class(c):
    if pd.isnull(c):
        return np.NaN
    elif bool(re.match(".*inpatient", c.lower())):
        return "Inpatient"
    elif bool(re.match(".*outpatient", c.lower())):
        return "Outpatient"
    elif bool(re.match(".*emergency", c.lower())):
        return "Emergency"
    else:
        return "Other"
    
def clean_mortality(m):
    if pd.isnull(m):
        return np.NaN
    elif bool(re.match("^expired", m.lower())):
        return "Expired"
    else:
        return "Alive"
    
def hhc_site(m):
    if pd.isnull(m):
        return np.NaN
    elif bool(re.match(".*Hartford Hospital", m)):
        return "Main"
    else:
        return "Other"


def try_parse_float(s, val=np.nan, print_error=False):
  try:
    return float(s)
  except ValueError:
    print(s)
    return val
