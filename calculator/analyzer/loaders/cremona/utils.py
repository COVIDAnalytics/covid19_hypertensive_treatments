import numpy as np
import pandas as pd
import datetime

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

from analyzer.utils import remove_missing

# ICD9 COVID diagnosis Italian codes
LIST_DIAGNOSIS = ['4808', '4803', 'V0182', '7982']
LIST_REMOVE_COMORBIDITIES = ["Immunizations and screening for infectious disease",
                             "Pneumonia (except that caused by tuberculosis or Genderually transmitted disease)",
                             "Respiratory failure; insufficiency; arrest (adult)",
                             "Residual codes; unclassified",
                             "Diabetes mellitus without complication",
                             "Diabetes mellitus with complications",
                             "Influenza",
                             "Acute and unspecified renal failure"]

SWAB_WITH_LAB_COLUMNS = ['Age',
                        'Gender',
                        'Body Temperature',
                        #'Systolic Blood Pressure',
                        'Respiratory Frequency',
                        'Cardiac Frequency',
                        'C-Reactive Protein (CRP)',
                        'Blood Calcium',
                        'CBC: Leukocytes',
                        'Aspartate Aminotransferase (AST)',
                        'ABG: PaO2',
                        'Prothrombin Time (INR)',
                        'CBC: Hemoglobin',
                        'ABG: pH',
                        'Cholinesterase',
                        'Blood Urea Nitrogen (BUN)',
                        'ABG: MetHb',
                        'Total Bilirubin',
                        'CBC: Mean Corpuscular Volume (MCV)',
                        'Glycemia']

SUBSET_COLUMNS_WITHOUT_ABG = ['Age', 'Gender', 'Body Temperature', 
                            'ABG: Oxygen Saturation (SaO2)','Cardiac Frequency',  'Respiratory Frequency', 
                            #'Systolic Blood Pressure', 
                            'Alanine Aminotransferase (ALT)', 'Aspartate Aminotransferase (AST)', 
                            'Total Bilirubin',  'Blood Calcium', 'Blood Creatinine', 'Blood Sodium', 
                            'Blood Urea Nitrogen (BUN)', 'CBC: Hemoglobin', 'CBC: Mean Corpuscular Volume (MCV)', 
                            'CBC: Platelets', 'CBC: Red cell Distribution Width (RDW)', 'CBC: Leukocytes', 
                            'C-Reactive Protein (CRP)', 'Prothrombin Time (INR)']

COLUMNS_WITHOUT_ABG = ['Age', 'Gender', 'Body Temperature', 'Cardiac Frequency',
                    'Respiratory Frequency', 'ABG: Oxygen Saturation (SaO2)',
                    #'Systolic Blood Pressure', 
                    'Activated Partial Thromboplastin Time (aPTT)', 'Blood Urea Nitrogen (BUN)',
                    'Alanine Aminotransferase (ALT)', 'Aspartate Aminotransferase (AST)',
                    'Blood Amylase', 'Blood Calcium', 'Blood Creatinine', 'Blood Sodium',
                    'C-Reactive Protein (CRP)', 'CBC: Hemoglobin', 'CBC: Leukocytes',
                    'CBC: Mean Corpuscular Volume (MCV)', 'CBC: Platelets',
                    'CBC: Red cell Distribution Width (RDW)', 'Cholinesterase',
                    'Glycemia', 'Potassium Blood Level',
                    'Prothrombin Time (INR)', 'Total Bilirubin']

SPANISH_ITALIAN_DATA = ['Age', 'Gender', 'Body Temperature', 
                        'ABG: Oxygen Saturation (SaO2)', 'Cardiac Frequency', 
                        # 'Systolic Blood Pressure', 'Essential hypertension',
                        'Alanine Aminotransferase (ALT)', 'Aspartate Aminotransferase (AST)', 
                        'Blood Creatinine', 'Blood Sodium', 'Blood Urea Nitrogen (BUN)', 
                        'Potassium Blood Level', 'CBC: Hemoglobin', 'CBC: Mean Corpuscular Volume (MCV)',  
                        'CBC: Platelets', 'CBC: Leukocytes', 'C-Reactive Protein (CRP)', 'Glycemia', 
                        'Prothrombin Time (INR)', 'Cardiac dysrhythmias', 'Chronic kidney disease', 
                        'Coronary atherosclerosis and other heart disease', 'Diabetes']


# Discharge codes
# 1,2,5,6,9 = discharged, 4 = deceased
DISCHARGE_CODES = [1, 2, 4, 5, 6, 9]
DISCHARGE_CODE_RELEASED = 4

DIAGNOSIS_COLUMNS = ['Dia1', 'Dia2', 'Dia3', 'Dia4', 'Dia5']
DEMOGRAPHICS_FEATURES = ['Gender', 'Age', 'Outcome']


RENAMED_LAB_COLUMNS = {
        'ALT: ALT': 'Alanine Aminotransferase (ALT)',
        'AST: AST': 'Aspartate Aminotransferase (AST)',
        'Creatinina UAR: CREATININA SANGUE': 'Blood Creatinine',
        'Potassio: POTASSIEMIA': 'Potassium Blood Level',
        'Proteina C Reattiva: PCR - PROTEINA C REATTIVA': 'C-Reactive Protein (CRP)',
        'Glucosio ematico: GLICEMIA': 'Glycemia',
        'Azoto ematico UAR: AZOTO UREICO EMATICO': 'Blood Urea Nitrogen (BUN)',
        'Emogasanalisi su sangue arterioso: ACIDO LATTICO': 'ABG: Lactic Acid',
        'Emogasanalisi su sangue arterioso: IONE BICARBONATO STD': 'ABG: standard bicarbonate (sHCO3)',
        'Emogasanalisi su sangue arterioso: ECCESSO DI BASI': 'ABG: Base Excess',
        'Emogasanalisi su sangue arterioso: PO2': 'ABG: PaO2',
        'Emogasanalisi su sangue arterioso: OSSIGENO SATURAZIONE': 'ABG: Oxygen Saturation (SaO2)',
        'Emogasanalisi su sangue arterioso: PCO2': 'ABG: PaCO2',
        'Emogasanalisi su sangue arterioso: PH EMATICO': 'ABG: pH',
        'Emogasanalisi su sangue arterioso: CARBOSSIEMOGLOBINA': 'ABG: COHb',
        'Emogasanalisi su sangue arterioso: METAEMOGLOBINA': 'ABG: MetHb',
        'Sodio: SODIEMIA': 'Blood Sodium',
        'TEMPO DI PROTROMBINA UAR: TEMPO DI PROTROMBINA RATIO': 'Prothrombin Time (INR)',
        'TEMPO DI TROMBOPLASTINA PARZIALE: TEMPO DI TROMBOPLASTINA PARZIALE ATTIVATO': 'Activated Partial Thromboplastin Time (aPTT)',
        'Calcemia: CALCEMIA': 'Blood Calcium',
        'BILIRUBINA TOTALE REFLEX: BILIRUBINA TOTALE': 'Total Bilirubin',
        'Amilasi: AMILASI NEL SIERO' : 'Blood Amylase',
        'Colinesterasi: COLINESTERASI': 'Cholinesterase',
        'Emocromocitometrico (Urgenze): VOLUME CORPUSCOLARE MEDIO': 'CBC: Mean Corpuscular Volume (MCV)',
        'Emocromocitometrico (Urgenze): PIASTRINE': 'CBC: Platelets',
        'Emocromocitometrico (Urgenze): VALORE DISTRIBUTIVO GLOBULI ROSSI': 'CBC: Red cell Distribution Width (RDW)',
        'Emocromocitometrico (Urgenze): LEUCOCITI': 'CBC: Leukocytes',
        'Emocromocitometrico (Urgenze): EMOGLOBINA': 'CBC: Hemoglobin',
        }

VITAL_SIGNS = ['SaO2',
               'P. Max',
               #  'P. Min',  # Keep only max because it is more precise
               'F. Card.',
               'F. Resp.',
               'Temp.',
               'Dolore',
               'GCS',
               'STICKGLI']

RENAMED_VITALS_COLUMNS = {
        "P. Max": "Systolic Blood Pressure",
        #  "P. Min": "Diastolic Blood Pressure",
        "F. Card.": "Cardiac Frequency",
        "Temp.": "Body Temperature",
        "F. Resp.": "Respiratory Frequency"
        }


LAB_FEATURES_NOT_CONTAIN = ['NOTA',  # Remove notes
                            'AFRO',  # No normalized creatinine
                            'CAUCAS',  # No normalized creatinine
                            'UREA EMATICA'  # We keep BUN directly
                           ]
LAB_FEATURES_NOT_MATCH = ['IONE BICARBONATO',  # We keep standard directly
                          '(PT) TEMPO DI PROTROMBINA',  # We keep only Prothrombin Time
                          'HCT',  # Remove Hematocrit to keep Hemoglobin
                          'EMATOCRITO', # Remove Hematocrit to keep Hemoglobin
                          'ERITROCITI',   # Redundant with Hemoglobin
                          'BE(ECF)',  # Remove Base Excess ECF (Keep normal one BE)
                          'CTCO2',  # Redundant with PaCO2
                          'FHHB',  # Redundant with Hemoglobin (also with Hematocrit)
                          'FO2HB',  # Redundant with Hemoglobin
                          'CALCIO IONIZZATO',  # Redundant with Blood Calcium
                          'CONCENTRAZIONE HB MEDIA',  # Redundant with MCV
                          'CONTENUTO HB MEDIO',  # Redundant with MCV
                          'CLORUREMIA',  # Redundant with Sodium
                          ]

COLS_TREATMENTS = ['HOSPITAL', 'COUNTRY', 'DT_HOSPITAL_ADMISSION', 'GENDER',
                    'RACE', 'PREGNANT', 'AGE', 'DIABETES', 'HYPERTENSION',
                    'DISLIPIDEMIA', 'OBESITY', 'SMOKING', 'RENALINSUF',
                    'ANYLUNGDISEASE', 'AF', 'VIH', 'ANYHEARTDISEASE',
                    'MAINHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE',
                    'LIVER_DISEASE', 'CANCER', 'HOME_OXIGEN_THERAPY', 'IN_PREVIOUSASPIRIN',
                    'IN_OTHERANTIPLATELET', 'IN_ORALANTICOAGL', 'IN_ACEI_ARB', 'IN_BETABLOCKERS',
                    'IN_BETAGONISTINHALED', 'IN_GLUCORTICOIDSINHALED','IN_DVITAMINSUPLEMENT',
                    'IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT', 'FAST_BREATHING', 'MAXTEMPERATURE_ADMISSION',
                    'SAT02_BELOW92', 'DDDIMER_B', 'PROCALCITONIN_B', 'PCR_B', 'TRANSAMINASES_B', 'LDL_B',
                    'BLOOD_PRESSURE_ABNORMAL_B', 'CREATININE', 'SODIUM', 'LEUCOCYTES', 'LYMPHOCYTES',
                    'HEMOGLOBIN', 'PLATELETS', 'GLASGOW_COMA_SCORE', 'CHESTXRAY_BNORMALITY',
                    'CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS','ACEI_ARBS',
                    'ONSET_DATE_DIFF', 'TEST_DATE_DIFF', 'CLOROQUINE', 'ANTIVIRAL','ANTICOAGULANTS',
                    'REGIMEN', 'DEATH', 'COMORB_DEATH']

# This is the list of HCUP used for the mortality paper
COVID_MORTALITY_PAPER_HCUP_LIST = [49,50,87,90,95,146]

DIABETES = [49, 50, 174]
HYPERTENSION = [87, 88, 171]
DISLIPIDEMIA = [53]
OBESITY = [58]
RENALINSUF = [146]
ANYLUNGDISEASE = [116, 117, 121, 122]
AF = [95]
VIH = [5]
ANYHEARTDISEASE = [90, 92, 93, 95]
ANYCEREBROVASCULARDISEASE = [98, 100, 101, 102]
CONECTIVEDISEASE = [198, 199]
LIVER_DISEASE = [6, 139]
CANCER = [11, 12, 13, 14, 15, 16, 17, 18, 19, 
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
        40, 41, 42, 43]
# HCUP_LIST FOR THE TREATMENTS PAPER    
COMORBS_TREATMENTS_NAMES = ['DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY', 'RENALINSUF',
            'ANYLUNGDISEASE', 'AF', 'VIH', 'ANYHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE',
            'CONECTIVEDISEASE', 'LIVER_DISEASE', 'CANCER']

COMORBS_TREATMENTS_HCUP = [DIABETES, HYPERTENSION, DISLIPIDEMIA, OBESITY, RENALINSUF,
            ANYLUNGDISEASE, AF, VIH, ANYHEARTDISEASE, ANYCEREBROVASCULARDISEASE,
            CONECTIVEDISEASE, LIVER_DISEASE, CANCER]


HCUP_LIST = list(set(DIABETES + HYPERTENSION + DISLIPIDEMIA + OBESITY + RENALINSUF + \
            ANYLUNGDISEASE + AF + VIH + ANYHEARTDISEASE + ANYCEREBROVASCULARDISEASE + \
            CONECTIVEDISEASE + LIVER_DISEASE + CANCER))

# HOPE TREATMENTS

IN_TREATMENTS_NAME = ['HOME_OXIGEN_THERAPY', 'IN_PREVIOUSASPIRIN', 'IN_OTHERANTIPLATELET',
                'IN_ORALANTICOAGL', 'IN_ACEI_ARB', 'IN_BETABLOCKERS', 'IN_BETAGONISTINHALED',
                'IN_GLUCORTICOIDSINHALED', 'IN_DVITAMINSUPLEMENT', 'IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT']

HOME_OXIGEN_THERAPY = 'V03AN01'
IN_PREVIOUSASPIRIN = 'N02BA01'
IN_OTHERANTIPLATELET = 'B01AC'
IN_ORALANTICOAGL = 'B01'
IN_ACEI_ARB = 'C09'
IN_BETABLOCKERS = 'C07A'
IN_BETAGONISTINHALED = 'R03AC'
IN_GLUCORTICOIDSINHALED = 'R03BA'
IN_DVITAMINSUPLEMENT = 'A11'
IN_BENZODIACEPINES = 'N05'
IN_ANTIDEPRESSANT = 'N06A'

IN_TREATMENTS = [HOME_OXIGEN_THERAPY, IN_PREVIOUSASPIRIN, IN_OTHERANTIPLATELET,
                IN_ORALANTICOAGL, IN_ACEI_ARB, IN_BETABLOCKERS, IN_BETAGONISTINHALED,
                IN_GLUCORTICOIDSINHALED, IN_DVITAMINSUPLEMENT, IN_BENZODIACEPINES, IN_ANTIDEPRESSANT]

VITALS_TREAT = ['Respiratory Frequency', 'Body Temperature', 'Systolic Blood Pressure']
LABS_TREAT = ['ABG: Oxygen Saturation (SaO2)', 'Azoto ematico UAR: D-DIMERO', 'PROCALCITONINA: PROCALCITONINA',
              'C-Reactive Protein (CRP)', 'Alanine Aminotransferase (ALT)', 'LATTICODEIDROGENASI: (LDH) LATTICODEIDROGENASI',
              'Blood Creatinine', 'Blood Sodium', 'CBC: Leukocytes', 'Emocromo + formula: LINFOCITI (N)', 'CBC: Hemoglobin', 'CBC: Platelets']

VITALS_TREAT_RENAME = {'Respiratory Frequency': 'FAST_BREATHING', 
                        'Body Temperature': 'MAXTEMPERATURE_ADMISSION', 
                        'Systolic Blood Pressure': 'BLOOD_PRESSURE_ABNORMAL_B'}

LABS_TREAT_RENAME = {'ABG: Oxygen Saturation (SaO2)': 'SAT02_BELOW92', 
                    'Azoto ematico UAR: D-DIMERO': 'DDDIMER_B', 
                    'PROCALCITONINA: PROCALCITONINA': 'PROCALCITONIN_B',
                    'C-Reactive Protein (CRP)': 'PCR_B', 
                    'Alanine Aminotransferase (ALT)': 'TRANSAMINASES_B', 
                    'LATTICODEIDROGENASI: (LDH) LATTICODEIDROGENASI': 'LDL_B',
                    'Blood Creatinine': 'CREATININE', 
                    'Blood Sodium': 'SODIUM', 
                    'CBC: Leukocytes': 'LEUCOCYTES', 
                    'Emocromo + formula: LINFOCITI (N)': 'LYMPHOCYTES', 
                    'CBC: Hemoglobin': 'HEMOGLOBIN', 
                    'CBC: Platelets': 'PLATELETS'}

# TREATMENTS
TREATMENTS_NAME = ['CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS', 'ACEI_ARBS', 'CLOROQUINE', 'ANTIVIRAL', 'ANTICOAGULANTS']
CORTICOSTEROIDS = 'H02'
INTERFERONOR = 'L03'
TOCILIZUMAB = 'L04AC07'
ANTIBIOTICS = 'J01'
ACEI_ARBS = 'C09'
CLOROQUINE = 'P01BA02'
ANTIVIRAL = 'J05AR'
ANTICOAGULANTS = 'B01AB'

TREATMENTS = [CORTICOSTEROIDS, INTERFERONOR, TOCILIZUMAB, ANTIBIOTICS, ACEI_ARBS, CLOROQUINE, ANTIVIRAL, ANTICOAGULANTS]

# HCUP for COMORB_DEATH columns. SEPSIS = 2; Acute Renal Failure: 145; Heart Failure: 97; Embolic Event: 105
COMORB_DEATH = [2, 145, 97, 105]

def clean_lab_features(lab_feat):
    features = [x for x in lab_feat
                if all(s not in x for s in LAB_FEATURES_NOT_CONTAIN) and
                all(s != x for s in LAB_FEATURES_NOT_MATCH)]
    return features


def export_comorbidities(df, file_name):
    #  Convert (to export for R processing)
    # TODO: Improve this code
    comorb_df = pd.DataFrame(columns=['id', 'comorb'])
    for i in range(len(df)):
        d_temp = df.iloc[i]
        df_temp = pd.DataFrame({'id': [d_temp['NumeroScheda']] * 6,
                                'comorb': [d_temp['Principale']] + \
                                    [d_temp[d] for d in DIAGNOSIS_COLUMNS]})
        comorb_df = comorb_df.append(df_temp)

    comorb_df = comorb_df.dropna().reset_index()
    comorb_df.to_csv(file_name)


def comorbidities_long(df):
    #  Convert (to export for R processing)
    # TODO: Improve this code
    comorb_df = pd.DataFrame(columns=['id', 'comorb'])
    for i in range(len(df)):
        d_temp = df.iloc[i]
        df_temp = pd.DataFrame({'id': [d_temp['NumeroScheda']] * 6,
                                'comorb': [d_temp['Principale']] + \
                                    [d_temp[d] for d in DIAGNOSIS_COLUMNS]})
        comorb_df = comorb_df.append(df_temp)

    comorb_df = comorb_df.dropna().reset_index()
    return comorb_df


def get_lab_dates(t):
    # TODO: Find better way to do so. Nested try-except is not nice.
    try:
        date = datetime.datetime.strptime(t, '%d/%m/%Y %H:%M')
    except ValueError:
        try:
            date = datetime.datetime.strptime(t, '%d/%m/%Y')
        except ValueError:
            try:
                date = datetime.datetime.strptime(t, '%d/%m/%y %H:%M')
            except ValueError:
                date = datetime.datetime.strptime(t, '%d/%m/%y')

    return date

def get_age(t):

    try:
        today = pd.Timestamp(year=2020, month=4, day=1)
        age = np.round((today - t).days/365)
        return age
    except:
        return np.NaN


def cleanup_demographics(demographics):

    demographics = demographics[['N_SCHEDA_PS', 'PZ_SESSO_PS', "PZ_DATA_NASCITA_PS"]]
    try:
        demographics['PZ_DATA_NASCITA_PS'] = \
            pd.to_datetime(demographics['PZ_DATA_NASCITA_PS'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        demographics.loc[:, 'PZ_DATA_NASCITA_PS'] = \
            pd.to_datetime(demographics['PZ_DATA_NASCITA_PS'], format='%m/%d/%Y')

    demographics.loc[:, 'Age'] = demographics['PZ_DATA_NASCITA_PS'].apply(get_age)
    demographics = demographics.drop('PZ_DATA_NASCITA_PS', axis = 1)
    demographics = demographics.rename(columns = {'N_SCHEDA_PS' : 'NOSOLOGICO', 'PZ_SESSO_PS' : 'Gender'})
    demographics['Gender'] = (demographics['Gender'] == 'F').astype(int)
    demographics['NOSOLOGICO'] = demographics['NOSOLOGICO'].astype(str)

    return demographics


def create_vitals_dataset(vitals, patients, lab_tests=True):
    vital_signs = VITAL_SIGNS.copy()
    if lab_tests:
        vital_signs.remove('SaO2')  # Remove oxygen saturation if we have lab values (it is there)

    # Cleanup commas in numbers
    vitals.loc[:, 'VALORE_PARAMETRO'] = \
            vitals.loc[:, 'VALORE_PARAMETRO'].apply(lambda x: x.replace(",", "."))

    dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=patients)
    for p in patients:
        vitals_p = vitals[vitals['NOSOLOGICO'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
        for vital_name in vital_signs:
            # Take mean if multiple values
            vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
            vital_value = pd.to_numeric(vital_value).mean()
            dataset_vitals.loc[p, vital_name] = vital_value

    #dataset_vitals['Temp.'] = fahrenheit_covert(dataset_vitals['Temp.'])

    # Adjust missing columns
    dataset_vitals = remove_missing(dataset_vitals, nan_threshold=100)

    # Rename to English
    dataset_vitals = dataset_vitals.rename(columns=RENAMED_VITALS_COLUMNS)

    return dataset_vitals



def create_lab_dataset(lab, patients):
    # Remove missing test (groups) with more than 40% nonzeros
    lab_tests = lab['COD_INTERNO_PRESTAZIONE'].unique().tolist()
    dataset_lab_tests = pd.DataFrame(False, columns=lab_tests, index=patients)

    #Unstack the dataset and transform the entries in True/False
    dataset_lab_tests = lab[['NOSOLOGICO', 'COD_INTERNO_PRESTAZIONE', 'VALORE_TESTO']].groupby(['NOSOLOGICO', 'COD_INTERNO_PRESTAZIONE']).count().unstack().notna()
    dataset_lab_tests.columns = [i[1] for i in dataset_lab_tests.columns] # because of groupby, the columns are a tuple

    # 30% removes tests that are not present and the COVID-19 lab test
    lab_tests_reduced = remove_missing(dataset_lab_tests, missing_type=False, nan_threshold=100, impute=False)

    # Filter data entries per test
    lab_reduced = lab[lab['COD_INTERNO_PRESTAZIONE'].isin(lab_tests_reduced.columns)]

    # Create lab features for each exam
    dataset_lab = {}
    for lab_test in lab_tests_reduced.columns:
        # Create dataset
        lab_test_temp = lab_reduced.loc[lab_reduced['COD_INTERNO_PRESTAZIONE'] == lab_test]
        lab_test_features = lab_test_temp['PRESTAZIONE'].unique().tolist()

        # Remove unnecessary features
        lab_test_features = clean_lab_features(lab_test_features)

        # Add name of lab_test
        test_name = lab[lab['COD_INTERNO_PRESTAZIONE'] == lab_test]['DESCR_PRESTAZIONE'].values[0]
        lab_test_features_names = [test_name.strip() + ": " + x for x in lab_test_features]

        dataset_lab_test = pd.DataFrame(np.nan, columns=lab_test_features_names, index=patients)
        for p in patients:
            lab_p = lab_test_temp[lab_test_temp['NOSOLOGICO'] == p][['COD_INTERNO_PRESTAZIONE', 'DATA_RICHIESTA', 'PRESTAZIONE', 'VALORE']]
            for lab_name in lab_test_features:
                if any(lab_p['PRESTAZIONE'] == lab_name):
                    lab_p_name = lab_p[lab_p['PRESTAZIONE'] == lab_name]
                    idx = lab_p_name['DATA_RICHIESTA'].idxmin()  # Pick first date of test if multiple
                    dataset_lab_test.loc[p, test_name.strip() + ": " + lab_name] = lab_p_name.loc[idx]['VALORE']
        dataset_lab[lab_test] = dataset_lab_test

    # Create full dataset
    dataset_lab_full = pd.concat([v for _,v in dataset_lab.items()],
                                 axis=1, sort=True).astype(np.float64)
    dataset_lab_full = remove_missing(dataset_lab_full, nan_threshold=100)


    # Rename dataset laboratory
    dataset_lab_full = dataset_lab_full.rename(columns=RENAMED_LAB_COLUMNS)

    return dataset_lab_full

def create_dataset_comorbidities(comorb_long, icd_category, patients):

    #Load the diagnoses dict
    if icd_category == 9:
        icd_dict = pd.read_csv('../../../analyzer/hcup_dictionary_icd9.csv')
    else:
        icd_dict = pd.read_csv('../../../analyzer/hcup_dictionary_icd10.csv')

    #The codes that are not mapped are mostly procedure codes or codes that are not of interest
    icd_descr = pd.merge(comorb_long, icd_dict, how='inner', left_on=['DIAGNOSIS_CODE'], right_on=['DIAGNOSIS_CODE'])

    #Create a list with the categories that we want
    comorb_descr = icd_descr.loc[icd_descr['HCUP_ORDER'].isin(HCUP_LIST)]

    #Limit only to the HCUP Description and drop the duplicates
    comorb_descr = comorb_descr[['NOSOLOGICO','GROUP_HCUP']].drop_duplicates()

    #Convert from long to wide format
    comorb_descr = pd.get_dummies(comorb_descr, columns=['GROUP_HCUP'], prefix=['GROUP_HCUP'])

    #Now we will remove the GROUP_HCUP_ from the name of each column
    comorb_descr = comorb_descr.rename(columns = lambda x: x.replace('GROUP_HCUP_', ''))

    #Let's combine the diabetes columns to one
    comorb_descr['Diabetes'] = comorb_descr[["Diabetes mellitus with complications", "Diabetes mellitus without complication"]].max(axis=1)

    #Drop the other two columns
    comorb_descr = comorb_descr.drop(columns=['Diabetes mellitus with complications', 'Diabetes mellitus without complication'])

    dataset_comorbidities = pd.DataFrame(comorb_descr.groupby(['NOSOLOGICO'], as_index=False).max())

    df_patients = pd.DataFrame(patients, columns = ['NOSOLOGICO'])
    dataset_comorbidities =  pd.merge(df_patients, dataset_comorbidities, how='left',
        left_on=['NOSOLOGICO'], right_on = ['NOSOLOGICO'])
    dataset_comorbidities = dataset_comorbidities.fillna(0)
    
    return dataset_comorbidities

def create_dataset_discharge(discharge, patients, icu=None):

    dataset_discharge = pd.DataFrame(columns=DEMOGRAPHICS_FEATURES, index=patients)
    dataset_discharge.loc[:, DEMOGRAPHICS_FEATURES] = discharge[['NOSOLOGICO'] + DEMOGRAPHICS_FEATURES].set_index('NOSOLOGICO')
    #dataset_discharge.loc[:, 'Gender'] = dataset_discharge.loc[:, 'Gender'].astype('category')
    #dataset_discharge.Gender = dataset_discharge.Gender.cat.codes.astype('category')
    dataset_discharge = dataset_discharge[['Outcome']]
    dataset_discharge.loc[:, 'Outcome'] = dataset_discharge.loc[:, 'Outcome'].astype('category')

    if icu is not None:
        dataset_discharge = dataset_discharge.join(icu.set_index('NOSOLOGICO'))


    return dataset_discharge


def cleanup_discharge_info(discharge_info):

    covid_patients = discharge_info['Principale'].isin(LIST_DIAGNOSIS)

    for d in DIAGNOSIS_COLUMNS:
        covid_patients = covid_patients | discharge_info[d].isin(LIST_DIAGNOSIS)

    discharge_info = discharge_info[covid_patients]

    # Keep discharge codes and transform the dependent variable to binary
    discharge_info = discharge_info[discharge_info['Modalità di dimissione'].isin(DISCHARGE_CODES)]
    discharge_info['Modalità di dimissione'] = \
        (discharge_info['Modalità di dimissione'] == DISCHARGE_CODE_RELEASED).apply(int) #transform to binary

    # Drop Duplicated Observations
    discharge_info.drop_duplicates(['NumeroScheda', 'Modalità di dimissione'],
                                    inplace=True)
    discharge_info.drop_duplicates(['NumeroScheda'], inplace=True)

    #Keep only important columns and rename them
    discharge_info = discharge_info[['NumeroScheda', 'Sesso', 'Età', 'Modalità di dimissione']]
    discharge_info = discharge_info.rename(
            columns={'NumeroScheda': 'NOSOLOGICO',
                     'Sesso': 'Gender',
                     'Età':'Age',
                     'Modalità di dimissione':'Outcome'})
    discharge_info.NOSOLOGICO = discharge_info.NOSOLOGICO.apply(str)

    return discharge_info


def fahrenheit_covert(temp_celsius):
    temp_fahrenheit = ((temp_celsius * 9)/5)+ 32
    return temp_fahrenheit

def filter_patients(datasets):

    patients = datasets[0]['NOSOLOGICO'].astype(np.int64)

    # Get common patients
    for d in datasets[1:]:
        patients = d[d['NOSOLOGICO'].astype(np.int64).isin(patients)]['NOSOLOGICO'].unique()


    # Remove values not in patients (in place)
    for d in datasets:
        d.drop(d[~d['NOSOLOGICO'].astype(np.int64).isin(patients)].index, inplace=True)

    return patients


def get_swabs(lab):

    covid = lab[lab.COD_INTERNO_PRESTAZIONE == 'COV19']
    covid = covid[covid.VALORE_TESTO.isin(['POSITIVO', 'Negativo', 'Debolmente positivo'])]
    covid.VALORE_TESTO = covid.VALORE_TESTO.isin(['POSITIVO','Debolmente positivo']).astype(int).astype('category')
    covid = covid[~ covid.NOSOLOGICO.duplicated()] # drop duplicated values
    swab = covid[['NOSOLOGICO', 'VALORE_TESTO']]
    swab = swab.rename(columns = {'VALORE_TESTO': 'Swab'})
    swab['Swab'] = swab['Swab'].astype('int')

    return swab

def get_regimen(cloroquine, antiviral, anticoagulant):
    if cloroquine == 0:
        return 'Non-Chloroquine'

    elif cloroquine == 1 and antiviral == 1 and anticoagulant == 1:
        return 'All'
    
    elif cloroquine == 1 and antiviral == 1 and anticoagulant == 0:
        return 'Chloroquine and Antivirals'

    elif cloroquine == 1 and antiviral == 0 and anticoagulant == 1:
        return 'Chloroquine and Anticoagulants'

    elif cloroquine == 1 and antiviral == 0 and anticoagulant == 0:
        return 'Chloroquine Only'
    