import numpy as np
import pandas as pd
import datetime

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


# ICD9 COVID diagnosis Italian codes
LIST_DIAGNOSIS = ['4808', '4803', 'V0182', '7982']
LIST_REMOVE_COMORBIDITIES = ["Immunizations and screening for infectious disease",
                             "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
                             "Respiratory failure; insufficiency; arrest (adult)",
                             "Residual codes; unclassified",
                             "Diabetes mellitus without complication",
                             "Diabetes mellitus with complications"]

# Discharge codes
# 1,2,5,6,9 = discharged, 4 = deceased
DISCHARGE_CODES = [1, 2, 4, 5, 6, 9]
DISCHARGE_CODE_RELEASED = 4

DIAGNOSIS_COLUMNS = ['Dia1', 'Dia2', 'Dia3', 'Dia4', 'Dia5']


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
        "Temp.": "Temperature Celsius",
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


def get_lab_dates(t):
    try:
        date = datetime.datetime.strptime(t, '%d/%m/%Y %H:%M')
    except ValueError:
        date = datetime.datetime.strptime(t, '%d/%m/%Y')

    return date


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
