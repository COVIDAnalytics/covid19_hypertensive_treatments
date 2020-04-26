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
                             "Diabetes mellitus with complications", 
                             "Influenza", 
                             "Acute and unspecified renal failure"]

# Discharge codes
# 1,2,5,6,9 = discharged, 4 = deceased
DISCHARGE_CODES = [1, 2, 4, 5, 6, 9]
DISCHARGE_CODE_RELEASED = 4

DIAGNOSIS_COLUMNS = ['Dia1', 'Dia2', 'Dia3', 'Dia4', 'Dia5']

ANAGRAPHICS_FEATURES = ['Sex', 'Age', 'Outcome']


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

def get_age(t):

    try:
        today = pd.Timestamp(year=2020, month=4, day=1)
        age = np.round((today - t).days/365)
        return age
    except:
        return np.NaN

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

def cleanup_anagraphics(anagraphics):

    anagraphics = anagraphics[['N_SCHEDA_PS', 'PZ_SESSO_PS', "PZ_DATA_NASCITA_PS"]]
    anagraphics['PZ_DATA_NASCITA_PS'] = pd.to_datetime(anagraphics['PZ_DATA_NASCITA_PS'], format='%Y-%m-%d %H:%M:%S')
    anagraphics['Age'] = anagraphics['PZ_DATA_NASCITA_PS'].apply(get_age)
    anagraphics = anagraphics.drop('PZ_DATA_NASCITA_PS', axis = 1)
    anagraphics = anagraphics.rename(columns = {'N_SCHEDA_PS' : 'NOSOLOGICO', 'PZ_SESSO_PS' : 'Sex'})
    anagraphics['Sex'] = (anagraphics['Sex'] == 'F').astype(int)
    anagraphics['NOSOLOGICO'] = anagraphics['NOSOLOGICO'].astype(str)    

    return anagraphics


def create_vitals_dataset(vitals, patients, lab_tests=True):
    vital_signs = VITAL_SIGNS
    if lab_tests:
        vital_signs.remove('SaO2')  # Remove oxygen saturation if we have lab values (it is there)

    dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=patients)
    for p in patients:
        vitals_p = vitals[vitals['NOSOLOGICO'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
        for vital_name in vital_signs:
            # Take mean if multiple values
            vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
            vital_value = pd.to_numeric(vital_value).mean()
            dataset_vitals.loc[p, vital_name] = vital_value

    # Adjust missing columns
    dataset_vitals = remove_missing(dataset_vitals)

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
    lab_tests_reduced = remove_missing(dataset_lab_tests, missing_type=False, nan_threashold=30, impute=False)

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
    dataset_lab_full = remove_missing(dataset_lab_full)


    # Rename dataset laboratory
    dataset_lab_full = dataset_lab_full.rename(columns=RENAMED_LAB_COLUMNS)

    return dataset_lab_full

def create_dataset_comorbidities(comorbidities, patients):

    # False and True are transformed to 0 and 1 categories
    dataset_comorbidities = comorbidities.astype('int').astype('category')

    #Join the two categories of Diabetes
    dataset_comorbidities["Diabetes"] = np.zeros(len(dataset_comorbidities))
    dataset_comorbidities["Diabetes"] = ((dataset_comorbidities['Diabetes mellitus without complication'].astype(int) > 0) | \
        (dataset_comorbidities['Diabetes mellitus with complications'].astype(int) > 0)).astype(int).astype('category')
    #  dataset_comorbidities.index = [str(i) for i in comorbidities.index]

    # Keep only the comorbidities that appear more than 10 times and remove pneumonia ones
    cols_keep = list(dataset_comorbidities.columns[dataset_comorbidities.sum() >10])

    for e in LIST_REMOVE_COMORBIDITIES:
        cols_keep.remove(e)
    dataset_comorbidities = dataset_comorbidities[cols_keep]
    
    dataset_comorbidities['NOSOLOGICO'] = dataset_comorbidities['NOSOLOGICO'].apply(int).apply(str)


    return dataset_comorbidities.set_index('NOSOLOGICO')


def create_dataset_discharge(anagraphics, patients, icu=None):

    dataset_anagraphics = pd.DataFrame(columns=ANAGRAPHICS_FEATURES, index=patients)
    dataset_anagraphics.loc[:, ANAGRAPHICS_FEATURES] = anagraphics[['NOSOLOGICO'] + ANAGRAPHICS_FEATURES].set_index('NOSOLOGICO')
    dataset_anagraphics.loc[:, 'Sex'] = dataset_anagraphics.loc[:, 'Sex'].astype('category')
    dataset_anagraphics.Sex = dataset_anagraphics.Sex.cat.codes.astype('category')
    dataset_anagraphics.loc[:, 'Outcome'] = dataset_anagraphics.loc[:, 'Outcome'].astype('category')

    if icu is not None:
        dataset_anagraphics = dataset_anagraphics.join(icu.set_index('NOSOLOGICO'))


    return dataset_anagraphics


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
                     'Sesso': 'Sex',
                     'Età':'Age',
                     'Modalità di dimissione':'Outcome'})
    discharge_info.NOSOLOGICO = discharge_info.NOSOLOGICO.apply(str)

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


def get_swabs(lab):

    covid = lab[lab.COD_INTERNO_PRESTAZIONE == 'COV19']
    covid = covid[covid.VALORE_TESTO.isin(['POSITIVO', 'Negativo', 'Debolmente positivo'])]
    covid.VALORE_TESTO = covid.VALORE_TESTO.isin(['POSITIVO','Debolmente positivo']).astype(int).astype('category')
    covid = covid[~ covid.NOSOLOGICO.duplicated()] # drop duplicated values
    swab = covid[['NOSOLOGICO', 'VALORE_TESTO']]
    swab = swab.rename(columns = {'VALORE_TESTO': 'Swab'})
    swab['Swab'] = swab['Swab'].astype('int')

    return swab

