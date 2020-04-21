import pandas as pd
import datetime
import numpy as np
import pickle

# from analyzer.icd9.icd9 import ICD9

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer


def export_comorbidities(df, file_name):
    #  Convert (to export for R processing)
    # TODO: Improve this code
    comorb_df = pd.DataFrame(columns=['id', 'comorb'])
    for i in range(len(df)):
        d_temp = df.iloc[i]
        df_temp = pd.DataFrame({'id': [d_temp['NumeroScheda']] * 6,
                                'comorb': [d_temp['Principale'],
                                           d_temp['Dia1'],
                                           d_temp['Dia2'],
                                           d_temp['Dia3'],
                                           d_temp['Dia4'],
                                           d_temp['Dia5']]})
        comorb_df = comorb_df.append(df_temp)

    comorb_df = comorb_df.dropna().reset_index()
    comorb_df.to_csv(file_name)


def fix_outcome(outcome):
    if 'DIMESSO' in outcome:
        return 'DIMESSO'
    elif outcome == 'DECEDUTO':
        return outcome
    else:
        raise ValueError('Not recognized')


def n_days(t):
    if isinstance(t, str):
        date = datetime.datetime.strptime(t, '%m/%d/%y')
        new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
        day_of_the_year = (date - new_year_day).days + 1
        return day_of_the_year
    else:
        return np.NaN


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

def clean_lab_features(lab_feat):
    features = [x for x in lab_feat
                if ('NOTA' not in x) and  # Remove notes
                ('AFRO' not in x) and    # No normalized creatinine
                ('CAUCAS' not in x) and  # No normalized creatinine
                ('UREA EMATICA' not in x) and  # We keep BUN directly
                ('IONE BICARBONATO' != x) and  # We keep standard directly
                ('TEMPO DI PROTROMBINA RATIO' != x) # We keep only Prothrombin Time
    ]

    return features

def drop_anagraphics_duplicates(anagraphics):
    # Drop duplicates
    n_orig = len(anagraphics)
    anagraphics.drop_duplicates(['CODICE FISCALE', "ESITO"], inplace=True)
    n_unique = len(anagraphics)
    print("Removed %d duplicates with same (ID, outcome)" % (n_orig - n_unique))

    # Drop patients with multiple diagnosis
    n_orig = len(anagraphics)
    anagraphics.drop_duplicates(['CODICE FISCALE'], inplace=True)
    n_unique = len(anagraphics)
    print("Removed %d duplicates with same (ID)" % (n_orig - n_unique))

    # Drop patients with multiple diagnosis
    n_orig = len(anagraphics)
    anagraphics.drop_duplicates(['NOSOLOGICO'], inplace=True)
    n_unique = len(anagraphics)
    print("Removed %d duplicates with same (Nosologico)" % (n_orig - n_unique))


def add_codice_fiscale(df, anagraphics):
    df['CODICE FISCALE'] = ""
    patients_codice_fiscale = anagraphics['CODICE FISCALE'].unique().tolist()
    for p in patients_codice_fiscale:
        nosologico = anagraphics[anagraphics['CODICE FISCALE'] == p]['NOSOLOGICO'].values
        vitals.loc[vitals['SCHEDA_PS'] == nosologico[0], 'CODICE FISCALE'] = p


def load_cremona(path, lab_tests=True):

    # TODO:
    # 1. Get general data discharge -> Extract diagnosis
    #    (Use R package icd) => TODO!
    # 2. Merge with vitals from ER
    # 3. Merge with lab tests from ER
    # 4. Add ICU admissions (later)

    # Parameters

    # ICD9 COVID diagnosis Italian codes
    list_diagnosis = ['4808', '4803', 'V0182', '7982']
    list_remove_comorbidities = ["Immunizations and screening for infectious disease",
                                 "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
                                 "Respiratory failure; insufficiency; arrest (adult)",
                                 "Residual codes; unclassified"]

    # Discharge codes
    # 1,2,5,6,9 = discharged, 4 = deceased
    discharge_codes = [1, 2, 4, 5, 6, 9]
    discharge_code_deceased = 4


    # Load data
    #-------------------------------------------------------------------------------------
    discharge_info = pd.read_csv('%s/general/discharge_info.csv' % path)
    covid_patients = discharge_info['Principale'].isin(list_diagnosis) | \
            discharge_info['Dia1'].isin(list_diagnosis) | \
            discharge_info['Dia2'].isin(list_diagnosis) | \
            discharge_info['Dia3'].isin(list_diagnosis) | \
            discharge_info['Dia4'].isin(list_diagnosis) | \
            discharge_info['Dia5'].isin(list_diagnosis)

    discharge_info = discharge_info[covid_patients]

    diagnosis = pd.concat([discharge_info['Principale'],
        discharge_info['Dia1'],
        discharge_info['Dia2'],
        discharge_info['Dia3'],
        discharge_info['Dia4'],
        discharge_info['Dia5']
        ]).dropna().str.strip().unique()


    # Export comorbidities to process with R
    export_comorbidities(discharge_info, '%s/general/nosologico_com.csv' % path)

    # TODO: Run R command automatically
    # import subprocess
    # subprocess.call(['R', '%s/general/R_script.R' % path])

    #Import the comorbidity csv and dictionary from R
    dataset_comorbidities = pd.read_csv('%s/general/comorbidities.csv' % path, index_col="id")
    dataset_comorbidities.drop(dataset_comorbidities.columns[0], axis = 1, inplace = True)
    with open('%s/general/dict_ccs.pkl' % path, "rb") as f:
        dict_ccs = pickle.load(f)

    #Change the name of the cols from CCS code to comorbidity name
    old_cols = list(dataset_comorbidities.columns)
    new_cols = [dict_ccs[i] for i in old_cols]
    dataset_comorbidities.columns = new_cols

    # Keep only the comorbidities that appear more than 10 times and remove pneumonia ones
    cols_keep = list(dataset_comorbidities.columns[dataset_comorbidities.sum() >10])
    for e in list_remove_comorbidities:
        cols_keep.remove(e)
    dataset_comorbidities = dataset_comorbidities[cols_keep]
    # False and True are transformed to 0 and 1 categories
    dataset_comorbidities = dataset_comorbidities.astype('int').astype('category')

    # Keep only the patients that we have in the comorbidities dataframe
    pat_comorb = dataset_comorbidities.index
    discharge_info = discharge_info[discharge_info.NumeroScheda.isin(pat_comorb)]

    # Keep discharge codes and transform the dependent variable to binary
    discharge_info = discharge_info[discharge_info['Modalità di dimissione'].isin(discharge_codes)]
    discharge_info['Modalità di dimissione'] = \
        (discharge_info['Modalità di dimissione'] == discharge_code_deceased).apply(int) #transform to binary

    # Drop Duplicated Observations
    discharge_info.drop_duplicates(['NumeroScheda', 'Modalità di dimissione'], inplace=True)
    discharge_info.drop_duplicates(['NumeroScheda'], inplace=True)

    #Keep only important columns and rename them
    discharge_info = discharge_info[['NumeroScheda', 'Sesso', 'Età', 'Modalità di dimissione']]
    discharge_info = discharge_info.rename(
            columns={'NumeroScheda': 'NOSOLOGICO',
                     'Sesso': 'Sex',
                     'Età':'Age',
                     'Modalità di dimissione':'Outcome'})
    discharge_info.NOSOLOGICO = discharge_info.NOSOLOGICO.apply(str)

    # Load vitals
    vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)

    # Load lab test
    lab = pd.read_csv('%s/emergency_room/lab_results.csv' % path)
    lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
    lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(get_lab_dates)


    # Filter by Nosologico (vitals and anagraphics)
    patients_nosologico = vitals[vitals['NOSOLOGICO'].isin(discharge_info["NOSOLOGICO"])]['NOSOLOGICO'].unique()
    patients_nosologico = lab[lab['NOSOLOGICO'].isin(patients_nosologico)]['NOSOLOGICO'].unique()
    discharge_info = discharge_info[discharge_info['NOSOLOGICO'].isin(patients_nosologico)]
    vitals = vitals[vitals['NOSOLOGICO'].isin(patients_nosologico)]
    lab = lab[lab['NOSOLOGICO'].isin(patients_nosologico)]
    dataset_comorbidities.index = [str(i) for i in dataset_comorbidities.index]
    dataset_comorbidities = dataset_comorbidities.loc[patients_nosologico]


    # Create final dataset
    #-------------------------------------------------------------------------------------
    # Anagraphics (translated to english)
    anagraphics_features = ['Sex', 'Age', 'Outcome']
    dataset_anagraphics = pd.DataFrame(columns=anagraphics_features, index=patients_nosologico)
    dataset_anagraphics.loc[:, anagraphics_features] = discharge_info[['NOSOLOGICO'] + anagraphics_features].set_index('NOSOLOGICO')
    dataset_anagraphics.loc[:, 'Sex'] = dataset_anagraphics.loc[:, 'Sex'].astype('category')
    dataset_anagraphics.loc[:, 'Outcome'] = dataset_anagraphics.loc[:, 'Outcome'].astype('category')

    # Data with ER vitals
    vital_signs = ['SaO2', 'P. Max', 'P. Min', 'F. Card.', 'F. Resp.', 'Temp.', 'Dolore', 'GCS', 'STICKGLI']
    if lab_tests:
        vital_signs.remove('SaO2')  # Remove oxygen saturation if we have lab values (it is there)

    dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=patients_nosologico)
    for p in patients_nosologico:
        vitals_p = vitals[vitals['NOSOLOGICO'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
        for vital_name in vital_signs:
            # Take mean if multiple values
            vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
            vital_value = pd.to_numeric(vital_value).mean()
            dataset_vitals.loc[p, vital_name] = vital_value


    # Adjust missing columns
    dataset_vitals = remove_missing(dataset_vitals)

    # Rename to English
    dataset_vitals = dataset_vitals.rename(columns={"P. Max": "systolic_blood_pressure",
                                                    "P. Min": "diastolic_blood_pressure",
                                                    "F. Card.": "cardiac_frequency",
                                                    "Temp.": "temperature_celsius",
                                                    "F. Resp.": "respiratory_frequency"
                                                    })




    # Remove missing test (groups) with more than 40% nonzeros
    lab_tests = lab['COD_INTERNO_PRESTAZIONE'].unique().tolist()
    dataset_lab_tests = pd.DataFrame(False, columns=lab_tests, index=patients_nosologico)
    for p in patients_nosologico:
        for lab_test_name in lab[lab['NOSOLOGICO'] == p]['COD_INTERNO_PRESTAZIONE']:
            dataset_lab_tests.loc[p, lab_test_name] = True

    # 30% removes tests that are not present and the COVID-19 lab test
    lab_tests_reduced = remove_missing(dataset_lab_tests, missing_type=False, nan_threashold=30, impute=False)

    # Data with lab results

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

        dataset_lab_test = pd.DataFrame(np.nan, columns=lab_test_features_names, index=patients_nosologico)
        for p in patients_nosologico:
            lab_p = lab_test_temp[lab_test_temp['NOSOLOGICO'] == p][['COD_INTERNO_PRESTAZIONE', 'DATA_RICHIESTA', 'PRESTAZIONE', 'VALORE']]
            for lab_name in lab_test_features:
                if any(lab_p['PRESTAZIONE'] == lab_name):
                    lab_p_name = lab_p[lab_p['PRESTAZIONE'] == lab_name]
                    idx = lab_p_name['DATA_RICHIESTA'].idxmin()  # Pick first date of test if multiple
                    dataset_lab_test.loc[p, test_name.strip() + ": " + lab_name] = lab_p_name.loc[idx]['VALORE']
        dataset_lab[lab_test] = dataset_lab_test

    dataset_lab_full = pd.concat([v for _,v in dataset_lab.items()],
                                 axis=1, sort=True).astype(np.float64)
    dataset_lab_full = remove_missing(dataset_lab_full)


    # Rename dataset laboratory
    dataset_lab_full = dataset_lab_full.rename(columns={
        'ALT: ALT': 'Alanine Aminotransferase (ALT)',
        'AST: AST': 'Aspartate Aminotransferase (AST)',
        'Creatinina UAR: CREATININA SANGUE': 'Blood Creatinine',
        'Potassio: POTASSIEMIA': 'Potassium Blood Level',
        'Cloruremia: CLORUREMIA': 'Chlorine Blood Level',
        'Proteina C Reattiva: PCR - PROTEINA C REATTIVA': 'C-Reactive Protein (CRP)',
        'Glucosio ematico: GLICEMIA': 'Glycemia',
        'Azoto ematico UAR: AZOTO UREICO EMATICO': 'Blood Urea Nitrogen (BUN)',
        'Emogasanalisi su sangue arterioso: ACIDO LATTICO': 'ABG: Lactic Acid',
        'Emogasanalisi su sangue arterioso: FO2HB': 'ABG: FO2Hb',
        'Emogasanalisi su sangue arterioso: CTCO2': 'ABG: CTCO2',
        'Emogasanalisi su sangue arterioso: HCT': 'ABG: Hematocrit (HCT)',
        'Emogasanalisi su sangue arterioso: IONE BICARBONATO STD': 'ABG: standard bicarbonate (sHCO3)',
        'Emogasanalisi su sangue arterioso: BE(ECF)': 'ABG: Base Excess (ecf)',
        'Emogasanalisi su sangue arterioso: ECCESSO DI BASI': 'ABG: Base Excess',
        'Emogasanalisi su sangue arterioso: FHHB': 'ABG: FHHb',
        'Emogasanalisi su sangue arterioso: PO2': 'ABG: PaO2',
        'Emogasanalisi su sangue arterioso: OSSIGENO SATURAZIONE': 'ABG: Oxygen Saturation (SaO2)',
        'Emogasanalisi su sangue arterioso: PCO2': 'ABG: PaCO2',
        'Emogasanalisi su sangue arterioso: PH EMATICO': 'ABG: pH',
        'Emogasanalisi su sangue arterioso: CALCIO IONIZZATO': 'ABG: Ionized Calcium',
        'Emogasanalisi su sangue arterioso: CARBOSSIEMOGLOBINA': 'ABG: COHb',
        'Emogasanalisi su sangue arterioso: METAEMOGLOBINA': 'ABG: MetHb',
        'Sodio: SODIEMIA': 'Blood Sodium',
        'TEMPO DI PROTROMBINA UAR: (PT) TEMPO DI PROTROMBINA': 'Prothrombin Time (PT)',
        'TEMPO DI TROMBOPLASTINA PARZIALE: TEMPO DI TROMBOPLASTINA PARZIALE ATTIVATO': 'Activated Partial Thromboplastin Time (aPTT)',
        'Calcemia: CALCEMIA': 'Blood Calcium',
        'BILIRUBINA TOTALE REFLEX: BILIRUBINA TOTALE': 'Total Bilirubin',
        'Amilasi: AMILASI NEL SIERO' : 'Blood Amylase',
        'Colinesterasi: COLINESTERASI': 'Cholinesterase',
        'Emocromocitometrico (Urgenze): VOLUME CORPUSCOLARE MEDIO': 'CBC: Mean Corpuscular Volume (MCV)',
        'Emocromocitometrico (Urgenze): CONCENTRAZIONE HB MEDIA': 'CBC: Mean Corpuscular Hemoglobin Concentration (MCHC)',
        'Emocromocitometrico (Urgenze): PIASTRINE': 'CBC: Platelets',
        'Emocromocitometrico (Urgenze): EMATOCRITO': 'CBC: Hematocrit (HCT)',
        'Emocromocitometrico (Urgenze): VALORE DISTRIBUTIVO GLOBULI ROSSI': 'CBC: Red cell Distribution Width (RDW)',
        'Emocromocitometrico (Urgenze): LEUCOCITI': 'CBC: Leukocytes',
        'Emocromocitometrico (Urgenze): EMOGLOBINA': 'CBC: Hemoglobin',
        'Emocromocitometrico (Urgenze): CONTENUTO HB MEDIO': 'CBC: Mean corpuscular haemoglobin (MCH)',
        'Emocromocitometrico (Urgenze): ERITROCITI': 'CBC: Erythrocytes'
        })


    data = {'anagraphics': dataset_anagraphics,
            'comorbidities': dataset_comorbidities,
            'vitals': dataset_vitals,
            'lab': dataset_lab_full}

    return data
