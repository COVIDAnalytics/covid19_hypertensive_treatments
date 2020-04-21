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
                ('IONE BICARBONATO' != x)]  # We keep standard directly

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
                     'Sesso': 'sex',
                     'Età':'age',
                     'Modalità di dimissione':'outcome'})
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
    anagraphics_features = ['sex', 'age', 'outcome']
    dataset_anagraphics = pd.DataFrame(columns=anagraphics_features, index=patients_nosologico)
    dataset_anagraphics.loc[:, anagraphics_features] = discharge_info[['NOSOLOGICO'] + anagraphics_features].set_index('NOSOLOGICO')
    dataset_anagraphics.loc[:, 'sex'] = dataset_anagraphics.loc[:, 'sex'].astype('category')
    dataset_anagraphics.loc[:, 'outcome'] = dataset_anagraphics.loc[:, 'outcome'].astype('category')

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
    dataset_lab_full = dataset_lab_full.rename({
        'ALT: ALT': 'Alanine Aminotransferase (ALT)',
        'AST: AST': 'Aspartate Aminotransferase (AST)',
        'Creatinina UAR: CREATININA SANGUE': 'Blood Creatinine',
        'Potassio: POTASSIEMIA': 'Potassium Blood Level',
        'Cloruremia: CLORUREMIA': 'Chlorine Blood Level',
        'Proteina C Reattiva: PCR - PROTEINA C REATTIVA': 'C-Reactive Protein (CRP)',
        'Glucosio ematico: GLICEMIA': 'Glycemia',
        'Azoto ematico UAR: AZOTO UREICO EMATICO': 'Blood Urea Nitrogen (BUN)',
        'Emogasanalisi su sangue arterioso: ACIDO LATTICO': 'ABG: Lactic Acid',
        'Emogasanalisi su sangue arterioso: FO2HB': 'ABG: FO2HB',
        'Emogasanalisi su sangue arterioso: CTCO2': 'ABG: CTCO2',
        'Emogasanalisi su sangue arterioso: HCT': 'ABG: Hematocrit (HCT)'
        'Emogasanalisi su sangue arterioso: IONE BICARBONATO STD': 'ABG: standard bicarbonate (sHCO3)',
        'Emogasanalisi su sangue arterioso: BE(ECF)': 'ABG: BE(ecf)', # TOFIX WITH ECCESSO DI BASI
        'Emogasanalisi su sangue arterioso: FHHB': 'ABG: FHHb',
        'Emogasanalisi su sangue arterioso: PO2': 'ABG: PO2',
 'Emogasanalisi su sangue arterioso: ECCESSO DI BASI',
 'Emogasanalisi su sangue arterioso: OSSIGENO SATURAZIONE',
 'Emogasanalisi su sangue arterioso: PCO2',
 'Emogasanalisi su sangue arterioso: PH EMATICO',
 'Emogasanalisi su sangue arterioso: CALCIO IONIZZATO',
 'Emogasanalisi su sangue arterioso: CARBOSSIEMOGLOBINA',
 'Emogasanalisi su sangue arterioso: METAEMOGLOBINA',
 'Sodio: SODIEMIA',
 'TEMPO DI PROTROMBINA UAR: (PT) TEMPO DI PROTROMBINA',
 'TEMPO DI PROTROMBINA UAR: TEMPO DI PROTROMBINA RATIO',
 'Calcemia: CALCEMIA',
'BILIRUBINA TOTALE REFLEX: BILIRUBINA TOTALE',
 'TEMPO DI TROMBOPLASTINA PARZIALE: TEMPO DI TROMBOPLASTINA PARZIALE ATTIVATO',
 'Amilasi: AMILASI NEL SIERO',
 'Colinesterasi: COLINESTERASI',
 'Emocromocitometrico (Urgenze): VOLUME CORPUSCOLARE MEDIO',
 'Emocromocitometrico (Urgenze): CONCENTRAZIONE HB MEDIA',
 'Emocromocitometrico (Urgenze): PIASTRINE',
 'Emocromocitometrico (Urgenze): EMATOCRITO',
 'Emocromocitometrico (Urgenze): VALORE DISTRIBUTIVO GLOBULI ROSSI',
 'Emocromocitometrico (Urgenze): LEUCOCITI',
 'Emocromocitometrico (Urgenze): EMOGLOBINA',
 'Emocromocitometrico (Urgenze): CONTENUTO HB MEDIO',
 'Emocromocitometrico (Urgenze): ERITROCITI']



        })






    # TODO:

    # 1. Join dataframes
    # 2. Check missing
    # 3. Keep only not missing
    # 4. Check names


    #  perc_missing = get_percentages(dataset_lab_tests, missing_type=False)
    #  perc_missing['text'] = [lab[lab['COD_INTERNO_PRESTAZIONE'] == perc_missing.index[i]]['DESCR_PRESTAZIONE'].values[0] for i in range(len(perc_missing))]




    from IPython import embed; embed()


    import ipdb; ipdb.set_trace()

    # Remove duplicate features




    # Adjust missing columns
    #  dataset_lab = remove_missing(dataset_lab)
    dataset_lab = dataset_lab.rename(columns =
                                {'VOLUME CORPUSCOLARE MEDIO': 'Mean Corpuscular Volume (MCV)',
                                'PIASTRINE': 'Platelets',
                                'EMATOCRITO': 'Hematocrit (?)', # TOFIX
                                'ALT': 'Alanine Aminotransferase (ALT)',
                                'AST': 'Aspartate Aminotransferase (AST)',
                                'CREATININA SANGUE': 'Blood Creatinine',
                                'POTASSIEMIA': 'Potassium Blood Level',
                                'CLORUREMIA': 'Chlorine Blood Level',
                                'PCR - PROTEINA C REATTIVA': 'C-Reactive Protein (CRP)',
                                'GLICEMIA': 'Glycemia',
                                'UREA EMATICA': 'Hematic Urea (?)',   # TOFIX
                                'AZOTO UREICO EMATICO': 'Blood Urea Nitrogen (BUN)',  # TOFIX
                                'ACIDO LATTICO': 'Lactic Acid',
                                'FO2HB': 'FO2HB',  # TOFIX Oxygen saturation
                                'CTCO2': 'Bicarbonate (CTCO2)',
                                'HCT': 'Hematocrit Levels (HCT)',  # TOFIX
                                'IONE BICARBONATO STD': "IONE BICARBONATO STD (?)",
                                'BE(ECF)': 'BEECF',
                                'FHHB': 'FHHB',
                                'IONE BICARBONATO': "IONE BICARBONATO (?)",
                                'PO2': 'Partial Pressure of Oxygen (PO2)',
                                'ECCESSO DI BASI': 'Base Excess (BE)',
                                'OSSIGENO SATURAZIONE': 'SaO2',  # TOFIX Oxygen saturation
                                'PCO2': 'Partial Pressure of Carbon Dioxide (PCO2)',
                                'PH EMATICO': 'Hematic PH',
                                'CALCIO IONIZZATO': 'Ionized Calcium',
                                'CARBOSSIEMOGLOBINA': 'Carboxyhemoglobin',
                                'METAEMOGLOBINA': 'Methemoglobin',
                                'SODIEMIA': 'Sodium Levels',
                                '(PT) TEMPO DI PROTROMBINA': 'Prothrombin Time',
                                'TEMPO DI PROTROMBINA RATIO': 'Prothrombin Time Ratio',
                                'CALCEMIA': 'Calcium Levels',
                                'BILIRUBINA TOTALE': 'Bilurin',
                                'TEMPO DI TROMBOPLASTINA PARZIALE ATTIVATO': 'Partial Thromboplastin Time Activated (APTT)',
                                'VALORE DISTRIBUTIVO GLOBULI ROSSI': 'Red Cell Distribution (RDW)',
                                'LEUCOCITI': 'Leukocytes',
                                'EMOGLOBINA': 'Hemoglobin',
                                'CONTENUTO HB MEDIO': 'Mean Corpuscular Hemoglobin (MCH)',
                                'ERITROCITI': 'Erythrocytes',
                                'CONCENTRAZIONE HB MEDIA': 'Mean Corpuscular Hemoglobin Concentration (MCHC)',
                                'AMILASI NEL SIERO': 'Amylase Serum Level',
                                'COLINESTERASI': 'Cholinesterase'})

    #  dataset_lab.drop([c for c in dataset_lab.columns if '?' in c],
    #                   axis='columns', inplace=True)

    import ipdb; ipdb.set_trace()

    data = {'anagraphics': dataset_anagraphics,
            'comorbidities': dataset_comorbidities,
            'vitals': dataset_vitals,
            'lab': dataset_lab}

    return data

    # # Convert (to export for R processing)
    # dataset_comorbidities = pd.DataFrame(columns=['id', 'comorb'])
    # for i in range(len(discharge_info)):
    #     d_temp = discharge_info.iloc[i]
    #     df_temp = pd.DataFrame({'id': [d_temp['NumeroScheda']] * 6,
    #                             'comorb': [d_temp['Principale'],
    #                                        d_temp['Dia1'],
    #                                        d_temp['Dia2'],
    #                                        d_temp['Dia3'],
    #                                        d_temp['Dia4'],
    #                                        d_temp['Dia5']]})
    #     dataset_comorbidities = dataset_comorbidities.append(df_temp)

    # dataset_comorbidities = dataset_comorbidities.dropna().reset_index()
    # dataset_comorbidities.to_csv('comorb.csv')

    #  anagraphics = pd.read_csv("%s/anagraphics/anagraphics.csv" % path).dropna(how='all')
    #  anagraphics['NOSOLOGICO']= anagraphics['NOSOLOGICO'].astype(str)
    #
    #  # Load drugs
    #  drugs_cremona = pd.read_csv("%s/therapies/drugs_cremona.csv" % path)
    #  drugs_orgoglio_po = pd.read_csv("%s/therapies/drugs_oglio_po.csv" % path)
    #  drugs = drugs_cremona.append(drugs_orgoglio_po).dropna(how='all')
    #
    #  # Load comorbidities
    #  comorbidities_data = pd.read_csv('%s/therapies/active_substances_comorbidities.csv' % path)[['Active_substance', 'therapy_for_filtered']]
    #
    #  # Load vital signs in ER
    #  vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    #  vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    #  vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)
    #
    #  # Load ICU admissions
    #  icu = pd.read_csv('%s/icu/icu_transfers.csv' % path)
    #  icu['NOSOLOGICO']= icu['NOSOLOGICO'].astype(str)
    #  icu.drop_duplicates(['NOSOLOGICO'], inplace=True)
    #  idx_icu = icu['DESCR_REP_A'].str.contains('TERAPIA INTENSIVA CR') | icu['DESCR_REP_A'].str.contains('TERAPIA INTENSIVA OP')
    #  icu = icu.loc[idx_icu]
    #
    #  # Load arterial blood gas test
    #  lab = pd.read_csv('%s/emergency_room/lab_results.csv' % path)
    #  lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    #  lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
    #  lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(get_lab_dates)
    #
    #
    #  # Filter and merge
    #  #-------------------------------------------------------------------------------------
    #  # Select only covid patients
    #  anagraphics = anagraphics[anagraphics['ESITO TAMPONE'].str.contains('POSITIVO')]
    #
    #  # Pick only people with end date
    #  anagraphics = anagraphics[anagraphics["DATA DI DIMISSIONE/DECESSO"].notnull()]
    #
    #  # Remove patients who have been transferred
    #  anagraphics = anagraphics[~anagraphics['ESITO'].str.contains('TRASFERITO')]
    #
    #  # Create binary outcome
    #  anagraphics['ESITO'] = anagraphics['ESITO'].apply(fix_outcome)
    #
    #  # Drop anagraphics duplicates
    #  drop_anagraphics_duplicates(anagraphics)
    #
    #
    #  # Filter patients in all datasets
    #  #-------------------------------------------------------------------------------------
    #
    #  # Filter by Codice Fiscale (anagraphics and drugs)
    #  patients_codice_fiscale = anagraphics[anagraphics["CODICE FISCALE"].isin(drugs["Codice Fiscale"])]['CODICE FISCALE'].unique()
    #  anagraphics = anagraphics[anagraphics['CODICE FISCALE'].isin(patients_codice_fiscale)]
    #
    #  # Filter by Nosologico (vitals and anagraphics)
    #  patients_nosologico = vitals[vitals['NOSOLOGICO'].isin(anagraphics["NOSOLOGICO"])]['NOSOLOGICO'].unique()
    #  patients_nosologico = lab[lab['NOSOLOGICO'].isin(patients_nosologico)]['NOSOLOGICO'].unique()
    #  anagraphics = anagraphics[anagraphics['NOSOLOGICO'].isin(patients_nosologico)]
    #  vitals = vitals[vitals['NOSOLOGICO'].isin(patients_nosologico)]
    #  lab = lab[lab['NOSOLOGICO'].isin(patients_nosologico)]
    #
    #  # Filter again Codice Fiscale (drugs)
    #  patients_codice_fiscale = anagraphics['CODICE FISCALE'].unique()
    #  drugs_covid = drugs[drugs['Codice Fiscale'].isin(patients_codice_fiscale)]
    #
    #  assert len(patients_codice_fiscale) == len(patients_nosologico)
    #  print("Len patients :", len(patients_codice_fiscale))
    #
    #
    #  # Fix swapped dates
    #  #-------------------------------------------------------------------------------------
    #  idx_wrong = pd.to_datetime(anagraphics["DATA DI DIMISSIONE/DECESSO"], format='%m/%d/%y') < pd.to_datetime(anagraphics["DATA DI RICOVERO"], format='%m/%d/%y')
    #  wrong_dates = anagraphics[['DATA DI DIMISSIONE/DECESSO','DATA DI RICOVERO']].loc[idx_wrong]
    #  anagraphics.loc[idx_wrong, 'DATA DI RICOVERO'] = wrong_dates['DATA DI DIMISSIONE/DECESSO']
    #  anagraphics.loc[idx_wrong, 'DATA DI DIMISSIONE/DECESSO'] = wrong_dates['DATA DI RICOVERO']
    #
    #  # Add codice fiscale to datasets with nosologico
    #  #-------------------------------------------------------------------------------------
    #  vitals = vitals.merge(anagraphics[['CODICE FISCALE', 'NOSOLOGICO']], on='NOSOLOGICO')
    #  vitals.drop(['NOSOLOGICO'], axis='columns', inplace=True)
    #  lab = lab.merge(anagraphics[['CODICE FISCALE', 'NOSOLOGICO']], on='NOSOLOGICO')
    #  lab.drop(['NOSOLOGICO'], axis='columns', inplace=True)
    #
    #
    #  # Compute length of stay
    #  anagraphics['length_of_stay'] = anagraphics["DATA DI DIMISSIONE/DECESSO"].apply(n_days) - anagraphics["DATA DI RICOVERO"].apply(n_days)
    #  print("Minimum length of stay ", anagraphics['length_of_stay'].min())
    #  print("Maximum length of stay ", anagraphics['length_of_stay'].max())
    #
    #  # Add ICU information to anagraphics
    #  #-------------------------------------------------------------------------------------
    #  anagraphics['icu'] = 0
    #  anagraphics.loc[anagraphics['NOSOLOGICO'].isin(icu['NOSOLOGICO']), 'icu'] = 1
    #
    #
    #  # Create final dataset
    #  #-------------------------------------------------------------------------------------
    #  # Anagraphics (translated to english)
    #  anagraphics_features = ['SESSO', "ETA'", 'ESITO', 'icu']
    #  dataset_anagraphics = pd.DataFrame(columns=anagraphics_features, index=patients_codice_fiscale)
    #  dataset_anagraphics.loc[:, anagraphics_features] = anagraphics[['CODICE FISCALE'] + anagraphics_features].set_index('CODICE FISCALE')
    #  dataset_anagraphics = dataset_anagraphics.rename(columns={"ETA'": "age", "SESSO": "sex", "ESITO": "outcome"})
    #  dataset_anagraphics.loc[:, 'sex'] = dataset_anagraphics.loc[:, 'sex'].astype('category')
    #  dataset_anagraphics.loc[:, 'icu'] = dataset_anagraphics.loc[:, 'icu'].astype('category')
    #  dataset_anagraphics.loc[:, 'outcome'] = dataset_anagraphics.loc[:, 'outcome'].astype('category')
    #  dataset_anagraphics['outcome'] = dataset_anagraphics['outcome'].cat.rename_categories({'DIMESSO': 'discharged', 'DECEDUTO': 'deceased'})
    #
    #  # Comorbidities from drugs
    #  comorbidities = comorbidities_data['therapy_for_filtered'].dropna().unique().tolist()
    #  dataset_comorbidities = pd.DataFrame(0, columns=comorbidities, index=patients_codice_fiscale)
    #  for p in patients_codice_fiscale:
    #      drugs_p = drugs_covid[drugs_covid['Codice Fiscale'] == p]['Principio Attivo']
    #      for d in drugs_p:
    #          if d != 'NESSUN PRINCIPIO ATTIVO':
    #              comorb_d = comorbidities_data[comorbidities_data['Active_substance'] == d]['therapy_for_filtered']
    #              if len(comorb_d) != 1:
    #                  import ipdb; ipdb.set_trace()
    #                  raise ValueError('Error in dataset. We need only one entry per active substance')
    #              comorb_d = comorb_d.iloc[0]
    #              if not pd.isnull(comorb_d):
    #                  dataset_comorbidities.loc[p, comorb_d] = 1
    #  for c in comorbidities:  # Drop columns with all zeros
    #      if dataset_comorbidities[c].sum() == 0:
    #          dataset_comorbidities.drop(c, axis='columns', inplace=True)
    #  dataset_comorbidities = dataset_comorbidities.astype('category')
    #
    #
    #  # Data with ER vitals
    #  vital_signs = ['SaO2', 'P. Max', 'P. Min', 'F. Card.', 'F. Resp.', 'Temp.', 'Dolore', 'GCS', 'STICKGLI']
    #  if lab_tests:
    #      vital_signs.remove('Sa02')  # Remove oxygen saturation if we have lab values (it is there)
    #
    #  dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=patients_codice_fiscale)
    #  for p in patients_codice_fiscale:
    #      vitals_p = vitals[vitals['CODICE FISCALE'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
    #      for vital_name in vital_signs:
    #          # Take mean if multiple values
    #          vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
    #          vital_value = pd.to_numeric(vital_value).mean()
    #          dataset_vitals.loc[p, vital_name] = vital_value
    #
    #  # Adjust missing columns
    #  dataset_vitals = remove_missing(dataset_vitals)
    #
    #  # Rename to English
    #  dataset_vitals = dataset_vitals.rename(columns={"P. Max": "systolic_blood_pressure",
    #                                                  "P. Min": "diastolic_blood_pressure",
    #                                                  "F. Card.": "cardiac_frequency",
    #                                                  "Temp.": "temperature_celsius",
    #                                                  "F. Resp.": "respiratory_frequency"})
    #
    #
    #  # Data with lab results
    #  lab_features = lab['PRESTAZIONE'].unique().tolist()
    #  dataset_lab = pd.DataFrame(np.nan, columns=lab_features, index=patients_codice_fiscale)
    #  for p in patients_codice_fiscale:
    #      lab_p = lab[lab['CODICE FISCALE'] == p][['DATA_RICHIESTA', 'PRESTAZIONE', 'VALORE']]
    #      for lab_name in lab_p['PRESTAZIONE']:
    #          lab_p_name = lab_p[lab_p['PRESTAZIONE'] == lab_name]
    #          idx = lab_p_name['DATA_RICHIESTA'].idxmin()  # Pick first date of test if multiple
    #          dataset_lab.loc[p, lab_name] = lab_p_name.loc[idx]['VALORE']
    #
    #
    #  # Adjust missing columns
    #  dataset_lab = remove_missing(dataset_lab)
    #
    #  data = {'anagraphics': dataset_anagraphics,
    #          'comorbidities': dataset_comorbidities,
    #          'vitals': dataset_vitals,
    #          'lab': dataset_lab}
    #
    #  return data
