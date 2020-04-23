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


def load_swabs(path):

    # Load vitals
    vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)

    # Load lab data
    lab = pd.read_csv('%s/emergency_room/lab_results.csv' % path)
    lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
    lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(get_lab_dates)

    # Identify which patients have a swab
    covid = lab[lab.COD_INTERNO_PRESTAZIONE == 'COV19']
    covid = covid[covid.VALORE_TESTO.isin(['POSITIVO', 'Negativo', 'Debolmente positivo'])]
    covid.VALORE_TESTO = covid.VALORE_TESTO.isin(['POSITIVO','Debolmente positivo']).astype(int).astype('category')
    covid = covid[~ covid.NOSOLOGICO.duplicated()] # drop duplicated values
    swab = covid[['NOSOLOGICO', 'VALORE_TESTO']].set_index('NOSOLOGICO')
    swab = swab.rename(columns = {'VALORE_TESTO': 'Swab Result'})['Swab Result'].astype('category') # Transform the dataframe in a series

    # Keep only the patients that have a swab
    covid_pats = covid.NOSOLOGICO
    lab = lab[lab.NOSOLOGICO.isin(covid_pats)]
    lab_tests = lab['COD_INTERNO_PRESTAZIONE'].unique().tolist()


    #Keep the vitals of patients with swab
    covid_pats = list(set(covid_pats).intersection(set(vitals.NOSOLOGICO)))
    vitals = vitals[vitals.NOSOLOGICO.isin(covid_pats)]
    swab = swab[swab.index.isin(covid_pats)]

    #Create vitals dataset
    vital_signs = ['P. Max', 'P. Min', 'F. Card.', 'F. Resp.', 'Temp.', 'Dolore', 'GCS', 'STICKGLI']
    dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=covid_pats)
    for p in covid_pats:
        vitals_p = vitals[vitals['NOSOLOGICO'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
        for vital_name in vital_signs:
            # Take mean if multiple values
            vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
            vital_value = pd.to_numeric(vital_value).mean()
            dataset_vitals.loc[p, vital_name] = vital_value

    # Adjust missing columns
    dataset_vitals = remove_missing(dataset_vitals)

    # Rename to English
    dataset_vitals = dataset_vitals.rename(columns={"P. Max": "Systolic Blood Pressure",
                                                    "P. Min": "Diastolic Blood Pressure",
                                                    "F. Card.": "Cardiac Frequency",
                                                    "Temp.": "Temperature Celsius",
                                                    "F. Resp.": "Respiratory Frequency"
                                                    })


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

        dataset_lab_test = pd.DataFrame(np.nan, columns=lab_test_features_names, index=covid_pats)
        for p in covid_pats:
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


    X = dataset_lab_full.join(dataset_vitals)
    y = swab
    return X, y


