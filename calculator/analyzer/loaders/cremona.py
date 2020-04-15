import pandas as pd
import datetime
import numpy as np

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

#Julia
#  from julia.api import Julia
#  jl = Julia(compiled_modules=False)
#  from interpretableai import iai


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


def get_abg_dates(t):
    try:
        date = datetime.datetime.strptime(t, '%m/%d/%Y %I:%M %p')
    except ValueError:
        date = datetime.datetime.strptime(t, '%m/%d/%Y')

    return date


def remove_missing(df, nan_threashold=35):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_values = pd.DataFrame({'percent_missing': percent_missing})
    df_features = missing_values[missing_values['percent_missing'] < nan_threashold].index.tolist()
    df = df[df_features]

    imp_mean = IterativeImputer(random_state=0)
    imp_mean.fit(df)
    imputed_df = imp_mean.transform(df)

    return pd.DataFrame(imputed_df, index=df.index, columns=df.columns)


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


def load_cremona(path):

    # Load data
    #-------------------------------------------------------------------------------------
    anagraphics = pd.read_csv("%s/anagraphics/anagraphics.csv" % path).dropna(how='all')
    anagraphics['NOSOLOGICO']= anagraphics['NOSOLOGICO'].astype(str)

    # Load drugs
    drugs_cremona = pd.read_csv("%s/therapies/drugs_cremona.csv" % path)
    drugs_orgoglio_po = pd.read_csv("%s/therapies/drugs_oglio_po.csv" % path)
    drugs = drugs_cremona.append(drugs_orgoglio_po).dropna(how='all')

    # Load comorbidities
    comorbidities_data = pd.read_csv('%s/therapies/active_substances_comorbidities.csv' % path)[['Active_substance', 'therapy_for_filtered']]

    # Load vital signs in ER
    vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)

    # Load ICU admissions
    icu = pd.read_csv('%s/icu/icu_transfers.csv' % path)
    icu['NOSOLOGICO']= icu['NOSOLOGICO'].astype(str)
    icu.drop_duplicates(['NOSOLOGICO'], inplace=True)
    idx_icu = icu['DESCR_REP_A'].str.contains('TERAPIA INTENSIVA CR') | icu['DESCR_REP_A'].str.contains('TERAPIA INTENSIVA OP')
    icu = icu.loc[idx_icu]

    # Load arterial blood gas test
    abg = pd.read_csv('%s/emergency_room/arterial_blood_gas.csv' % path)
    abg = abg.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    abg['NOSOLOGICO'] = abg['NOSOLOGICO'].astype(str)
    abg['DATA_RICHIESTA'] = abg['DATA_RICHIESTA'].apply(get_abg_dates)


    # Filter and merge
    #-------------------------------------------------------------------------------------
    # Select only covid patients
    anagraphics = anagraphics[anagraphics['ESITO TAMPONE'].str.contains('POSITIVO')]

    # Pick only people with end date
    anagraphics = anagraphics[anagraphics["DATA DI DIMISSIONE/DECESSO"].notnull()]

    # Remove patients who have been transferred
    anagraphics = anagraphics[~anagraphics['ESITO'].str.contains('TRASFERITO')]

    # Create binary outcome
    anagraphics['ESITO'] = anagraphics['ESITO'].apply(fix_outcome)

    # Drop anagraphics duplicates
    drop_anagraphics_duplicates(anagraphics)


    # Filter patients in all datasets
    #-------------------------------------------------------------------------------------

    # Filter by Codice Fiscale (anagraphics and drugs)
    patients_codice_fiscale = anagraphics[anagraphics["CODICE FISCALE"].isin(drugs["Codice Fiscale"])]['CODICE FISCALE'].unique()
    anagraphics = anagraphics[anagraphics['CODICE FISCALE'].isin(patients_codice_fiscale)]

    # Filter by Nosologico (vitals and anagraphics)
    patients_nosologico = vitals[vitals['NOSOLOGICO'].isin(anagraphics["NOSOLOGICO"])]['NOSOLOGICO'].unique()
    patients_nosologico = abg[abg['NOSOLOGICO'].isin(patients_nosologico)]['NOSOLOGICO'].unique()
    anagraphics = anagraphics[anagraphics['NOSOLOGICO'].isin(patients_nosologico)]
    vitals = vitals[vitals['NOSOLOGICO'].isin(patients_nosologico)]
    abg = abg[abg['NOSOLOGICO'].isin(patients_nosologico)]

    # Filter again Codice Fiscale (drugs)
    patients_codice_fiscale = anagraphics['CODICE FISCALE'].unique()
    drugs_covid = drugs[drugs['Codice Fiscale'].isin(patients_codice_fiscale)]

    assert len(patients_codice_fiscale) == len(patients_nosologico)
    print("Len patients :", len(patients_codice_fiscale))


    # Fix swapped dates
    #-------------------------------------------------------------------------------------
    idx_wrong = pd.to_datetime(anagraphics["DATA DI DIMISSIONE/DECESSO"], format='%m/%d/%y') < pd.to_datetime(anagraphics["DATA DI RICOVERO"], format='%m/%d/%y')
    wrong_dates = anagraphics[['DATA DI DIMISSIONE/DECESSO','DATA DI RICOVERO']].loc[idx_wrong]
    anagraphics.loc[idx_wrong, 'DATA DI RICOVERO'] = wrong_dates['DATA DI DIMISSIONE/DECESSO']
    anagraphics.loc[idx_wrong, 'DATA DI DIMISSIONE/DECESSO'] = wrong_dates['DATA DI RICOVERO']

    # Add codice fiscale to datasets with nosologico
    #-------------------------------------------------------------------------------------
    vitals = vitals.merge(anagraphics[['CODICE FISCALE', 'NOSOLOGICO']], on='NOSOLOGICO')
    vitals.drop(['NOSOLOGICO'], axis='columns', inplace=True)
    abg = abg.merge(anagraphics[['CODICE FISCALE', 'NOSOLOGICO']], on='NOSOLOGICO')
    abg.drop(['NOSOLOGICO'], axis='columns', inplace=True)


    # Compute length of stay
    anagraphics['length_of_stay'] = anagraphics["DATA DI DIMISSIONE/DECESSO"].apply(n_days) - anagraphics["DATA DI RICOVERO"].apply(n_days)
    print("Minimum length of stay ", anagraphics['length_of_stay'].min())
    print("Maximum length of stay ", anagraphics['length_of_stay'].max())

    # Add ICU information to anagraphics
    #-------------------------------------------------------------------------------------
    anagraphics['icu'] = 0
    anagraphics.loc[anagraphics['NOSOLOGICO'].isin(icu['NOSOLOGICO']), 'icu'] = 1


    # Create final dataset
    #-------------------------------------------------------------------------------------
    # Anagraphics (translated to english)
    anagraphics_features = ['SESSO', "ETA'", 'ESITO', 'icu']
    dataset_anagraphics = pd.DataFrame(columns=anagraphics_features, index=patients_codice_fiscale)
    dataset_anagraphics.loc[:, anagraphics_features] = anagraphics[['CODICE FISCALE'] + anagraphics_features].set_index('CODICE FISCALE')
    dataset_anagraphics = dataset_anagraphics.rename(columns={"ETA'": "age", "SESSO": "sex", "ESITO": "outcome"})
    dataset_anagraphics.loc[:, 'sex'] = dataset_anagraphics.loc[:, 'sex'].astype('category')
    dataset_anagraphics.loc[:, 'icu'] = dataset_anagraphics.loc[:, 'icu'].astype('category')
    dataset_anagraphics.loc[:, 'outcome'] = dataset_anagraphics.loc[:, 'outcome'].astype('category')
    dataset_anagraphics['outcome'] = dataset_anagraphics['outcome'].cat.rename_categories({'DIMESSO': 'discharged', 'DECEDUTO': 'deceased'})

    # Comorbidities from drugs
    comorbidities = comorbidities_data['therapy_for_filtered'].dropna().unique().tolist()
    dataset_comorbidities = pd.DataFrame(0, columns=comorbidities, index=patients_codice_fiscale)
    for p in patients_codice_fiscale:
        drugs_p = drugs_covid[drugs_covid['Codice Fiscale'] == p]['Principio Attivo']
        for d in drugs_p:
            if d != 'NESSUN PRINCIPIO ATTIVO':
                comorb_d = comorbidities_data[comorbidities_data['Active_substance'] == d]['therapy_for_filtered']
                if len(comorb_d) != 1:
                    import ipdb; ipdb.set_trace()
                    raise ValueError('Error in dataset. We need only one entry per active substance')
                comorb_d = comorb_d.iloc[0]
                if not pd.isnull(comorb_d):
                    dataset_comorbidities.loc[p, comorb_d] = 1
    for c in comorbidities:  # Drop columns with all zeros
        if dataset_comorbidities[c].sum() == 0:
            dataset_comorbidities.drop(c, axis='columns', inplace=True)
    dataset_comorbidities = dataset_comorbidities.astype('category')


    # Data with ER vitals
    #  vital_signs = ['SaO2', 'P. Max', 'P. Min', 'F. Card.', 'F. Resp.', 'Temp.', 'Dolore', 'GCS', 'STICKGLI']
    vital_signs = ['P. Max', 'P. Min', 'F. Card.', 'F. Resp.', 'Temp.', 'Dolore', 'GCS', 'STICKGLI']
    dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=patients_codice_fiscale)
    for p in patients_codice_fiscale:
        vitals_p = vitals[vitals['CODICE FISCALE'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
        for vital_name in vital_signs:
            # Take mean if multiple values
            vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
            vital_value = pd.to_numeric(vital_value).mean()
            dataset_vitals.loc[p, vital_name] = vital_value

    # Adjust missing columns
    dataset_vitals = remove_missing(dataset_vitals)

    # Rename to English
    dataset_vitals = dataset_vitals.rename(columns={"P_ Max": "systolic_blood_pressure",
                                                    "P_ Min": "diastolic_blood_pressure",
                                                    "F_ Card_": "cardiac_frequency",
                                                    "Temp_": "temperature_celsius",
                                                    "F_ Resp_": "respiratory_frequency"})


    # Data with arterial blood gas
    abg_features = abg['PRESTAZIONE'].unique().tolist()
    dataset_abg = pd.DataFrame(np.nan, columns=abg_features, index=patients_codice_fiscale)
    for p in patients_codice_fiscale:
        abg_p = abg[abg['CODICE FISCALE'] == p][['DATA_RICHIESTA', 'PRESTAZIONE', 'VALORE']]
        for abg_name in abg_p['PRESTAZIONE']:
            abg_p_name = abg_p[abg_p['PRESTAZIONE'] == abg_name]
            idx = abg_p_name['DATA_RICHIESTA'].idxmin()  # Pick first date of test if multiple
            dataset_abg.loc[p, abg_name] = abg_p_name.loc[idx]['VALORE']

    # Adjust missing columns
    dataset_abg = remove_missing(dataset_abg)

    data = {'anagraphics': dataset_anagraphics,
            'comorbidities': dataset_comorbidities,
            'vitals': dataset_vitals,
            'abg': dataset_abg}

    return data
