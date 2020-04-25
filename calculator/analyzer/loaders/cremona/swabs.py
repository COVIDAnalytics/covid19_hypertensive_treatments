import pandas as pd
import numpy as np
import pickle

import analyzer.loaders.cremona.utils as u

def load_swabs(path, lab_tests):

    # Load anagraphics
    anagraphics = pd.read_csv("%s/emergency_room/general.csv" % path)
    anagraphics = anagraphics[['N_SCHEDA_PS', 'PZ_SESSO_PS', "PZ_DATA_NASCITA_PS"]]
    anagraphics['PZ_DATA_NASCITA_PS'] = pd.to_datetime(anagraphics['PZ_DATA_NASCITA_PS'], format='%Y-%m-%d %H:%M:%S')
    anagraphics['Age'] = anagraphics['PZ_DATA_NASCITA_PS'].apply(u.get_age)
    anagraphics = anagraphics.drop('PZ_DATA_NASCITA_PS', axis = 1)
    anagraphics = anagraphics.rename(columns = {'N_SCHEDA_PS' : 'NOSOLOGICO', 'PZ_SESSO_PS' : 'Sex'})
    anagraphics['Sex'] = (anagraphics['Sex'] == 'F').astype(int)
    anagraphics['NOSOLOGICO'] = anagraphics['NOSOLOGICO'].astype(str)

    # Load vitals
    vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)

    # Load lab data
    lab = pd.read_csv('%s/emergency_room/lab_results.csv' % path)
    lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
    lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(u.get_lab_dates)

    # Identify which patients have a swab
    covid = lab[lab.COD_INTERNO_PRESTAZIONE == 'COV19']
    covid = covid[covid.VALORE_TESTO.isin(['POSITIVO', 'Negativo', 'Debolmente positivo'])]
    covid.VALORE_TESTO = covid.VALORE_TESTO.isin(['POSITIVO','Debolmente positivo']).astype(int).astype('category')
    covid = covid[~ covid.NOSOLOGICO.duplicated()] # drop duplicated values
    swab = covid[['NOSOLOGICO', 'VALORE_TESTO']].set_index('NOSOLOGICO')
    swab = swab.rename(columns = {'VALORE_TESTO': 'Swab'})['Swab'].astype('category') # Transform the dataframe in a series

    # Keep only the patients that have a swab
    covid_pats = covid.NOSOLOGICO
    lab = lab[lab.NOSOLOGICO.isin(covid_pats)]
    lab_tests = lab['COD_INTERNO_PRESTAZIONE'].unique().tolist()


    #Keep the vitals of patients with swab
    covid_pats = list(set(covid_pats).intersection(set(vitals.NOSOLOGICO)).intersection(set(anagraphics.NOSOLOGICO)))
    vitals = vitals[vitals['NOSOLOGICO'].isin(covid_pats)]
    swab = swab[swab.index.isin(covid_pats)]
    anagraphics = anagraphics[anagraphics['NOSOLOGICO'].isin(covid_pats)].set_index('NOSOLOGICO')
    dataset_anagraphics = anagraphics.join(swab)
    

    #Create vitals dataset
    vital_signs = u.VITAL_SIGNS
    if lab_tests:
        vital_signs.remove('SaO2')  # Remove oxygen saturation if we have lab values (it is there)

    dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=covid_pats)
    for p in covid_pats:
        vitals_p = vitals[vitals['NOSOLOGICO'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
        for vital_name in vital_signs:
            # Take mean if multiple values
            vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
            vital_value = pd.to_numeric(vital_value).mean()
            dataset_vitals.loc[p, vital_name] = vital_value

    # Adjust missing columns
    dataset_vitals = u.remove_missing(dataset_vitals)

    # Rename to English
    dataset_vitals = dataset_vitals.rename(columns=u.RENAMED_VITALS_COLUMNS)


    #Unstack the dataset and transform the entries in True/False
    dataset_lab_tests = lab[['NOSOLOGICO', 'COD_INTERNO_PRESTAZIONE', 'VALORE_TESTO']].groupby(['NOSOLOGICO', 'COD_INTERNO_PRESTAZIONE']).count().unstack().notna()
    dataset_lab_tests.columns = [i[1] for i in dataset_lab_tests.columns] # because of groupby, the columns are a tuple


    # 30% removes tests that are not present and the COVID-19 lab test
    lab_tests_reduced = u.remove_missing(dataset_lab_tests, missing_type=False, nan_threashold=30, impute=False)


    # Filter data entries per test
    lab_reduced = lab[lab['COD_INTERNO_PRESTAZIONE'].isin(lab_tests_reduced.columns)]


    # Create lab features for each exam
    dataset_lab = {}
    for lab_test in lab_tests_reduced.columns:
        # Create dataset
        lab_test_temp = lab_reduced.loc[lab_reduced['COD_INTERNO_PRESTAZIONE'] == lab_test]
        lab_test_features = lab_test_temp['PRESTAZIONE'].unique().tolist()

        # Remove unnecessary features
        lab_test_features = u.clean_lab_features(lab_test_features)

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
    dataset_lab_full = u.remove_missing(dataset_lab_full)

    dataset_lab_full = dataset_lab_full.rename(columns=u.RENAMED_LAB_COLUMNS)

    data = {'anagraphics': dataset_anagraphics,
            'vitals': dataset_vitals,
            'lab': dataset_lab_full}

    return data


