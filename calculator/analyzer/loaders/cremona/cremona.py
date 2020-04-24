import pandas as pd
#  import datetime
import numpy as np
import pickle


import analyzer.loaders.cremona.utils as u




def load_cremona(path, lab_tests=True):

    # Load data
    discharge_info = pd.read_csv('%s/general/discharge_info.csv' % path)
    covid_patients = discharge_info['Principale'].isin(u.LIST_DIAGNOSIS)

    for d in u.DIAGNOSIS_COLUMNS:
        covid_patients = covid_patients | discharge_info[d].isin(u.LIST_DIAGNOSIS)

    discharge_info = discharge_info[covid_patients]

    # Export comorbidities to process with R
    u.export_comorbidities(discharge_info, '%s/general/nosologico_com.csv' % path)

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


    # False and True are transformed to 0 and 1 categories
    dataset_comorbidities = dataset_comorbidities.astype('int').astype('category')

    # Keep only the patients that we have in the comorbidities dataframe
    pat_comorb = dataset_comorbidities.index
    discharge_info = discharge_info[discharge_info.NumeroScheda.isin(pat_comorb)]

    #Join the two categories of Diabetes
    dataset_comorbidities["Diabetes"] = np.zeros(len(dataset_comorbidities))
    dataset_comorbidities["Diabetes"] = ((dataset_comorbidities['Diabetes mellitus without complication'].astype(int) > 0) | \
        (dataset_comorbidities['Diabetes mellitus with complications'].astype(int) > 0)).astype(int).astype('category')

    # Keep discharge codes and transform the dependent variable to binary
    discharge_info = discharge_info[discharge_info['Modalità di dimissione'].isin(u.DISCHARGE_CODES)]
    discharge_info['Modalità di dimissione'] = \
        (discharge_info['Modalità di dimissione'] == u.DISCHARGE_CODE_RELEASED).apply(int) #transform to binary

    # Drop Duplicated Observations
    discharge_info.drop_duplicates(['NumeroScheda',
                                    'Modalità di dimissione'],
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

    # Load vitals
    vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)

    # Load lab test
    lab = pd.read_csv('%s/emergency_room/lab_results.csv' % path)
    lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
    lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(u.get_lab_dates)


    # Filter by Nosologico (vitals and anagraphics)
    patients_nosologico = vitals[vitals['NOSOLOGICO'].isin(discharge_info["NOSOLOGICO"])]['NOSOLOGICO'].unique()
    patients_nosologico = lab[lab['NOSOLOGICO'].isin(patients_nosologico)]['NOSOLOGICO'].unique()
    discharge_info = discharge_info[discharge_info['NOSOLOGICO'].isin(patients_nosologico)]
    vitals = vitals[vitals['NOSOLOGICO'].isin(patients_nosologico)]
    lab = lab[lab['NOSOLOGICO'].isin(patients_nosologico)]
    dataset_comorbidities.index = [str(i) for i in dataset_comorbidities.index]
    dataset_comorbidities = dataset_comorbidities.loc[patients_nosologico]

    # Keep only the comorbidities that appear more than 10 times and remove pneumonia ones
    cols_keep = list(dataset_comorbidities.columns[dataset_comorbidities.sum() >10])
    for e in u.LIST_REMOVE_COMORBIDITIES:
        cols_keep.remove(e)
    dataset_comorbidities = dataset_comorbidities[cols_keep]

    # Add ICU data
    icu = pd.read_csv("%s/icu/icu_transfers.csv" %path)
    icu = icu[['RC_CODNOSO', 'TRASF_TI']]
    icu = icu.dropna()
    icu.RC_CODNOSO = icu.RC_CODNOSO.apply(int).apply(str)
    icu = icu.rename(columns={'RC_CODNOSO':'NOSOLOGICO','TRASF_TI':'ICU'})
    icu = icu[icu.NOSOLOGICO.isin(patients_nosologico)].set_index("NOSOLOGICO")
    icu.ICU = icu.ICU.astype(int).astype('category')

    # Create final dataset
    #-------------------------------------------------------------------------------------
    # Anagraphics (translated to english)
    anagraphics_features = ['Sex', 'Age', 'Outcome']
    dataset_anagraphics = pd.DataFrame(columns=anagraphics_features, index=patients_nosologico)
    dataset_anagraphics.loc[:, anagraphics_features] = discharge_info[['NOSOLOGICO'] + anagraphics_features].set_index('NOSOLOGICO')
    dataset_anagraphics.loc[:, 'Sex'] = dataset_anagraphics.loc[:, 'Sex'].astype('category')
    dataset_anagraphics.loc[:, 'Outcome'] = dataset_anagraphics.loc[:, 'Outcome'].astype('category')
    dataset_anagraphics = dataset_anagraphics.join(icu)
    dataset_anagraphics.Sex = dataset_anagraphics.Sex.cat.codes.astype('category')

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
    dataset_vitals = u.remove_missing(dataset_vitals)

    # Rename to English
    dataset_vitals = dataset_vitals.rename(columns=u.RENAMED_VITALS_COLUMNS)


    # Remove missing test (groups) with more than 40% nonzeros
    lab_tests = lab['COD_INTERNO_PRESTAZIONE'].unique().tolist()
    dataset_lab_tests = pd.DataFrame(False, columns=lab_tests, index=patients_nosologico)

    for p in patients_nosologico:
        for lab_test_name in lab[lab['NOSOLOGICO'] == p]['COD_INTERNO_PRESTAZIONE']:
            dataset_lab_tests.loc[p, lab_test_name] = True

    # 30% removes tests that are not present and the COVID-19 lab test
    lab_tests_reduced = u.remove_missing(dataset_lab_tests, missing_type=False, nan_threashold=30, impute=False)

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
        lab_test_features = u.clean_lab_features(lab_test_features)

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
    dataset_lab_full = u.remove_missing(dataset_lab_full)


    # Rename dataset laboratory
    dataset_lab_full = dataset_lab_full.rename(columns=u.RENAMED_LAB_COLUMNS)

    import ipdb; ipdb.set_trace()


    data = {'anagraphics': dataset_anagraphics,
            'comorbidities': dataset_comorbidities,
            'vitals': dataset_vitals,
            'lab': dataset_lab_full}

    return data
