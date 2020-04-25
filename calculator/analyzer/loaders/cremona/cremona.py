import pandas as pd
#  import datetime
import numpy as np
import pickle


import analyzer.loaders.cremona.utils as u


def load_cremona(path, lab_tests=True):

    # Load discharge info
    discharge_info = pd.read_csv('%s/general/discharge_info.csv' % path)

    # Export comorbidities to process with R
    u.export_comorbidities(discharge_info, '%s/general/patient_comorbidity.csv' % path)
    # TODO: Run R command automatically
    # import subprocess
    # subprocess.call(['R', 'analyzer/loaders/group_comorbidities.R'])
    #Import the comorbidity csv and dictionary from R
    comorbidities = pd.read_csv('%s/general/comorbidities.csv' % path)
    comorbidities.drop("Unnamed: 0", axis = 1, inplace = True)
    comorbidities = comorbidities.rename(columns={"id": "NOSOLOGICO"})
    ccs_map = pd.read_csv('./analyzer/ccs_map.csv', index_col=0, header=None, squeeze=True)

    #Change the name of the cols from CCS code to comorbidity name
    comorbidities.columns = ['NOSOLOGICO'] + [ccs_map.loc[i] for i in list(comorbidities.columns[1:].astype(np.int64))]

    # False and True are transformed to 0 and 1 categories
    comorbidities = comorbidities.astype('int').astype('category')
    # Cleanup discharge information
    discharge_info = u.cleanup_discharge_info(discharge_info)

    # Load vitals
    vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
    vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)

    # Load lab test
    lab = pd.read_csv('%s/emergency_room/lab_results.csv' % path)
    lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
    lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(u.get_lab_dates)

    # Load ICU data
    icu = pd.read_csv("%s/icu/icu_transfers.csv" %path)
    icu = icu[['RC_CODNOSO', 'TRASF_TI']]
    icu = icu.dropna()
    icu.RC_CODNOSO = icu.RC_CODNOSO.apply(int).apply(str)
    icu = icu.rename(columns={'RC_CODNOSO':'NOSOLOGICO','TRASF_TI':'ICU'})
    icu.ICU = icu.ICU.astype(int).astype('category')

    # Filter patients common to all datasets
    patients = u.filter_patients([discharge_info, comorbidities,
                                  vitals, icu, lab])

    # Create final dataset
    dataset_anagraphics = u.create_dataset_anagraphics(discharge_info, patients, icu=icu)
    dataset_comorbidities = u.create_dataset_comorbidities(comorbidities, patients)
    dataset_vitals = u.create_vitals_dataset(vitals, patients, lab_tests=lab_tests)
    dataset_lab = u.create_lab_dataset(lab, patients)

    data = {'anagraphics': dataset_anagraphics,
            'comorbidities': dataset_comorbidities,
            'vitals': dataset_vitals,
            'lab': dataset_lab}

    return data
