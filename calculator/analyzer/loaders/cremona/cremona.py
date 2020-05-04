import pandas as pd
#  import datetime
import numpy as np
import pickle


import analyzer.loaders.cremona.utils as u


def load_cremona(path, discharge_data = True, comorbidities_data = True, vitals_data = True, lab_tests=True, demographics_data = False, swabs_data = False):

    # Load discharge info
    discharge_info = pd.read_csv('%s/general/discharge_info.csv' % path)

    comorb_long = u.comorbidities_long(discharge_info)
    comorb_long.drop(["index"], axis = 1, inplace = True)
    comorb_long.rename(columns={'comorb':'DIAGNOSIS_CODE',
                                'id':'NOSOLOGICO'}, inplace = True)
    comorb_long['NOSOLOGICO'] = comorb_long['NOSOLOGICO'].apply(str)

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

    # Load demographics
    demographics = pd.read_csv("%s/emergency_room/general.csv" % path)
    demographics = u.cleanup_demographics(demographics)

    # Load Swab
    swabs = u.get_swabs(lab)

    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
    list_datasets = np.asarray([discharge_info, comorb_long, vitals, lab, demographics, swabs])
    dataset_array = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])

    # Filter patients common to all datasets
    patients = u.filter_patients(list_datasets[dataset_array])

    datasets = []

    # Create final dataset
    if discharge_data:
        dataset_discharge = u.create_dataset_discharge(discharge_info, patients, icu=icu)
        datasets.append(dataset_discharge)

    if comorbidities_data:
        dataset_comorbidities= u.create_dataset_comorbidities(comorb_long, 9, patients)
        datasets.append(dataset_comorbidities.set_index('NOSOLOGICO'))

    if vitals_data:
        dataset_vitals = u.create_vitals_dataset(vitals, patients, lab_tests=lab_tests)
        datasets.append(dataset_vitals)

    if lab_tests:
        dataset_lab = u.create_lab_dataset(lab, patients)
        datasets.append(dataset_lab)

    if demographics_data:
        datasets.append(demographics.set_index('NOSOLOGICO'))

    if swabs_data:
        datasets.append(swabs.set_index('NOSOLOGICO'))

    datasets = np.asarray(datasets)

    data = dict(zip(name_datasets[dataset_array], datasets))

    return data
