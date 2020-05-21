import pandas as pd
import os
import datetime
import np

def load_hartford(path, discharge_data = True, comorbidities_data = True, vitals_data = True, lab_tests=True, demographics_data = False, swabs_data = False):

    # Load discharge info
    df = pd.read_csv(path)
    df.set_index('PATIENT_ID')

    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
    dataset_array = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])

    datasets = []

    # Create final dataset
    if discharge_data:
        dataset_discharge = df[u.ADMISSIONS_COLUMNS]
        datasets.append(dataset_discharge)

    if comorbidities_data:
        dataset_comorbidities= df[u.COMORBIDITIES_COLUMNS]
        datasets.append(dataset_comorbidities.set_index(''))

    if vitals_data:
        dataset_vitals = df[u.VITALS_COLUMNS]
        datasets.append(dataset_vitals)

    if lab_tests:
        dataset_lab = df[u.LAB_COLUMNS]
        datasets.append(dataset_lab)

    if demographics_data:
        demographics = df[u.DEMOGRAPHICS_COLUMNS]
        datasets.append(demographics.set_index('NOSOLOGICO'))

    if swabs_data:
        datasets.append(None)

    datasets = np.asarray(datasets)

    data = dict(zip(name_datasets[dataset_array], datasets))

    return data