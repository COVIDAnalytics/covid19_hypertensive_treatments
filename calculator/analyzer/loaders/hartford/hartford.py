
path = '/nfs/sloanlab001/data/HartfordHealthCare/HHCtoMIT/Oncology/May2020/20200512 '

def load_hartford(path, discharge_data = True, comorbidities_data = True, vitals_data = True, lab_tests=True, demographics_data = False, swabs_data = False):

    # Load discharge info
    discharge_info = []

    comorb_long = pd.read_csv('%sco-morbidities med hx.txt' % path, sep='\t', header = None)
    # comorb_long.drop(["index"], axis = 1, inplace = True)
    # comorb_long.rename(columns={'comorb':'DIAGNOSIS_CODE',
    #                             'id':'NOSOLOGICO'}, inplace = True)
    # comorb_long['NOSOLOGICO'] = comorb_long['NOSOLOGICO'].apply(str)

    # Cleanup discharge information
    # discharge_info = u.cleanup_discharge_info(discharge_info)

    # Load vitals
    vitals = pd.read_csv('%svitals.txt' % path, sep='\t')
    vitals.FLO_MEAS_NAME.value_counts().to_csv("/home/hwiberg/research/hhc/vitals_fields.csv")
    # vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
    # vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)

    # Load lab test
    lab = pd.read_csv('%slabs.txt' % path, sep='\t')
    lab.LAB_NAME.value_counts().to_csv("/home/hwiberg/research/hhc/lab_fields.csv")
    lab.query('LAB_NAME == `SARS COV 2 COVDR`')
    # lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
    # lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
    # lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(u.get_lab_dates)

    # # Load ICU data
    # icu = pd.read_csv("%s/icu/icu_transfers.csv" %path)
    # icu = icu[['RC_CODNOSO', 'TRASF_TI']]
    # icu = icu.dropna()
    # icu.RC_CODNOSO = icu.RC_CODNOSO.apply(int).apply(str)
    # icu = icu.rename(columns={'RC_CODNOSO':'NOSOLOGICO','TRASF_TI':'ICU'})
    # icu.ICU = icu.ICU.astype(int).astype('category')

    # Load demographics
    demographics = pd.read_csv('%spatient.txt' % path, sep='\t')
    # demographics = u.cleanup_demographics(demographics)

    # Load Swab
    # swabs = u.get_swabs(lab)

    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
    list_datasets = np.asarray([discharge_info, comorb_long, vitals, lab, demographics, swabs])
    dataset_array = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])

    # Filter patients common to all datasets
    patients = u.filter_patients(list_datasets[dataset_array])

    encounters = pd.read_csv('%spatient_encounter.txt' % path, sep='\t')

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