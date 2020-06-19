import pandas as pd
import numpy as np 
import analyzer.loaders.cremona.utils as u

path = '../../../../data/cremona'

# Load Cremona data
discharge_info = pd.read_csv('%s/general/discharge_info.csv' % path)
comorb_long = u.comorbidities_long(discharge_info)
covid_patients = discharge_info['Principale'].isin(u.LIST_DIAGNOSIS)

for d in u.DIAGNOSIS_COLUMNS:
    covid_patients = covid_patients | discharge_info[d].isin(u.LIST_DIAGNOSIS)

discharge_info = discharge_info[covid_patients]

# Keep discharge codes and transform the dependent variable to binary
discharge_info = discharge_info[discharge_info['Modalità di dimissione'].isin(u.DISCHARGE_CODES)]
discharge_info['Modalità di dimissione'] = \
    (discharge_info['Modalità di dimissione'] == u.DISCHARGE_CODE_RELEASED).apply(int) #transform to binary

# Drop Duplicated Observations
discharge_info.drop_duplicates(['NumeroScheda', 'Modalità di dimissione'],
                                inplace=True)
discharge_info.drop_duplicates(['NumeroScheda'], inplace=True)
discharge_info = discharge_info[['NumeroScheda', 'Sesso', 'Età', 'Data di ricovero', 'Modalità di dimissione']]
discharge_info = discharge_info.rename(columns={'NumeroScheda': 'NOSOLOGICO',
                                                'Sesso': 'Gender',
                                                'Età':'Age',
                                                'Data di ricovero': 'DT_HOSPITAL_ADMISSION',
                                                'Modalità di dimissione':'Outcome'})
discharge_info['DT_HOSPITAL_ADMISSION'] = pd.to_datetime(discharge_info['DT_HOSPITAL_ADMISSION'])
discharge_info.NOSOLOGICO = discharge_info.NOSOLOGICO.apply(str)
# Cleanup discharge information: keep only covid patients
patients = discharge_info['NOSOLOGICO']

comorb_long.drop(["index"], axis = 1, inplace = True)
comorb_long.rename(columns={'comorb':'DIAGNOSIS_CODE',
                            'id':'NOSOLOGICO'}, inplace = True)
comorb_long['NOSOLOGICO'] = comorb_long['NOSOLOGICO'].apply(str)

# Load Treatment data
feb = pd.read_csv('%s/therapies/drugs_2020_02.csv' % path)
mar = pd.read_csv('%s/therapies/drugs_2020_03.csv' % path)
apr = pd.read_csv('%s/therapies/drugs_2020_04.csv' % path)
may = pd.read_csv('%s/therapies/drugs_2020_05.csv' % path)

drugs = pd.concat([feb, mar, apr, may]).reset_index(drop = True)

# Modify the Nosologico format to match the one from discharge
drugs['Nosologico'] = drugs['Nosologico'].apply(lambda x: x[:-7])
patients = np.asarray(list(set(patients).intersection(set(drugs['Nosologico']))))

# Load vitals
vitals = pd.read_csv('%s/emergency_room/vital_signs.csv' % path)
vitals = vitals.rename(columns={"SCHEDA_PS": "NOSOLOGICO"})
vitals['NOSOLOGICO'] = vitals['NOSOLOGICO'].astype(str)
dataset_vitals = u.create_vitals_dataset(vitals, patients, lab_tests=True)
dataset_vitals = dataset_vitals[u.VITALS_TREAT]
dataset_vitals = dataset_vitals.rename(columns = u.VITALS_TREAT_RENAME)
dataset_vitals.loc[dataset_vitals['FAST_BREATHING'].notna(), 'FAST_BREATHING'] = (dataset_vitals.loc[dataset_vitals['FAST_BREATHING'].notna(), 'FAST_BREATHING'] > 20).astype(int)
dataset_vitals.loc[dataset_vitals['BLOOD_PRESSURE_ABNORMAL_B'].notna(), 'BLOOD_PRESSURE_ABNORMAL_B'] = (dataset_vitals.loc[dataset_vitals['BLOOD_PRESSURE_ABNORMAL_B'].notna(), 'BLOOD_PRESSURE_ABNORMAL_B'] > 140).astype(int)

# Load lab test
lab = pd.read_csv('%s/emergency_room/lab_results.csv' % path)
lab = lab.rename(columns={"SC_SCHEDA": "NOSOLOGICO"})
lab['NOSOLOGICO'] = lab['NOSOLOGICO'].astype(str)
lab['DATA_RICHIESTA'] = lab['DATA_RICHIESTA'].apply(u.get_lab_dates)
lab = lab[lab['NOSOLOGICO'].isin(patients)]
dates_pcr = lab[lab.DESCR_PRESTAZIONE == 'CORONAVIRUS COVID19 PCR Real Time                                               '][['NOSOLOGICO', 'DATA_RICHIESTA']].drop_duplicates('NOSOLOGICO').reset_index(drop = True)

dataset_lab = u.create_lab_dataset(lab, patients)
dataset_lab = dataset_lab[u.LABS_TREAT]
dataset_lab = dataset_lab.rename(columns = u.LABS_TREAT_RENAME)
dataset_lab.loc[dataset_lab['SAT02_BELOW92'].notna(), 'SAT02_BELOW92'] = (dataset_lab.loc[dataset_lab['SAT02_BELOW92'].notna(), 'SAT02_BELOW92'] < 92).astype(int)
dataset_lab.loc[dataset_lab['DDDIMER_B'].notna(), 'DDDIMER_B'] = (dataset_lab.loc[dataset_lab['DDDIMER_B'].notna(), 'DDDIMER_B'] > 0.5).astype(int)
dataset_lab.loc[dataset_lab['PROCALCITONIN_B'].notna(), 'PROCALCITONIN_B'] = (dataset_lab.loc[dataset_lab['PROCALCITONIN_B'].notna(), 'PROCALCITONIN_B'] > 0.5).astype(int)
dataset_lab.loc[dataset_lab['PCR_B'].notna(), 'PCR_B'] = (dataset_lab.loc[dataset_lab['PCR_B'].notna(), 'PCR_B'] > 100).astype(int)
dataset_lab.loc[dataset_lab['TRANSAMINASES_B'].notna(), 'TRANSAMINASES_B'] = (dataset_lab.loc[dataset_lab['TRANSAMINASES_B'].notna(), 'TRANSAMINASES_B'] > 40).astype(int)
dataset_lab.loc[dataset_lab['LDL_B'].notna(), 'LDL_B'] = (dataset_lab.loc[dataset_lab['LDL_B'].notna(), 'LDL_B'] > 300).astype(int)

# Filter patients that are diagnosed for covid
drugs = drugs[drugs['Nosologico'].isin(patients)]

# Clean drugs dataframe
drugs = drugs.drop(['note1', 'note2'], axis = 1)
drugs = drugs[drugs['Principio Attivo'] != '-']
drugs['Principio Attivo'] = drugs['Principio Attivo'].str.lower()

# Count the occurrence of a particular treatment
df = drugs.groupby(['Principio Attivo', 'Nosologico']).count().Reparto.reset_index().drop('Reparto', axis = 1)
active_principles, counts = np.unique(df['Principio Attivo'], return_counts=True)
data_treat = pd.DataFrame([active_principles, counts]).T
data_treat.columns = ['Active Principle', 'Occurrence']
data_treat = data_treat.sort_values('Occurrence', ascending = False)
data_treat['Proportion'] = (data_treat['Occurrence']/len(drugs.Nosologico.unique())).apply(lambda x: np.round(x, 3))
ATCs = list(drugs.ATC.unique())
ATC_antibiotics = np.asarray([ATCs[i] for i in range(len(ATCs)) if ATCs[i][:3] == 'J01']) # Standard code for antibiotic
ATC_antivirals = np.asarray([ATCs[i] for i in range(len(ATCs)) if ATCs[i][:3] == 'J05']) # Standard code for antiviral

# Create the dataframe in the HOPE format
cremona_treatments = pd.DataFrame(0, index=range(len(patients)), columns=['NOSOLOGICO'] + u.COLS_TREATMENTS)
cremona_treatments['NOSOLOGICO'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'NOSOLOGICO'].reset_index(drop = True)
cremona_treatments['HOSPITAL'] = np.repeat('Cremona', len(patients))
cremona_treatments['COUNTRY'] = np.repeat('Italy', len(patients))
cremona_treatments['DT_HOSPITAL_ADMISSION'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'DT_HOSPITAL_ADMISSION'].reset_index(drop = True)
cremona_treatments['GENDER'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'Gender'].reset_index(drop = True)
cremona_treatments[['RACE', 'PREGNANT', 'SMOKING', 'MAINHEARTDISEASE', 'GLASGOW_COMA_SCORE', 'CHESTXRAY_BNORMALITY', 'ONSET_DATE_DIFF']] = np.nan
cremona_treatments['AGE'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'Age'].reset_index(drop = True)

# Get the dataframe to map DIAGNOSIS CODE to HCUP_ORDER
icd_dict = pd.read_csv('../../../analyzer/hcup_dictionary_icd9.csv')
comorb_long = comorb_long.merge(icd_dict[['DIAGNOSIS_CODE', 'HCUP_ORDER']], how = 'left', on = 'DIAGNOSIS_CODE').dropna().reset_index(drop = True)

# Fill the comorbidities table
for i in range(len(u.COMORBS_TREATMENTS_HCUP)):
    name = u.COMORBS_TREATMENTS_NAMES[i]
    hcups = u.COMORBS_TREATMENTS_HCUP[i]
    for j in patients:
        cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, name] = int(sum(comorb_long.loc[comorb_long['NOSOLOGICO'] == j, 'HCUP_ORDER'].isin(hcups)) > 0)

# Fill IN_TREATMENTS
for i in range(len(u.IN_TREATMENTS)):
    name = u.IN_TREATMENTS_NAME[i]
    treat = u.IN_TREATMENTS[i]
    for j in patients:
        cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, name] = int(sum(drugs.loc[drugs['Nosologico'] == j, 'ATC'].apply(lambda x: treat in x)) > 0)

# Fill in Lab and Vitals
for j in patients:
    cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, dataset_lab.columns] = dataset_lab.loc[j, :].to_frame().T.values
    cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, dataset_vitals.columns] = dataset_vitals.loc[j, :].to_frame().T.values

# Fill TREATMENTS
for i in range(len(u.TREATMENTS)):
    name = u.TREATMENTS_NAME[i]
    treat = u.TREATMENTS[i]
    for j in patients:
        cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, name] = int(sum(drugs.loc[drugs['Nosologico'] == j, 'ATC'].apply(lambda x: treat in x)) > 0)

# Fill in Outcome and days diff from covid test to hospitalization
for j in patients:
    cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, 'DEATH'] = discharge_info.loc[discharge_info['NOSOLOGICO'] == j, 'Outcome'].values
    if j in dates_pcr['NOSOLOGICO'].to_list():
        cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, 'TEST_DATE_DIFF'] = int((dates_pcr.loc[dates_pcr['NOSOLOGICO'] == j, 'DATA_RICHIESTA'].values - cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, 'DT_HOSPITAL_ADMISSION'].values ).item()/(60*60*60*60*60*60*60))
    else:
        cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, 'TEST_DATE_DIFF'] = np.NaN

# Add the regimen for each patient
cremona_treatments['REGIMEN'] = cremona_treatments.apply(lambda row: u.get_regimen(row['CLOROQUINE'], row['ANTIVIRAL'], row['ANTICOAGULANTS']), axis = 1)

# Fill COMORB_DEATH
for j in patients:
    cremona_treatments.loc[cremona_treatments['NOSOLOGICO'] == j, 'COMORB_DEATH'] = int(sum(comorb_long.loc[comorb_long['NOSOLOGICO'] == j, 'HCUP_ORDER'].isin(u.COMORB_DEATH)) > 0)