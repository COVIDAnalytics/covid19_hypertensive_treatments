import pandas as pd
import numpy as np 
import analyzer.loaders.cremona.utils as u

path = '../../../../data/cremona'

# Load Cremona data
discharge_info = pd.read_csv('%s/general/discharge_info.csv' % path)
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

comorb_long = u.comorbidities_long(discharge_info)
comorb_long.drop(["index"], axis = 1, inplace = True)
comorb_long.rename(columns={'comorb':'DIAGNOSIS_CODE',
                            'id':'NOSOLOGICO'}, inplace = True)
comorb_long['NOSOLOGICO'] = comorb_long['NOSOLOGICO'].apply(str)

# Cleanup discharge information: keep only covid patients
discharge_info = u.cleanup_discharge_info(discharge_info)
patients = discharge_info['NOSOLOGICO']

# Load Treatment data
feb = pd.read_csv('%s/therapies/drugs_2020_02.csv' % path)
mar = pd.read_csv('%s/therapies/drugs_2020_03.csv' % path)
apr = pd.read_csv('%s/therapies/drugs_2020_04.csv' % path)
may = pd.read_csv('%s/therapies/drugs_2020_05.csv' % path)

drugs = pd.concat([feb, mar, apr, may]).reset_index(drop = True)

# Modify the Nosologico format to match the one from discharge
drugs['Nosologico'] = drugs['Nosologico'].apply(lambda x: x[:-7])
patients = np.asarray(list(set(patients).intersection(set(drugs['Nosologico']))))

# Filter patients that are diagnosed for covid
drugs = drugs[drugs['Nosologico'].isin(patients)]

# Clean drugs dataframe
drugs = drugs.drop(['note1', 'note2'], axis = 1)
drugs = drugs[drugs['Principio Attivo'] != '-']
drugs['Principio Attivo'] = drugs['Principio Attivo'].str.lower()
# drugs = drugs[drugs.AIC.apply(lambda x: x[0] != 'N')]
# drugs.AIC = drugs.AIC.astype(int)
# drugs.AIC = drugs.AIC.astype(str)
# AICs = drugs.AIC.astype(str).unique()

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

cremona_treatments = pd.DataFrame(0, index=range(len(patients)), columns=['NOSOLOGICO'] + u.COLS_TREATMENTS)
cremona_treatments['NOSOLOGICO'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'NOSOLOGICO'].reset_index(drop = True)
cremona_treatments['HOSPITAL'] = np.repeat('Cremona', len(patients))
cremona_treatments['COUNTRY'] = np.repeat('Italy', len(patients))
cremona_treatments['DT_HOSPITAL_ADMISSION'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'DT_HOSPITAL_ADMISSION'].reset_index(drop = True)
cremona_treatments['GENDER'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'Gender'].reset_index(drop = True)
cremona_treatments[['RACE', 'PREGNANT', 'SMOKING', 'MAINHEARTDISEASE']] = np.nan
cremona_treatments['AGE'] = discharge_info.loc[discharge_info['NOSOLOGICO'].isin(patients), 'Age'].reset_index(drop = True)

for i in range(len(u.COMORBS_TREATMENTS_HCUP)):
    name = u.COMORBS_TREATMENTS_NAMES[i]
    hcups = u.COMORBS_TREATMENTS_HCUP[i]
    for j in patients:
        cremona_treatments.loc[comorb_long['NOSOLOGICO'] == j, name] = (sum(comorb_long.loc[comorb_long['NOSOLOGICO'] == j, 'DIAGNOSIS_CODE'].isin(hcups)) > 0)