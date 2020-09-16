import numpy as np
import pandas as pd
import analyzer.loaders.cremona.utils as uc
import analyzer.loaders.partners.utils as u
import re
from sklearn.impute import KNNImputer

data_path = '../../covid19_partners/data/v3/'
save_path = '../../covid19_partners/processed/'
version = '2020-09-16'

#%% Process demographic data
 
partners_demographics_raw = pd.read_csv(data_path+'demographics_redone/demographics2.csv')
partners_demographics_raw['COV2Positive'] = partners_demographics_raw['COV2Result'].isin(u.COVID_LABELS)
partners_demographics_raw['DEATH'] = (partners_demographics_raw['deathMinPostAdmission'].notna()).astype(int)
cov_count = partners_demographics_raw.groupby('pdgID')['COV2Positive'].agg(['count','sum'])

partners_demographics_raw['Encounter_Number'] = partners_demographics_raw.groupby(['pdgID']).cumcount()+1

## Only include patients who have (1) at least 1 positive covid result and (2) no encounters separated by more than 2 days
patients_exclude = list(partners_demographics_raw.loc[partners_demographics_raw['minFromPrevHospitalization'] > 60*24*2,'pdgID'].unique())
patients_exclude.extend([37739,47285]) ## weird cases of multiple encounters
patients_covid = partners_demographics_raw.loc[partners_demographics_raw['COV2Result'].isin(u.COVID_LABELS),'pdgID'].unique()
patients_include = set(patients_covid).difference(patients_exclude)

partners_demographics_byenc = partners_demographics_raw.loc[partners_demographics_raw['pdgID'].isin(patients_include),:]
partners_demographics_byenc['length_add'] = partners_demographics_byenc['minFromPrevHospitalization'].fillna(0) + partners_demographics_byenc['hospitalLOSMin']
lengths = partners_demographics_byenc.groupby('pdgID')['length_add'].sum().rename('TotalLOSMin').reset_index()
partners_demographics_byenc = pd.merge(partners_demographics_byenc, lengths, how = 'left', left_on = 'pdgID', right_on = 'pdgID')
partners_demographics_byenc['minFromFirstHospitalization'] = partners_demographics_byenc.groupby('pdgID')['minFromPrevHospitalization'].cumsum(skipna = True).fillna(0) + \
    partners_demographics_byenc.groupby('pdgID')['hospitalLOSMin'].cumsum(skipna = True).fillna(0) - \
        partners_demographics_byenc['hospitalLOSMin'].fillna(0)
    

## restrict to first entry
## confirm that outcomes are (1) consistent across all records for a patient, (2) no duplicate entries for an encounter
assert np.all(partners_demographics_byenc.groupby('pdgID')['DEATH'].nunique() == 1)
assert partners_demographics_byenc['hospitalEncounterHash'].nunique() == len(partners_demographics_byenc['hospitalEncounterHash'])

partners_demographics_byenc['EncounterID'] = partners_demographics_byenc['pdgID'].astype(str)+'_'+partners_demographics_byenc['Encounter_Number'].astype(str)
encounter_recode = partners_demographics_byenc[['hospitalEncounterHash','EncounterID']]
partners_demographics = partners_demographics_byenc.groupby('pdgID').first().reset_index()

# Clean columns and identify death events 
partners_demographics = partners_demographics.rename(columns = {'age': 'AGE', 'gender': 'GENDER','ethnicity':'ETHNICITY'})
partners_demographics['GENDER'] = partners_demographics['GENDER'].str.upper()

eth_map = pd.read_csv('../../covid19_partners/processed/ethnicity_recode.csv')
eth_map.replace({'OTHER':'CAUC'}, inplace = True) ## assume unavailable --> Caucasian (mode imputation)
partners_demographics = pd.merge(partners_demographics, eth_map[['ETHNICITY','RACE']], on = 'ETHNICITY', how = 'left')

demographics_clean = partners_demographics[['pdgID','hospitalEncounterHash','AGE','GENDER','RACE','TotalLOSMin','DEATH']]
demographics_clean.set_index('pdgID', inplace = True)
# demographics_clean.to_csv('../../covid19_partners/processed/demographics_clean.csv')

partners_demographics.TotalLOSMin.sort_values()


#%% process ICU data
icu_raw = pd.read_csv(data_path+'demographics_redone/icu_admission.csv')
icu = pd.merge(icu_raw.loc[icu_raw['ADTEventTypeDSC']=='Transfer In',:],encounter_recode,how='left',on='hospitalEncounterHash')
icu = icu.loc[~icu['EncounterID'].isnull()]
icu = icu[['pdgID', 'hospitalEncounterHash','EncounterID']].drop_duplicates()
icu_patients = icu['pdgID'].unique()

demographics_clean['ICU_Admission'] = demographics_clean.index.isin(icu_patients)

demographics_clean['ICU_Admission'].mean()

#%% Process labs data
labs_raw = pd.read_csv(data_path+'demographics_redone/labs.csv')
labs_o2 = pd.read_csv(data_path+'demographics_redone/labs_o2sat.csv')
labs_raw = pd.concat([labs_raw, labs_o2],  axis=0)
labs = pd.merge(labs_raw, encounter_recode, how = 'left', on = 'hospitalEncounterHash')
labs = labs.loc[~labs['EncounterID'].isnull(),:]
labs['componentNM'] = labs['componentNM'].apply(lambda x: u.PARTNERS_LABS[x])
labs = labs.sort_values(['pdgID', 'resultDTSMinPostAdmission'], ascending = [True, True])
labs = labs.drop_duplicates(['pdgID', 'componentNM']).reset_index(drop = True)

labs.query('componentNM == "C-Reactive Protein (CRP)"')[['ReferenceRangeLowNBR', 'ReferenceRangeHighNBR', 'ReferenceRangeUnitCD']]

labs.query('componentNM == "D-DIMER"')[['ReferenceRangeLowNBR', 'ReferenceRangeHighNBR', 'ReferenceRangeUnitCD']]


# labs.groupby(['pdgID','componentNM']).resultDTSMinPostAdmission.count()
# t =  labs_raw.query('pdgID == 36148')

partners_labs = pd.DataFrame(np.NaN, index = labs['pdgID'].unique(), columns = set(u.PARTNERS_LABS.values()))

for patient in labs['pdgID'].unique():
    for exam in labs['componentNM'].unique():
            value = labs.loc[(labs['pdgID'] == patient) & (labs['componentNM'] == exam), 'admissionResultValueNBR'].values
            if len(value) == 0:
                continue
            else:
                partners_labs.loc[patient, exam] = value

partners_labs.drop(['SPECIMEN SOURCE/DESCRIPTION','SPECIMEN SOURCE', 
       'SARS-COV 2 (COVID-19) PCR','SARS-COV-2 - EXTERNAL', 'COVID-19 SOURCE'], axis = 1, inplace = True)
# Clean data for consistency
# partners_labs.loc[partners_labs['CBC: Hemoglobin'] > 20, 'CBC: Hemoglobin'] = np.NaN
# partners_labs.loc[partners_labs['CBC: Leukocytes'] > 50, 'CBC: Leukocytes'] = np.NaN
# partners_labs.loc[partners_labs['CBC: Mean Corpuscular Volume (MCV)'] > 120, 'CBC: Mean Corpuscular Volume (MCV)'] = np.NaN
# partners_labs.loc[partners_labs['ABSOLUTE LYMPHS'] > 10, 'ABSOLUTE LYMPHS'] = np.NaN
# partners_labs.loc[:, 'ABSOLUTE LYMPHS'] = partners_labs.loc[:, 'ABSOLUTE LYMPHS']*1000
# partners_labs.loc[partners_labs['CBC: Platelets'] > 800, 'CBC: Platelets'] = np.NaN
# partners_labs.loc[:, 'CBC: Platelets'] = partners_labs.loc[:, 'CBC: Platelets']*1000
# partners_labs.loc[:, 'CBC: Leukocytes'] = partners_labs.loc[:, 'CBC: Leukocytes']*1000
# partners_labs.loc[:, 'D-DIMER'] = partners_labs.loc[:, 'D-DIMER']/1000
# partners_labs.loc[partners_labs['D-DIMER'] > 20, 'D-DIMER'] = np.NaN
# partners_labs.loc[partners_labs['Blood Creatinine'] > 10, 'Blood Creatinine'] = np.NaN
# partners_labs.loc[partners_labs['PROCALCITONIN'] > 5, 'PROCALCITONIN'] = np.NaN
# partners_labs.loc[partners_labs['LDH'] > 1000, 'LDH'] = np.NaN

partners_labs = partners_labs.replace({9999999:np.NaN})
partners_labs.loc[:, 'LYMPHOCYTES'] = partners_labs.loc[:, 'LYMPHOCYTES']*1000
partners_labs.loc[:, 'PLATELETS'] = partners_labs.loc[:, 'PLATELETS']*1000
partners_labs.loc[:, 'LEUCOCYTES'] = partners_labs.loc[:, 'LEUCOCYTES']*1000

# partners_labs['SAT02_BELOW92'] = \
#     partners_labs['ABG: Oxygen Saturation (SaO2)'].apply(lambda x: 1 if x < 92 else np.nan if pd.isnull(x) else 0)
# partners_labs['DDDIMER_B'] = \
#     partners_labs['D-DIMER (UG/ML)'].apply(lambda x: 1 if x > 0.5 else np.nan if pd.isnull(x) else 0)
# partners_labs['PROCALCITONIN_B'] = \
#     partners_labs['PROCALCITONIN'].apply(lambda x: 1 if x > 0.5 else np.nan if pd.isnull(x) else 0)
# partners_labs['PCR_B'] = \
#     partners_labs['CRP (MG/L)'].apply(lambda x: 1 if x > 100 else np.nan if pd.isnull(x) else 0)

# partners_labs['ALT_B'] = \
#     partners_labs['Alanine Aminotransferase (ALT)'].apply(lambda x: 1 if x >= 40 else np.nan if pd.isnull(x) else 0)
# partners_labs['AST_B'] = \
#     partners_labs['Aspartate Aminotransferase (AST)'].apply(lambda x: 1 if x >= 40 else np.nan if pd.isnull(x) else 0)
# partners_labs['TRANSAMINASES_B'] = partners_labs[['ALT_B', 'AST_B']].max(axis=1)
    
# partners_labs['LDL_B'] = \
#     partners_labs['LDH'].apply(lambda x: 1 if x < 240 or x > 480 else np.nan if pd.isnull(x) else 0)

# partners_labs['Lab_Data'] = 1
# partners_join = demographics_clean.join(partners_labs, how = 'left')
# partners_join[['TotalLOSMin','Outcome','ICU_Admission','Lab_Data']].to_csv(save_path+'lab_indicator.csv')
# encs_nolabs = partners_join.index[partners_join.Lab_Data != 1]

## See if other encounters with labs for these patients
# dems_nolabs = demographics_clean.loc[demographics_clean.index.isin(encs_nolabs),:]
# dems_nolabs['Any_Labs'] = 0
# dems_nolabs.loc[dems_nolabs.index.isin(labs_raw['pdgID']),'Any_Labs']=1
# dems_nolabs[['pdgID','Any_Labs']].to_csv(save_path+'encounters_withoutlabs.csv')
# labs_raw.loc[labs_raw.index.isin(dems_nolabs.index),:].index.unique()

# partners_labs.describe()


#%%  Process vitals
vitals_raw = pd.read_csv(data_path+'demographics_redone/vitals.csv')
vitals = pd.merge(vitals_raw, encounter_recode, how = 'left', on = 'hospitalEncounterHash')
vitals = vitals.loc[~vitals['EncounterID'].isnull(),:]
vitals.loc[:, 'FlowsheetMeasureNM'] = vitals.loc[:, 'FlowsheetMeasureNM'].apply(lambda x: u.PARTNERS_VITALS[x])
vitals = vitals.sort_values(['pdgID', 'recordedDTSMinPostAdmission'], ascending = [True, True])
vitals = vitals.drop_duplicates(['pdgID', 'FlowsheetMeasureNM']).reset_index(drop = True)

partners_vitals = pd.DataFrame(np.NaN, index = vitals['pdgID'].unique(), columns = u.PARTNERS_VITALS.values())
for patient in vitals['pdgID'].unique():
    for exam in vitals['FlowsheetMeasureNM'].unique():
            value = vitals.loc[(vitals['pdgID'] == patient) & (vitals['FlowsheetMeasureNM'] == exam), 'MeasureTXT'].values
            if len(value) == 0:
                continue
            else:
                try:
                    partners_vitals.loc[patient, exam] = float(value)
                except:
                    partners_vitals.loc[patient, exam] = value

# Clean data for consistency                    
partners_vitals['Systolic Blood Pressure'] = partners_vitals['Systolic Blood Pressure'].apply(lambda x: float(x.split('/')[0]))
partners_vitals['MAXTEMPERATURE_ADMISSION'] = (partners_vitals['Body Temperature']-32)*5/9 # for now, assume that only one temperature is available at beginning of admission
# partners_vitals['BLOOD_PRESSURE_ABNORMAL_B'] = partners_vitals['Systolic Blood Pressure'].apply(lambda x: 1 if x < 100 else np.nan if pd.isnull(x) else 0)
# partners_vitals['FAST_BREATHING'] = partners_vitals['Respiratory Frequency'].apply(lambda x: 1 if x > 22 else np.nan if pd.isnull(x) else 0)
                                        
#%% Process previous diagnoses from medical history and problem list

## Load medical history
medhx_all = pd.read_csv(data_path+'medicalhistory_additional/medical_history.csv')
medhx_all = medhx_all[['pdgID', 'hospitalEncounterHash', 'firstContactDaysFromAdmission',
       'lastContactDaysFromAdmission', 'CurrentICD10ListTXT']].rename(
           {'firstContactDaysFromAdmission':'startDate', 'lastContactDaysFromAdmission':'endDate'}, axis=1)
# Status is active if contact with condition within past year, else inactive
medhx_all['Status'] = medhx_all['endDate'].apply(lambda x: 'Resolved' if x < -365 else 'Active')
medhx_all = pd.merge(medhx_all, encounter_recode, how = 'left', on = 'hospitalEncounterHash')
medhx_all = medhx_all.loc[~medhx_all['EncounterID'].isnull(),:]
medhx_all['Source'] = 'Medical History'

## Load problem list
problems_all = pd.read_csv(data_path+'medicalhistory_additional/problem_list.csv')
problems_all = problems_all[['pdgID', 'hospitalEncounterHash', 'firstDiagnosedDaysFromAdmission',
       'resolvedDaysFromAdmission', 'ProblemStatusDSC', 'CurrentICD10ListTXT']].rename(
           {'firstDiagnosedDaysFromAdmission':'startDate', 'resolvedDaysFromAdmission':'endDate',
            'ProblemStatusDSC':'Status'}, axis=1)
problems_all = pd.merge(problems_all, encounter_recode, how = 'left', on = 'hospitalEncounterHash')
problems_all = problems_all.loc[~problems_all['EncounterID'].isnull(),:]
problems_all['Source'] = 'Problem List'

dx_all = pd.concat([medhx_all, problems_all], axis = 0, ignore_index = False)
dx_all['CurrentICD10ListTXT'] = dx_all['CurrentICD10ListTXT'].apply(lambda x: x.replace('.', ''))
icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd10.csv')
dx_all = dx_all.merge(icd_dict[['DIAGNOSIS_CODE', 'HCUP_ORDER', 'GROUP_HCUP']], how = 'left', left_on = 'CurrentICD10ListTXT', right_on = 'DIAGNOSIS_CODE')

## Re-index all start/end dates with respect to first encounter (e.g. if 3 encounters and startDate = -10 for third encounter, must adjust for when first encounter was)
dx_all = dx_all.merge(partners_demographics_byenc[['pdgID', 'hospitalEncounterHash', 'minFromFirstHospitalization', 'EncounterID']], how = 'left')
dx_all['daysFromFirstHospitalization'] = np.round(dx_all['minFromFirstHospitalization']/(60*24))
dx_all['startDate_adj'] = dx_all['startDate'] + dx_all['daysFromFirstHospitalization']  ## subtract time between visits
dx_all['endDate_adj'] =  dx_all['endDate'] + dx_all['daysFromFirstHospitalization']
dx_all = dx_all.groupby(['pdgID', 'CurrentICD10ListTXT', 'Source',
       'DIAGNOSIS_CODE', 'HCUP_ORDER', 'GROUP_HCUP']).last().reset_index()

compare = dx_all[['pdgID','hospitalEncounterHash','GROUP_HCUP','daysFromFirstHospitalization','EncounterID','startDate','endDate','startDate_adj','endDate_adj']]


#%% Identify comorbidities (pre-existing and active)

dx_previous = dx_all.query('startDate_adj <=0 & Status == "Active"')
partners_comorbs = pd.DataFrame(0, index = demographics_clean.index, columns = uc.COMORBS_TREATMENTS_NAMES)
for i in range(len(uc.COMORBS_TREATMENTS_HCUP)):
    name = uc.COMORBS_TREATMENTS_NAMES[i]
    hcups = uc.COMORBS_TREATMENTS_HCUP[i]
    for j in partners_comorbs.index:
        partners_comorbs.loc[j, name] = int(sum(dx_previous.loc[dx_previous['pdgID'] == j, 'HCUP_ORDER'].isin(hcups)) > 0)


#%% Identify morbidities (new within encounter)

dx_new = dx_all.query('startDate_adj >= 0')

MORB_CODES = {'SEPSIS': [2],
    'ARF': [145], #ACUTE RENAL FAILURE
    'HF': [97],
    'EMBOLIC': [105]}

partners_morbidities = pd.DataFrame(0, index = demographics_clean.index, columns = MORB_CODES.keys())
for i in MORB_CODES.keys():
    name = i
    hcups = MORB_CODES[i]
    for j in partners_morbidities.index:
        partners_morbidities.loc[j, name] = int(sum(dx_new.loc[dx_new['pdgID'] == j, 'HCUP_ORDER'].isin(hcups)) > 0)
        
partners_morbidities.loc[:,'OUTCOME_VENT'] = np.nan

#%% Process medications
# Load the Medication ID
treatments = pd.read_csv(data_path+'medications_redone/hospital_meds.csv')
# treatments = pd.concat([treatments_rem, treatments_nonrem], axis = 0)
# treatments_home = pd.read_csv(data_path+'medications_redone/home_meds.csv')
treatments =  pd.merge(treatments, encounter_recode, how = 'left', on = 'hospitalEncounterHash')
treatments = treatments.loc[~treatments['EncounterID'].isnull(),:].drop_duplicates()

# treatments = treatments.merge(medid[['MedicationID', 'AHFSCD']], on = 'MedicationID')

treatments['ClinicalTrial'] = treatments['AIMS_Med_Name'].str.lower().str.contains('placebo')
treatments = treatments.loc[~treatments['ClinicalTrial'],:]

def parse_treatment(AIMS_Med_Name, therapeuticClassDSC, pharmaceuticalClassDsc):
    try:
        if AIMS_Med_Name.upper().find('REMDESIVIR') != -1:
            med_class = 'REMDESIVIR'
        elif therapeuticClassDSC.upper() in ['ANTICOAGULANTS','ANTIBIOTICS']:
            med_class = therapeuticClassDSC
        elif therapeuticClassDSC.upper() in ['ANTIVIRALS']:
            med_class = 'ANTIVIRAL'
        elif pharmaceuticalClassDsc.upper() in ['GLUCOCORTICOIDS']:
            med_class = 'CORTICOSTEROIDS'
        elif pharmaceuticalClassDsc.upper() in ['ANTIHYPERTENSIVES, ANGIOTENSIN RECEPTOR ANTAGONIST']:
            med_class = 'ACEI_ARBS'
        elif AIMS_Med_Name.upper().find('HYDROXYCHLOROQUINE') != -1:
            med_class = 'CLOROQUINE'
        else:
            med_class = np.nan
        return med_class
    except: 
        print(AIMS_Med_Name)
        print(therapeuticClassDSC)
        print(pharmaceuticalClassDsc)

treatments['Med_Class'] = treatments.apply(lambda row: parse_treatment(row['AIMS_Med_Name'], row['therapeuticClassDSC'], row['pharmaceuticalClassDsc']), axis = 1)

treatments.query('Med_Class == "REMDESIVIR"').AIMS_Med_Name

u.TREATMENTS_NAME = u.TREATMENTS_NAME + ['REMDESIVIR']

partners_treatments = pd.DataFrame(0, index = treatments['pdgID'].unique(), columns = u.TREATMENTS_NAME)
for j in partners_treatments.index:
    treat_list = treatments.loc[treatments['pdgID'] == j, 'Med_Class']
    for name in u.TREATMENTS_NAME:
        if name == 'ANTIVIRAL':
            if np.any(treat_list==name) | np.any(treat_list=='REMDESIVIR'):
                partners_treatments.loc[j, name] = 1
        elif np.any(treat_list==name):
            partners_treatments.loc[j, name] = 1


# partners_treatments = pd.DataFrame(0, index = treatments['pdgID'].unique(), columns = u.IN_TREATMENTS_NAME + u.TREATMENTS_NAME)
# # for i in range(1, len(u.IN_TREATMENTS)): #oxygen is not available, so we start from 1
# #     name = u.IN_TREATMENTS_NAME[i]
# #     treat = u.IN_TREATMENTS[i]
# #     for j in partners_treatments.index:
# #         partners_treatments.loc[j, name] = int(sum(treatments.loc[treatments['pdgID'] == j, 'AHFSCD'].apply(lambda x: uc.check_treatment(treat, x))) > 0)

# partners_treatments.loc[:, 'HOME_OXIGEN_THERAPY'] = np.NaN
# partners_treatments.mean(axis=0)

#%% Create final dataframe and binarize variables
# Patients must have demographics, labs, vitals available
encs = list(set(demographics_clean.index).intersection(set(partners_labs.index)).intersection(partners_vitals.index))
demographics_clean = demographics_clean.reindex(encs)
partners_labs = partners_labs.reindex(encs)
partners_vitals = partners_vitals.reindex(encs)
partners_comorbs = partners_comorbs.reindex(encs)
partners_morbidities = partners_morbidities.reindex(encs)
partners_treatments = partners_treatments.reindex(encs, fill_value = 0)

partners_all = pd.concat([demographics_clean, partners_labs, 
                          partners_vitals, partners_comorbs, partners_treatments, partners_morbidities],
                         axis = 1, ignore_index = False)

ft_summary = partners_all.describe().transpose()
ft_summary.to_csv(save_path+'feature_summary.csv')

partners_all.to_csv('../../covid19_treatments_data/partners_treatment_missing_'+version+'.csv', index = False)

#%% Impute missing data
partners_clean = partners_all.drop(['hospitalEncounterHash','GENDER','RACE',
                                    'TotalLOSMin','DEATH','ICU_Admission']+u.MORBIDITIES, axis=1)

## coalesce ABG and SaO2 readings given high missingness - better than imputation
partners_clean.loc[:, 'ABG: Oxygen Saturation (SaO2)'] = partners_clean['ABG: Oxygen Saturation (SaO2)'].fillna(partners_clean['SaO2'])
partners_clean.drop('SaO2', axis=1, inplace=True)

imp = KNNImputer(n_neighbors=10, weights="uniform")
partners_fit = imp.fit_transform(partners_clean)
partners_fit = pd.DataFrame(partners_fit, columns = partners_clean.columns, index = partners_clean.index)
partners_imputed = pd.concat([partners_fit, 
                               partners_all[['hospitalEncounterHash','GENDER','RACE',
                                    'TotalLOSMin','DEATH','ICU_Admission']+u.MORBIDITIES]], axis = 1)

#%% Create binary variables



partners_imputed['SAT02_BELOW92'] = \
    partners_imputed['ABG: Oxygen Saturation (SaO2)'].apply(lambda x: 1 if x < 92 else np.nan if pd.isnull(x) else 0)
# partners_imputed['DDDIMER_B'] = \
#     partners_imputed['D-DIMER (UG/ML)'].apply(lambda x: 1 if x > 0.5 else np.nan if pd.isnull(x) else 0)
partners_imputed['DDDIMER_B'] = \
    partners_imputed['D-DIMER'].apply(lambda x: 1 if x > 500 else np.nan if pd.isnull(x) else 0)
partners_imputed['PROCALCITONIN_B'] = \
    partners_imputed['PROCALCITONIN'].apply(lambda x: 1 if x > 0.5 else np.nan if pd.isnull(x) else 0)
# partners_imputed['PCR_B'] = \
#     partners_imputed['CRP (MG/L)'].apply(lambda x: 1 if x > 100 else np.nan if pd.isnull(x) else 0)
partners_imputed['PCR_B'] = \
    partners_imputed['C-Reactive Protein (CRP)'].apply(lambda x: 1 if x > 10 else np.nan if pd.isnull(x) else 0)

partners_imputed['ALT_B'] = \
    partners_imputed['Alanine Aminotransferase (ALT)'].apply(lambda x: 1 if x >= 40 else np.nan if pd.isnull(x) else 0)
partners_imputed['AST_B'] = \
    partners_imputed['Aspartate Aminotransferase (AST)'].apply(lambda x: 1 if x >= 40 else np.nan if pd.isnull(x) else 0)
partners_imputed['TRANSAMINASES_B'] = partners_imputed[['ALT_B', 'AST_B']].max(axis=1)
    
partners_imputed['LDL_B'] = \
    partners_imputed['LDH'].apply(lambda x: 1 if x < 240 or x > 480 else np.nan if pd.isnull(x) else 0)

partners_imputed['BLOOD_PRESSURE_ABNORMAL_B'] = partners_imputed['Systolic Blood Pressure'].apply(lambda x: 1 if x < 100 else np.nan if pd.isnull(x) else 0)
partners_imputed['FAST_BREATHING'] = partners_imputed['Respiratory Frequency'].apply(lambda x: 1 if x > 22 else np.nan if pd.isnull(x) else 0)



#%% Match format of other dataframes

partners_imputed.loc[:, 'SOURCE'] = 'PARTNERS'
partners_imputed.loc[:, 'SOURCE_COUNTRY'] = 'USA'
partners_imputed.loc[:, 'HOSPITAL'] = 'PARTNERS'
partners_imputed.loc[:, 'COUNTRY'] = 'USA'
partners_imputed.loc[:, 'DT_HOSPITAL_ADMISSION'] = np.nan
partners_imputed.loc[:, 'COMORB_DEATH'] = partners_imputed[['DEATH']+u.MORBIDITIES].max(axis=1)
partners_imputed.loc[:, 'REGIMEN'] = partners_imputed.apply(lambda row: uc.get_regimen(row['CLOROQUINE'], row['ANTIVIRAL'], row['ANTICOAGULANTS']), axis = 1)


df_compare = pd.read_csv('../../covid19_treatments_data/matched_all_treatments_der_val_update_addl_outcomes/hope_hm_cremona_all_treatments_validation_addl_outcomes.csv')
set(df_compare.columns).difference(partners_imputed.columns)
set(u.COLS_TREATMENTS).difference(partners_imputed.columns)

#Replace the ABG SaO2 test, which is almost absent, with the vital measurement
# partners_labs.loc[:, 'ABG: Oxygen Saturation (SaO2)'] = partners_vitals.loc[:, 'SaO2']
# partners_vitals = partners_vitals.rename(columns = uc.VITALS_TREAT_RENAME)
# partners_labs = partners_labs.rename(columns = uc.LABS_TREAT_RENAME)

partners_final = partners_imputed[u.COLS_TREATMENTS]

partners_final.to_csv('../../covid19_treatments_data/partners_treatments_'+version+'.csv', index = False)
