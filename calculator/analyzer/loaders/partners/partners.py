import numpy as np
import pandas as pd
import analyzer.loaders.cremona.utils as uc
import analyzer.loaders.partners.utils as u 
import re
path = '../../covid19_partners'

partners_demographics_march = pd.read_csv('%s/data/v2/20200301_20200331/demographics.csv' %path)
partners_demographics_april = pd.read_csv('%s/data/v2/20200401_20200430/demographics.csv' %path)
partners_demographics_may = pd.read_csv('%s/data/v2/20200501_20200531/demographics.csv' %path)
partners_demographics_june = pd.read_csv('%s/data/v2/20200601_20200610/demographics.csv' %path)
partners_demographics = pd.concat([partners_demographics_march, partners_demographics_april, partners_demographics_may, partners_demographics_june], axis = 0)
partners_demographics = partners_demographics.loc[partners_demographics['COV2Result'].isin(u.COVID_LABELS), :]
partners_demographics = partners_demographics.rename(columns = {'age': 'Age', 'gender': 'Gender', 'deathMinPostAdmission': 'Outcome'})
partners_demographics['Gender'] = (partners_demographics['Gender'] == 'Female').astype(int)
partners_demographics['Outcome'] = (partners_demographics['Outcome'].notna()).astype(int)
partners_demographics = partners_demographics.drop('hospitalLOSMin', axis = 1)
partners_demographics = partners_demographics.sort_values(['pdgID', 'Outcome'], ascending = [True, False])
partners_demographics = partners_demographics.drop_duplicates('pdgID').reset_index(drop=True)
partners_demographics = partners_demographics.set_index('pdgID')

labs_march = pd.read_csv('%s/data/v2/20200301_20200331/labs.csv' %path)
labs_april = pd.read_csv('%s/data/v2/20200401_20200430/labs.csv' %path)
labs_may = pd.read_csv('%s/data/v2/20200501_20200531/labs.csv' %path)
labs_june = pd.read_csv('%s/data/v2/20200601_20200610/labs.csv' %path)
labs = pd.concat([labs_march, labs_april, labs_may, labs_june], axis = 0)
labs = labs.loc[labs['pdgID'].isin(partners_demographics.index), :]
labs['componentNM'] = labs['componentNM'].apply(lambda x: u.PARTNERS_LABS[x])
labs = labs.sort_values(['pdgID', 'resultDTSMinPostAdmission'], ascending = [True, False])
labs = labs.drop_duplicates(['pdgID', 'componentNM']).reset_index(drop = True)

partners_labs = pd.DataFrame(np.NaN, index = labs['pdgID'].unique(), columns = u.PARTNERS_LABS.values())

for patient in labs['pdgID'].unique():
    for exam in labs['componentNM'].unique():
            value = labs.loc[(labs['pdgID'] == patient) & (labs['componentNM'] == exam), 'admissionResultValueNBR'].values
            if len(value) == 0:
                continue
            else:
                partners_labs.loc[patient, exam] = value

partners_labs.loc[partners_labs['CBC: Hemoglobin'] > 20, 'CBC: Hemoglobin'] = np.NaN
partners_labs.loc[partners_labs['CBC: Leukocytes'] > 50, 'CBC: Leukocytes'] = np.NaN
partners_labs.loc[partners_labs['CBC: Mean Corpuscular Volume (MCV)'] > 120, 'CBC: Mean Corpuscular Volume (MCV)'] = np.NaN
partners_labs.loc[partners_labs['ABSOLUTE LYMPHS'] > 10, 'ABSOLUTE LYMPHS'] = np.NaN
partners_labs.loc[:, 'ABSOLUTE LYMPHS'] = partners_labs.loc[:, 'ABSOLUTE LYMPHS']*1000
partners_labs.loc[partners_labs['CBC: Platelets'] > 800, 'CBC: Platelets'] = np.NaN
partners_labs.loc[:, 'CBC: Platelets'] = partners_labs.loc[:, 'CBC: Platelets']*1000
partners_labs.loc[:, 'CBC: Leukocytes'] = partners_labs.loc[:, 'CBC: Leukocytes']*1000
partners_labs.loc[:, 'D-DIMER'] = partners_labs.loc[:, 'D-DIMER']/1000
partners_labs.loc[partners_labs['D-DIMER'] > 20, 'D-DIMER'] = np.NaN
partners_labs.loc[partners_labs['Blood Creatinine'] > 10, 'Blood Creatinine'] = np.NaN
partners_labs.loc[partners_labs['PROCALCITONIN'] > 5, 'PROCALCITONIN'] = np.NaN
partners_labs.loc[partners_labs['LDH'] > 1000, 'LDH'] = np.NaN

vitals_march = pd.read_csv('%s/data/v2/20200301_20200331/vitals.csv' %path)
vitals_april = pd.read_csv('%s/data/v2/20200401_20200430/vitals.csv' %path)
vitals_may = pd.read_csv('%s/data/v2/20200501_20200531/vitals.csv' %path)
vitals_june = pd.read_csv('%s/data/v2/20200601_20200610/vitals.csv' %path)
vitals = pd.concat([vitals_march, vitals_april, vitals_may, vitals_june], axis = 0)
vitals = vitals.loc[vitals['pdgID'].isin(partners_demographics.index), :]
vitals.loc[:, 'FlowsheetMeasureNM'] = vitals.loc[:, 'FlowsheetMeasureNM'].apply(lambda x: u.PARTNERS_VITALS[x])
vitals = vitals.sort_values(['pdgID', 'recordedDTSMinPostAdmission'], ascending = [True, False])
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
partners_vitals['Systolic Blood Pressure'] = partners_vitals['Systolic Blood Pressure'].apply(lambda x: float(x.split('/')[0]))

comorbidities_march = pd.read_csv('%s/data/v2/20200301_20200331/medical_history.csv' %path)
comorbidities_april = pd.read_csv('%s/data/v2/20200401_20200430/medical_history.csv' %path)
comorbidities_may = pd.read_csv('%s/data/v2/20200501_20200531/medical_history.csv' %path)
comorbidities_june = pd.read_csv('%s/data/v2/20200601_20200610/medical_history.csv' %path)
comorbidities = pd.concat([comorbidities_march, comorbidities_april, comorbidities_may, comorbidities_june], axis = 0)
comorbidities = comorbidities.loc[comorbidities['pdgID'].isin(partners_demographics.index), :]
comorbidities['CurrentICD9ListTXT'] = comorbidities['CurrentICD9ListTXT'].apply(lambda x: x.replace('.', ''))
icd_dict = pd.read_csv('analyzer/hcup_dictionary_icd9.csv')
comorbidities = comorbidities.merge(icd_dict[['DIAGNOSIS_CODE', 'HCUP_ORDER', 'GROUP_HCUP']], how = 'left', left_on = 'CurrentICD9ListTXT', right_on = 'DIAGNOSIS_CODE')[['pdgID', 'DIAGNOSIS_CODE', 'HCUP_ORDER']]

partners_comorbs = pd.DataFrame(0, index = partners_demographics.index, columns = uc.COMORBS_TREATMENTS_NAMES)
for i in range(len(uc.COMORBS_TREATMENTS_HCUP)):
    name = uc.COMORBS_TREATMENTS_NAMES[i]
    hcups = uc.COMORBS_TREATMENTS_HCUP[i]
    for j in partners_comorbs.index:
        partners_comorbs.loc[j, name] = int(sum(comorbidities.loc[comorbidities['pdgID'] == j, 'HCUP_ORDER'].isin(hcups)) > 0)

patients = list(set(partners_demographics.index).intersection(set(partners_labs.index)).intersection(partners_vitals.index).intersection(partners_comorbs.index))
partners_demographics = partners_demographics.reindex(patients)
partners_labs = partners_labs.reindex(patients)
partners_vitals = partners_vitals.reindex(patients)
partners_comorbs = partners_comorbs.reindex(patients)

#Replace the ABG SaO2 test, which is almost absent, with the vital measurement
partners_labs.loc[:, 'ABG: Oxygen Saturation (SaO2)'] = partners_vitals.loc[:, 'SaO2']
partners_vitals = partners_vitals.rename(columns = uc.VITALS_TREAT_RENAME)
partners_labs = partners_labs.rename(columns = uc.LABS_TREAT_RENAME)

# Load the Medication ID
medid = pd.read_csv('%s/data/v2/medicationid_AHFSCD_link.csv' %path)
# medid = medid.drop_duplicates('MedicationID').reset_index(drop=True)
treatments_march = pd.read_csv('%s/data/v2/20200301_20200331/hospital_meds.csv' %path)
treatments_april = pd.read_csv('%s/data/v2/20200401_20200430/hospital_meds.csv' %path)
treatments_may = pd.read_csv('%s/data/v2/20200501_20200531/hospital_meds.csv' %path)
treatments_june = pd.read_csv('%s/data/v2/20200601_20200610/hospital_meds.csv' %path)
treatments = pd.concat([treatments_march, treatments_april, treatments_may, treatments_june], axis=0)
treatments = treatments.loc[treatments['pdgID'].isin(partners_demographics.index), :].reset_index(drop=True)
treatments = treatments.merge(medid[['MedicationID', 'AHFSCD']], on = 'MedicationID')

partners_treatments = pd.DataFrame(0, index = patients, columns = u.IN_TREATMENTS_NAME + u.TREATMENTS_NAME)
for i in range(1, len(u.IN_TREATMENTS)): #oxygen is not available, so we start from 1
    name = u.IN_TREATMENTS_NAME[i]
    treat = u.IN_TREATMENTS[i]
    for j in patients:
        partners_treatments.loc[j, name] = int(sum(treatments.loc[treatments['pdgID'] == j, 'AHFSCD'].apply(lambda x: uc.check_treatment(treat, x))) > 0)

for i in range(len(u.TREATMENTS)):
    name = u.TREATMENTS_NAME[i]
    treat = u.TREATMENTS[i]
    for j in patients:
        partners_treatments.loc[j, name] = int(sum(treatments.loc[treatments['pdgID'] == j, 'AHFSCD'].apply(lambda x: uc.check_treatment(treat, x))) > 0)

partners_treatments.loc[:, 'HOME_OXIGEN_THERAPY'] = np.NaN

partners_final = pd.DataFrame(np.NaN, index = patients, columns = uc.COLS_TREATMENTS)
partners_final.loc[:, 'HOSPITAL'] = 'PARTNERS'
partners_final.loc[:, 'COUNTRY'] = 'USA'
partners_final.loc[partners_demographics['Gender'] == 1, 'GENDER'] = 'FEMALE'
partners_final.loc[partners_demographics['Gender'] == 0, 'GENDER'] = 'MALE'
partners_final.loc[:, 'RACE'] = partners_demographics.loc[:, 'ethnicity']
partners_final.loc[:, 'AGE'] = partners_demographics.loc[:, 'Age']
partners_final.loc[:, ['DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA',
       'OBESITY', 'RENALINSUF', 'ANYLUNGDISEASE', 'AF', 'VIH',
       'ANYHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE',
       'CONECTIVEDISEASE', 'LIVER_DISEASE', 'CANCER']] = partners_comorbs.loc[:, ['DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA',
                                                                            'OBESITY', 'RENALINSUF', 'ANYLUNGDISEASE', 
                                                                            'AF', 'VIH', 'ANYHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE',
                                                                            'CONECTIVEDISEASE', 'LIVER_DISEASE', 'CANCER']]
partners_final[u.IN_TREATMENTS_NAME + u.TREATMENTS_NAME] = partners_treatments[u.IN_TREATMENTS_NAME + u.TREATMENTS_NAME]

partners_final.loc[partners_labs['SAT02_BELOW92'].notna(), 'SAT02_BELOW92'] = (partners_labs.loc[partners_labs['SAT02_BELOW92'].notna(), 'SAT02_BELOW92'] < 92).astype(int)
partners_final.loc[partners_labs['D-DIMER'].notna(), 'DDDIMER_B'] = (partners_labs.loc[partners_labs['D-DIMER'].notna(), 'D-DIMER'] > 0.5).astype(int)
partners_final.loc[partners_labs['PROCALCITONIN'].notna(), 'PROCALCITONIN_B'] = (partners_labs.loc[partners_labs['PROCALCITONIN'].notna(), 'PROCALCITONIN'] > 0.5).astype(int)
partners_final.loc[partners_labs['PCR_B'].notna(), 'PCR_B'] = (partners_labs.loc[partners_labs['PCR_B'].notna(), 'PCR_B'] > 100).astype(int)
partners_final.loc[partners_labs['TRANSAMINASES_B'].notna(), 'TRANSAMINASES_B'] = (partners_labs.loc[partners_labs['TRANSAMINASES_B'].notna(), 'TRANSAMINASES_B'] > 40).astype(int)
partners_final.loc[partners_labs['LDH'].notna(), 'LDL_B'] = ((partners_labs.loc[partners_labs['LDH'].notna(), 'LDH'] < 240)|(partners_labs.loc[partners_labs['LDH'].notna(), 'LDH'] > 480)) .astype(int)
partners_final.loc[:, ['CREATININE', 'SODIUM', 'LEUCOCYTES', 'HEMOGLOBIN', 'PLATELETS']] = partners_labs.loc[:, ['CREATININE', 'SODIUM', 'LEUCOCYTES', 'HEMOGLOBIN', 'PLATELETS']]
partners_final.loc[:, 'LYMPHOCYTES'] = partners_labs.loc[:, 'ABSOLUTE LYMPHS']

partners_final.loc[partners_vitals['FAST_BREATHING'].notna(), 'FAST_BREATHING'] = (partners_vitals.loc[partners_vitals['FAST_BREATHING'].notna(), 'FAST_BREATHING'] > 22).astype(int)
partners_final.loc[partners_vitals['BLOOD_PRESSURE_ABNORMAL_B'].notna(), 'BLOOD_PRESSURE_ABNORMAL_B'] = (partners_vitals.loc[partners_vitals['BLOOD_PRESSURE_ABNORMAL_B'].notna(), 'BLOOD_PRESSURE_ABNORMAL_B'] < 100).astype(int)
partners_final.loc[:, 'MAXTEMPERATURE_ADMISSION'] = partners_vitals.loc[:, 'MAXTEMPERATURE_ADMISSION']

partners_final['REGIMEN'] = partners_final.apply(lambda row: uc.get_regimen(row['CLOROQUINE'], row['ANTIVIRAL'], row['ANTICOAGULANTS']), axis = 1)
partners_final.loc[:, 'DEATH'] = partners_demographics.loc[:, 'Outcome']

# Fill COMORB_DEATH
for j in patients:
    partners_final.loc[j, 'COMORB_DEATH'] = int(sum(comorbidities.loc[comorbidities['pdgID'] == j, 'HCUP_ORDER'].isin(uc.COMORB_DEATH)) > 0)
partners_final.loc[:, 'COMORB_DEATH'] = partners_final.apply(lambda row: max(row['DEATH'], row['COMORB_DEATH']), axis = 1)

partners_final.to_csv('/Users/lucamingardi/Dropbox (MIT)/covid19_treatments/partners_treatments.csv', index = False)