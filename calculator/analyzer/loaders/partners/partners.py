import numpy as np
import pandas as pd
import analyzer.loaders.cremona.utils as uc
from analyzer.loaders.partners.utils import COVID_LABELS, PARTNERS_LABS, PARTNERS_VITALS, PARTNERS_COMORBIDITIES, PARTNERS_COMORBS, COLUMNS_WITHOUT_LAB
path = '../../covid19_partners'

partners_demographics_march = pd.read_csv('%s/data/v2/20200301_20200331/demographics.csv' %path)
partners_demographics_april = pd.read_csv('%s/data/v2/20200401_20200430/demographics.csv' %path)
partners_demographics_may = pd.read_csv('%s/data/v2/20200501_20200531/demographics.csv' %path)
partners_demographics_june = pd.read_csv('%s/data/v2/20200601_20200610/demographics.csv' %path)
partners_demographics = pd.concat([partners_demographics_march, partners_demographics_april, partners_demographics_may, partners_demographics_june], axis = 0)
partners_demographics = partners_demographics.loc[partners_demographics['COV2Result'].isin(COVID_LABELS), :]
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
labs['componentNM'] = labs['componentNM'].apply(lambda x: PARTNERS_LABS[x])
labs = labs.sort_values(['pdgID', 'resultDTSMinPostAdmission'], ascending = [True, False])
labs = labs.drop_duplicates(['pdgID', 'componentNM']).reset_index(drop = True)

partners_labs = pd.DataFrame(np.NaN, index = labs['pdgID'].unique(), columns = PARTNERS_LABS.values())

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
partners_labs.loc[partners_labs['CBC: Platelets'] > 800, 'CBC: Platelets'] = np.NaN

vitals_march = pd.read_csv('%s/data/v2/20200301_20200331/vitals.csv' %path)
vitals_april = pd.read_csv('%s/data/v2/20200401_20200430/vitals.csv' %path)
vitals_may = pd.read_csv('%s/data/v2/20200501_20200531/vitals.csv' %path)
vitals_june = pd.read_csv('%s/data/v2/20200601_20200610/vitals.csv' %path)
vitals = pd.concat([vitals_march, vitals_april, vitals_may, vitals_june], axis = 0)
vitals = vitals.loc[vitals['pdgID'].isin(partners_demographics.index), :]
vitals.loc[:, 'FlowsheetMeasureNM'] = vitals.loc[:, 'FlowsheetMeasureNM'].apply(lambda x: PARTNERS_VITALS[x])
vitals = vitals.sort_values(['pdgID', 'recordedDTSMinPostAdmission'], ascending = [True, False])
vitals = vitals.drop_duplicates(['pdgID', 'FlowsheetMeasureNM']).reset_index(drop = True)

partners_vitals = pd.DataFrame(np.NaN, index = vitals['pdgID'].unique(), columns = PARTNERS_VITALS.values())
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


# Load the Medication ID
medid = pd.read_csv('%s/data/v2/medicationid_AHFSCD_link.csv' %path)
medid = medid.drop_duplicates('MedicationID').reset_index(drop=True)
treatments_march = pd.read_csv('%s/data/v2/20200301_20200331/hospital_meds.csv' %path)
treatments_april = pd.read_csv('%s/data/v2/20200401_20200430/hospital_meds.csv' %path)
treatments_may = pd.read_csv('%s/data/v2/20200501_20200531/hospital_meds.csv' %path)
treatments_june = pd.read_csv('%s/data/v2/20200601_20200610/hospital_meds.csv' %path)
treatments = pd.concat([treatments_march, treatments_april, treatments_may, treatments_june], axis=0)
treatments = treatments.loc[treatments['pdgID'].isin(partners_demographics.index), :].reset_index(drop=True)
treatments = treatments.merge(medid[['MedicationID', 'AHFSCD']], on = 'MedicationID')