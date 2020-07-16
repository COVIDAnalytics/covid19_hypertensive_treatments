import pandas as pd 
import numpy as np

COVID_LABELS = ['Positive for 2019-novel Coronavirus (2019-nCoV) by PCR.', 
                'Positive', 'Detected', 'SARS-CoV-2 detected', 
                'Presumptive positive for SARS-Cov-2 (COVID-19) by PCR']

PARTNERS_LABS = {'HGB': 'CBC: Hemoglobin', 
                'MCV': 'CBC: Mean Corpuscular Volume (MCV)', 
                'PLT': 'CBC: Platelets', 
                'WBC': 'CBC: Leukocytes',
                'C REACTIVE PROTEIN': 'C-Reactive Protein (CRP)', 
                'ALT': 'Alanine Aminotransferase (ALT)', 
                'AST': 'Aspartate Aminotransferase (AST)', 
                'BUN': 'Blood Urea Nitrogen (BUN)',
                'CREATININE': 'Blood Creatinine',
                'GLUCOSE': 'Glycemia', 
                'POTASSIUM': 'Potassium Blood Level', 
                'SODIUM': 'Blood Sodium', 
                'INR': 'Prothrombin Time (INR)', 
                'O2 SAT (SO2, ARTERIAL)': 'ABG: Oxygen Saturation (SaO2)',
                'ABSOLUTE LYMPHS': 'ABSOLUTE LYMPHS',
                'CALCIUM': 'CALCIUM',
                'CREATINE KINASE': 'CREATINE KINASE', 
                'D-DIMER': 'D-DIMER',
                'D-DIMER (UG/ML)': 'D-DIMER (UG/ML)', 
                'HGB (BG)': 'HGB (BG)',
                'LDH': 'LDH', 
                'ALBUMIN': 'ALBUMIN',
                'PROCALCITONIN': 'PROCALCITONIN', 
                'COVID-19 SOURCE': 'COVID-19 SOURCE', 
                'SARS-COV 2 (COVID-19) PCR': 'SARS-COV 2 (COVID-19) PCR',
                'SPECIMEN SOURCE/DESCRIPTION': 'SPECIMEN SOURCE/DESCRIPTION',
                'SPECIMEN SOURCE': 'SPECIMEN SOURCE', 
                'CRP (MG/L)': 'CRP (MG/L)',
                'SARS-COV-2 - EXTERNAL': 'SARS-COV-2 - EXTERNAL'}


PARTNERS_VITALS = {'TEMPERATURE': 'Body Temperature',
                    'PULSE': 'Cardiac Frequency',
                    'PULSE OXIMETRY': 'SaO2',
                    'BLOOD PRESSURE': 'Systolic Blood Pressure',
                    'RESPIRATIONS': 'Respiratory Frequency',
                    'HEIGHT': 'HEIGHT',
                    'WEIGHT/SCALE': 'WEIGHT/SCALE'}
 
HCUP_LIST = [49,50,87,90,95,146]

PARTNERS_COMORBIDITIES =  ['Cardiac dysrhythmias',
                'Chronic kidney disease',
                'Coronary atherosclerosis and other heart disease',
                'Diabetes']

PARTNERS_COMORBS = {'Diabetes mellitus without complication': 'Diabetes',
                    'Diabetes mellitus with complications': 'Diabetes',
                    'Cardiac dysrhythmias': 'Cardiac dysrhythmias',
                    'Chronic kidney disease': 'Chronic kidney disease',
                    'Essential hypertension': 'Essential hypertension',
                    'Coronary atherosclerosis and other heart disease': 'Coronary atherosclerosis and other heart disease'}

COLUMNS_WITHOUT_LAB = ['Age', 'Gender', 'Body Temperature', 'SaO2',
                        'Cardiac Frequency','Cardiac dysrhythmias',
                        'Chronic kidney disease',
                        'Coronary atherosclerosis and other heart disease', 'Diabetes']

# HOPE TREATMENTS

IN_TREATMENTS_NAME = ['HOME_OXIGEN_THERAPY', 'IN_ACEI_ARB', 'IN_BETABLOCKERS', 'IN_BETAGONISTINHALED',
                'IN_GLUCORTICOIDSINHALED', 'IN_DVITAMINSUPLEMENT', 'IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT']

IN_TREATMENTS_LONG_NAME = ['IN_PREVIOUSASPIRIN', 'IN_OTHERANTIPLATELET', 'IN_ORALANTICOAGL']

HOME_OXIGEN_THERAPY = np.NaN # 'V03AN01'
IN_PREVIOUSASPIRIN = ['20:12.14', '20:12.18'] #['N02BA', 'B01AC06']
IN_OTHERANTIPLATELET = ['20:12.14', '20:12.18']
IN_ORALANTICOAGL = '20:12.04' # ['B01AA', 'B01AE', 'B01AF']
IN_ACEI_ARB = '24:32.08'  # 'C09'
IN_BETABLOCKERS = '24:24' # 'C07A'
IN_BETAGONISTINHALED = '12:12.08' # 'R03AC'
IN_GLUCORTICOIDSINHALED = '48:10.08.08' #'R03BA'
IN_DVITAMINSUPLEMENT = '88:16'
IN_BENZODIACEPINES = '28:12.08' # N05B
IN_ANTIDEPRESSANT = '28:16.04' # N06A

IN_TREATMENTS = [HOME_OXIGEN_THERAPY, IN_ACEI_ARB, IN_BETABLOCKERS, IN_BETAGONISTINHALED,
                IN_GLUCORTICOIDSINHALED, IN_DVITAMINSUPLEMENT, IN_BENZODIACEPINES, IN_ANTIDEPRESSANT]

IN_TREATMENTS_LONG = [IN_PREVIOUSASPIRIN, IN_OTHERANTIPLATELET, IN_ORALANTICOAGL]