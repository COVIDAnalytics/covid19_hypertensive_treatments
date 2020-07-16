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