import numpy as np

admission_cols = ['ONSET_DATE_DIFF', 'TEST_DATE_DIFF']

demographic_cols = ['GENDER', 'RACE', 'AGE']

comorb_cols = ['DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY', 'RENALINSUF',
       'ANYLUNGDISEASE', 'AF', 'VIH', 'ANYHEARTDISEASE',
       'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE',
       'LIVER_DISEASE', 'CANCER']

vital_cols = ['MAXTEMPERATURE_ADMISSION', 'SAT02_BELOW92', 'BLOOD_PRESSURE_ABNORMAL_B']

lab_cols = ['DDDIMER_B', 'PCR_B', 'TRANSAMINASES_B', 'LDL_B', 'CREATININE', 'SODIUM', 'LEUCOCYTES',
       'LYMPHOCYTES', 'HEMOGLOBIN', 'PLATELETS']

med_hx_cols = ['IN_PREVIOUSASPIRIN',
       'IN_OTHERANTIPLATELET', 'IN_ORALANTICOAGL',
       'IN_BETABLOCKERS', 'IN_BETAGONISTINHALED', 'IN_GLUCORTICOIDSINHALED',
       'IN_DVITAMINSUPLEMENT', 'IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT']

other_tx_cols = ['CLOROQUINE','ANTIVIRAL','ANTICOAGULANTS','CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS', 'ACEI_ARBS']

outcome_cols = ['DEATH', 'COMORB_DEATH','OUTCOME_VENT','DEATH','HF','ARF','SEPSIS','EMBOLIC']

excluded_cols = ['COUNTRY','FAST_BREATHING', 'PREGNANT', 
    'HOME_OXIGEN_THERAPY', 'MAINHEARTDISEASE', 'SMOKING', 'PROCALCITONIN_B', 
    'IN_ACEI_ARB', 'TBPASSED']

def create_dataset_treatment(data, treatment = None,
                        demographics = True,
                        comorbidities = True,
                        vitals = True,
                        lab_tests=True,
                        med_hx=True,
                        other_tx=True,
                        prediction = 'DEATH', include_regimen = False):

    cols_include = (demographics*demographic_cols + 
        comorbidities*comorb_cols + vitals*vital_cols + 
        lab_tests*lab_cols  + med_hx*med_hx_cols + other_tx*other_tx_cols)   

    # Remove treatment column of interest - replace NO_ if running negative treatment
    if (treatment != None) & other_tx:
        cols_include.remove(treatment.replace("NO_", ""))
        data_sub = data.loc[data['REGIMEN']==treatment,:]
    else:
        data_sub = data

    missing_cols = list(set(cols_include).difference(data.columns))
    cols_include = [x for x in cols_include if x not in missing_cols]
    
    if len(missing_cols) > 0:
        print("Warning: missing columns "+str(missing_cols))
        
    if include_regimen:
        cols_include.append('REGIMEN')

    return data_sub[cols_include], data_sub[prediction]


# def create_dataset(data_dict, discharge_data = True,
#                         comorbidities_data = True,
#                         vitals_data = True,
#                         lab_tests=True,
#                         demographics_data = False,
#                         swabs_data = False,
#                         prediction='Outcome'):

#     if demographics_data:
#         data = data_dict['demographics']

#     if discharge_data:
#         data = data.join(data_dict['discharge'])

#     if comorbidities_data:
#         data = data.join(data_dict['comorbidities'])

#     if vitals_data:
#         data = data.join(data_dict['vitals'])

#     if lab_tests:
#         data = data.join(data_dict['lab'])

#     if swabs_data:
#         data = data.join(data_dict['swab'])

#     X = data.loc[:, data.columns.difference(['Outcome', 'ICU', 'Swab'])]
#     y = data.loc[:, prediction]

#     X = X.astype(np.float32)
#     y = y.astype(int)

#     return X, y

def filter_outliers(df_X, filter_lb = 0.1, filter_ub = 99.9, o2 = "SaO2"):
    bounds_dict = {}

    for col in df_X:
        try: 
            # lb = max(descr[col].loc['mean'] - 3*descr[col].loc['std'],0)
            # ub = descr[col].loc['mean'] + 3*descr[col].loc['std']
            lb = np.floor(np.nanpercentile(df_X[col], filter_lb))
            ub = np.ceil(np.nanpercentile(df_X[col], filter_ub))
            med = np.round(np.nanpercentile(df_X[col], 50))
            
            if col == o2:
                lb = 75
                ub = 99
    
            bounds_dict[col] = {'min_val': lb,
                'max_val': ub,
                'default': med}
            outlier_inds = (lb > df_X[col]) | (df_X[col] > ub)
            v = sum(outlier_inds)
            print(col+': '+ "LB = "+str(lb)+", UB = "+str(ub)+" (Filter = "+str(v)+")")
    
            df_X[col][outlier_inds] = np.nan
        except: 
            print("Cannot filter column: ", col)


    return df_X, bounds_dict