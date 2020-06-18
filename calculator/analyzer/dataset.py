import numpy as np

admission_cols = ['ONSET_DATE_DIFF', 'TEST_DATE_DIFF']

demographic_cols = ['COUNTRY', 'GENDER', 'RACE', 'PREGNANT', 'AGE']

comorb_cols = ['DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY', 'SMOKING', 'RENALINSUF',
       'ANYLUNGDISEASE', 'AF', 'VIH', 'TBPASSED', 'ANYHEARTDISEASE',
       'MAINHEARTDISEASE', 'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE',
       'LIVER_DISEASE', 'CANCER']

vital_cols = ['FAST_BREATHING', 'MAXTEMPERATURE_ADMISSION', 'SAT02_BELOW92', 'BLOOD_PRESSURE_ABNORMAL_B']

lab_cols = ['DDDIMER_B', 'PROCALCITONIN_B', 'PCR_B', 'TRANSAMINASES_B', 'LDL_B', 'CREATININE', 'SODIUM', 'LEUCOCYTES',
       'LYMPHOCYTES', 'HEMOGLOBIN', 'PLATELETS']

med_hx_cols = ['HOME_OXIGEN_THERAPY', 'IN_PREVIOUSASPIRIN',
       'IN_OTHERANTIPLATELET', 'IN_ORALANTICOAGL', 'IN_ACEI_ARB',
       'IN_BETABLOCKERS', 'IN_BETAGONISTINHALED', 'IN_GLUCORTICOIDSINHALED',
       'IN_DVITAMINSUPLEMENT', 'IN_BENZODIACEPINES', 'IN_ANTIDEPRESSANT']

other_tx_cols = ['CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB', 'ANTIBIOTICS', 'ACEI_ARBS']

outcome_cols = ['DEATH', 'COMORB_DEATH']

def create_dataset_treatment(data, treatment,
                        demographics = True,
                        comorbidities = True,
                        vitals = True,
                        lab_tests=True,
                        med_hx=True,
                        other_tx=True,
                        prediction = 'DEATH'):
    
    data['REGIMEN'] = data['REGIMEN'].map(lambda x: x.replace(" ", "_"))
    cols_include = (demographics*demographic_cols + 
        comorbidities*comorb_cols + vitals*vital_cols + 
        lab_tests*lab_cols  + med_hx*med_hx_cols + other_tx*other_tx_cols)
    
    if treatment == 'All':
        cols_include.append('REGIMEN')
        data_sub = data
    else: 
        data_sub = data.loc[data['REGIMEN']==treatment,]

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