import numpy as np


def create_dataset(data_dict, discharge_data = True, 
                        comorbidities_data = True, 
                        vitals_data = True, 
                        lab_tests=True, 
                        demographics_data = False, 
                        swabs_data = False,
                        prediction='Outcome'):

    if discharge_data:
        data = data_dict['discharge']
    
    if demographics_data:
        data = data_dict['demographics']

    if comorbidities_data:
        data = data.join(data_dict['comorbidities'])

    if vitals_data:
        data = data.join(data_dict['vitals'])

    if lab_tests:
        data = data.join(data_dict['lab'])

    if swabs_data:
        data = data.join(data_dict['swab'])

    X = data.loc[:, data.columns.difference(['Outcome', 'ICU', 'Swab'])]
    y = data.loc[:, prediction]

    X = X.astype(np.float32)
    y = y.astype(int)

    return X, y

def filter_outliers(df_X, filter_lb = 0.1, filter_ub = 99.9):
    bounds_dict = []
    
    for col in df_X:
        # lb = max(descr[col].loc['mean'] - 3*descr[col].loc['std'],0)
        # ub = descr[col].loc['mean'] + 3*descr[col].loc['std']
        lb = np.floor(np.nanpercentile(df_X[col], filter_lb))
        ub = np.ceil(np.nanpercentile(df_X[col], filter_ub))
        med = np.round(np.nanpercentile(df_X[col], 50))
        
        bounds_dict.append({'name': col,
            'min_val': lb,
            'max_val': ub,
            'default': med})
        outlier_inds = (lb > df_X[col]) | (df_X[col] > ub)
        v = sum(outlier_inds)
        print(col+': '+ "LB = "+str(lb)+", UB = "+str(ub)+" (Filter = "+str(v)+")")
        
        df_X[col][outlier_inds] = np.nan
    
    return df_X, bounds_dict