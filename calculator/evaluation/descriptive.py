import numpy as np
import pandas as pd
import pickle

import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
import analyzer.dataset as ds


#%% Load function
def get_dataset(model_type, model_lab, columns, imputer, impute = False, path_cremona = '../data/cremona/', path_hm = '../data/spain/'):
    
    ## Variables determined by input
    prediction = 'Outcome'if model_type == 'mortality' else 'Swab'
    swabs_data = False if model_type == 'mortality' else True
    comorbidities_data = True if model_type == 'mortality' else False
    discharge_data = True if model_type == 'mortality' else False 
    lab_tests = True if model_lab == 'with_lab' else False

    # ## Constant variables
    extra_data = False
    demographics_data = True
    vitals_data = True

    name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])

    ## Load Cremona data
    data = cremona.load_cremona(path_cremona, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
    X_cremona, y_cremona = ds.create_dataset(data, discharge_data, comorbidities_data, vitals_data,
                                      lab_tests, demographics_data, swabs_data, prediction = prediction)

    if model_type == "mortality":
        ## Load Spain data
        data_spain = hmfundacion.load_fundacionhm(path_hm, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)
        X_spain, y_spain =  ds.create_dataset(data_spain, discharge_data, comorbidities_data, vitals_data,
                                             lab_tests, demographics_data, swabs_data, prediction = prediction)

        # Merge datasets, filter outliers, match format of stored model
        X0 = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
        y = pd.concat([y_cremona, y_spain], ignore_index=True)
    else: 
        X0, y = X_cremona, y_cremona

    X0, bounds_dict = ds.filter_outliers(X0)
    X0 = X0[columns] 

    if impute: 
        X = pd.DataFrame(imputer.transform(X0))
        X.columns =  X0.columns
    else:
        X = X0
            
    return X, y

def descriptive_table(data, features, title_mapping):

    
    cols_numeric = [i['name'] for i in features['numeric']]
    cols_categoric = data.columns.difference(cols_numeric)
    
    summary_numeric = np.transpose(data[cols_numeric].describe())
    summary_numeric['Type'] = 'Numeric'
    summary_categoric = np.transpose(data[cols_categoric].describe())
    summary_categoric['Type'] = 'Categoric'
    
    summary_full = summary_numeric.append(summary_categoric)
    summary_full['Missing_Pct'] = 1 - summary_full['count']/data.shape[0]
    # summary_full.drop(["count"], axis = 1, inplace = True)
    summary_full.columns = ['Count', 'Mean', 'Standard Deviation', 
                            'Minimum', '25th Percentile', '50th Percentile',
                            '75th Percentile', 'Maximum', 'Type', 'Percent Missing']
    summary_full["Feature"] = summary_full.index
    summary_full["Feature_Recoded"] = summary_full["Feature"].replace(title_mapping, inplace=False)
    final_cols = ['Feature_Recoded', 'Type', 'Count', 'Percent Missing', 'Mean', 'Standard Deviation',
              'Minimum', '25th Percentile', '50th Percentile','75th Percentile', 'Maximum']
    
    return summary_full[final_cols].sort_values(by = ['Type', 'Feature_Recoded'])

def generate_summary(model_type, model_lab, website_path, title_mapping):
    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
            model_file = pickle.load(file)
    
    model = model_file['model']
    features = model_file['json']
    columns = model_file['columns']
    imputer= model_file['imputer']
        
    X, y = get_dataset(model_type, model_lab, columns, imputer, impute = False, 
                       path_cremona = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_cremona/data/',
                       path_hm = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_hmfoundation/')
    
    data = X.copy()
    data['Outcome'] = y
    
    resAll = descriptive_table(data, features, title_mapping)
    resAll['Outcome'] = 'All'
    res1 = descriptive_table(data.query('Outcome == 1'), features, title_mapping)
    res1['Outcome'] = 'Non-survivor' if model_type == 'mortality' else 'Infection'
    res0 = descriptive_table(data.query('Outcome == 0'), features, title_mapping)
    res0['Outcome'] = 'Survivor' if model_type == 'mortality' else 'No Infection'
    
    return pd.concat([resAll, res1, res0])

# def clean_summary(summary):

    
    