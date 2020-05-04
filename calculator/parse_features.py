import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json

import analyzer.dataset as ds


#%% Load and merge data for mortality
# data = cremona.load_cremona('../data/cremona/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
# data_spain = hmfundacion.load_fundacionhm('../data/spain/', discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)

name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])

extra_data = False
demographics_data = True
discharge_data = True
comorbidities_data = True
vitals_data = True
lab_tests = True
swabs_data = False

prediction = 'Outcome'

mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
print(name_datasets[mask])

path = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_cremona/data/'
path_hm = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_hmfoundation/'
data = cremona.load_cremona(path, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
data_spain = hmfundacion.load_fundacionhm(path_hm, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, extra_data)


X_cremona, y_cremona = ds.create_dataset(data,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      swabs_data,
                                      prediction = prediction)

X_spain, y_spain =  ds.create_dataset(data_spain,
                                      discharge_data,
                                      comorbidities_data,
                                      vitals_data,
                                      lab_tests,
                                      demographics_data,
                                      extra_data,
                                      prediction = prediction)


# Merge dataset
X_mortality = pd.concat([X_cremona, X_spain], join='inner', ignore_index=True)
X_mortality, bounds_dict_mortality = ds.filter_outliers(X_mortality)


#%% Load data for swabs
discharge_data = False
comorbidities_data = False
vitals_data = True
lab_tests = True
demographics_data = True
swabs_data = True

prediction = 'Swab'

mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
print(name_datasets[mask])


data = cremona.load_cremona(path, discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)

# Create dataset
X, y = ds.create_dataset(data,
                         discharge_data,
                         comorbidities_data,
                         vitals_data,
                         lab_tests,
                         demographics_data,
                         swabs_data,
                         prediction=prediction)

X = X[u.SWAB_WITH_LAB_COLUMNS]
 
X_swab, bounds_dict_swab = ds.filter_outliers(X)

#%% Add explanations to bounds dictionary

feat_exclude = ['Cardiac dysrhythmias',
                'Chronic kidney disease',
                'Coronary atherosclerosis and other heart disease',
                'Diabetes','Essential hypertension', 'Gender']

feature_list = set(X_mortality.columns).union(set(X_swab.columns)).difference(feat_exclude)

explanations = pd.read_csv("explanations.csv")
explanation_dict = dict(zip(list(explanations.Feature), list(explanations.Explanation)))
         
feature_dict = {}
for col in feature_list:
    if col in bounds_dict_mortality.keys():
        feature_dict[col] = [bounds_dict_mortality[col]['min_val'], bounds_dict_mortality[col]['max_val'], explanation_dict[col]]
    else: 
         feature_dict[col] = [bounds_dict_swab[col]['min_val'], bounds_dict_swab[col]['max_val'], explanation_dict[col]]
         
store_json(feature_dict, "feature_information.json")
