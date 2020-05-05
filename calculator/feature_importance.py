import numpy as np
import pandas as pd
import os
import pickle

# Other packages
import analyzer.loaders.cremona.utils as u
import analyzer.loaders.cremona as cremona
import analyzer.loaders.hmfundacion.hmfundacion as hmfundacion
from analyzer.utils import store_json, change_SaO2
import analyzer.dataset as ds

import shap
import matplotlib.pyplot as plt

#%% ## Load Model and Data
# Select model type.  
# - **model_type** = 'mortality' or 'infection'
# - **model_lab** = 'with_lab' or 'without_lab'


model_type = 'infection'
model_lab = 'with_lab'

assert model_type in('mortality','infection'), "Invalid outcome"
assert model_lab in('with_lab','without_lab'), "Invalid lab specification"

#%% Set paths
website_path = '/Users/hollywiberg/git/website/'
path_cremona = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_cremona/data/'
path_hm = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_hmfoundation/'

#%% Load model corresponding to *model_type* and *model_lab*.

with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
    model_file = pickle.load(file)
    
model = model_file['model']
features = model_file['json']
columns = model_file['columns']
imputer= model_file['imputer']

#%% Load data: to be replaced once we store X_train and X_test. Currently no imputation


SEED = 1

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

X = pd.DataFrame(imputer.transform(X0))
X.columns =  X0.columns


#%% Evaluate Model with SHAP
## Calculate SHAP values (for each observation x feature)

explainer = shap.TreeExplainer(model);
shap_values = explainer.shap_values(X);


#%% Summarize SHAP values across all features
# This acts as an alterative to the standard variable importance plots. Higher SHAP values translate to higher probability of mortality.

shap.summary_plot(shap_values, X,show=False)
f = plt.gcf()
f.savefig('../results/'+model_type+'/model_'+model_lab+'/summary_plot.png', bbox_inches='tight')
 
#%% Deep-dive into individual features
# For a given feature, see how the SHAP varies across its possible values. 
# The interaction_index lets you choose a secondary index to visualize.
# If omitted, it will automatically find the variable with the highest interaction.

# shap.dependence_plot("Systolic Blood Pressure", shap_values, X, interaction_index = "Age")
for i in X.columns:
    shap.dependence_plot(i, shap_values, X,show=False)
    f = plt.gcf()
    f.savefig('../results/'+model_type+'/model_'+model_lab+'/dependence_plot_'+i+'.png', bbox_inches='tight')


#%% Evaluate drivers of individual predictions 
# Select index j for prediction to generate.

# shap.force_plot(explainer.expected_value, shap_values[1:10,:], X.iloc[[1:10,:]])
j=20
plot = shap.force_plot(explainer.expected_value, shap_values[j], X.iloc[[j]] , link="logit")
plot

#%%  As an alternative view, you can trace a 3D plot of the values and their impact on the prediction.
# shap.decision_plot(explainer.expected_value, shap_values[j], X.iloc[[j]], link='logit')