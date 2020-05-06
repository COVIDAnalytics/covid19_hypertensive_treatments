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


model_type = 'mortality'
model_lab = 'with_lab'

assert model_type in('mortality','infection'), "Invalid outcome"
assert model_lab in('with_lab','without_lab'), "Invalid lab specification"

#%% Set paths
website_path = '/Users/hollywiberg/git/website/'
data_path = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_clean_data/'

#%% Load model corresponding to *model_type* and *model_lab*.

with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
    model_file = pickle.load(file)
    
model = model_file['model']
features = model_file['json']
columns = model_file['columns']
imputer= model_file['imputer']

#%% Load data: to be replaced once we store X_train and X_test. Currently no imputation

data = pd.read_csv(data_path+model_type+"_"+model_lab+"/train.csv")
if model_type == "mortality":
    X = data.drop(["Unnamed: 0", "Outcome"], axis=1, inplace = False)
    y = data["Outcome"]
else:
    X = data.drop(["NOSOLOGICO","Swab"], axis=1, inplace = False)
    y = data["Swab"]
    

#%% Evaluate Model with SHAP
## Calculate SHAP values (for each observation x feature)

explainer = shap.TreeExplainer(model);
shap_values = explainer.shap_values(X);


#%% Summarize SHAP values across all features
# This acts as an alterative to the standard variable importance plots. Higher SHAP values translate to higher probability of mortality.
bp_col = X.columns != "Systolic Blood Pressure"
shap.summary_plot(shap_values[:,bp_col],  X[X.columns[bp_col]],
                  show=False,max_display=100)
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