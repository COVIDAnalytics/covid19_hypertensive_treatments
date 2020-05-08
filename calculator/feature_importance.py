import numpy as np
import pandas as pd
import os
import pickle
import itertools

# Other packages
import shap
import matplotlib.pyplot as plt

os.chdir('git/covid19_calculator/calculator/')

#%% Model Evaluation plot


#%% Run evaluation for all combinations

## Set paths
website_path = '/Users/hollywiberg/git/website/'
data_path = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_clean_data/'

for model_type, model_lab in itertools.product(['mortality','infection'],['with_lab','without_lab']):
    print("Model: %s, %s" %(model_type, model_lab))
    feature_importance(model_type, model_lab, website_path, data_path)


#%% Evaluate drivers of individual predictions 
# Select index j for prediction to generate.

# shap.force_plot(explainer.expected_value, shap_values[1:10,:], X.iloc[[1:10,:]])
j=20
plot = shap.force_plot(explainer.expected_value, shap_values[j], X.iloc[[j]] , link="logit")
plot

#%%  As an alternative view, you can trace a 3D plot of the values and their impact on the prediction.
# shap.decision_plot(explainer.expected_value, shap_values[j], X.iloc[[j]], link='logit')