import numpy as np
import pandas as pd
import os
import pickle
import itertools

# Other packages
import shap
import matplotlib.pyplot as plt

os.chdir('git/covid19_calculator/calculator/')

#%% Evaluate drivers of individual predictions 
# Select index j for prediction to generate.

# shap.force_plot(explainer.expected_value, shap_values[1:10,:], X.iloc[[1:10,:]])
j=20
plot = shap.force_plot(explainer.expected_value, shap_values[j], X.iloc[[j]] , link="logit")
plot

#%%  As an alternative view, you can trace a 3D plot of the values and their impact on the prediction.
# shap.decision_plot(explainer.expected_value, shap_values[j], X.iloc[[j]], link='logit')