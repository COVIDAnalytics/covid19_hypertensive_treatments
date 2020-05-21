import pandas as pd
import numpy as np
import os
import re
import datetime
import pickle

## pickle-specific files
import xgboost
import sklearn.impute
import shap

from sklearn import metrics

model_type = "mortality"
model_lab = "with_lab"
website_path = "/home/hwiberg/research/COVID_risk/website/"

print(model_type)
print(model_lab)

#Load model corresponding to model_type and lab
with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
    model_file = pickle.load(file)

# #Extract the inputs of the model
model = model_file['model']
columns = model_file['columns']
imputer= model_file['imputer']

df_hhc = pd.read_csv("/home/hwiberg/research/COVID_risk/covid19_hartford/hhc_20200520.csv")

## Filter 
# df_hhc = df_hhc.loc[~df_hhc['Date_Admission'].str.startswith("05"),:]

if model_lab == "with_lab":
	df_hhc.rename(columns={'SaO2':'ABG: Oxygen Saturation (SaO2)'}, inplace = True)

## Identify missing columns
missing_cols = list(set(columns).difference(df_hhc.columns))

X_missing = df_hhc.reindex(columns = columns)
y = df_hhc['Outcome']

X = imputer.transform(X_missing)
df_X = pd.DataFrame(X, columns = columns, dtype=np.float)

# df_X.loc[:,'C-Reactive Protein (CRP)'] = 10*df_X.loc[:,'C-Reactive Protein (CRP)'] 
#
# for x in missing_cols:
# 	df_X.loc[df_X[x]>0, x] = 1
# 	print(sum(df_X[x]))

preds = model.predict_proba(df_X)[:,1]

auc = metrics.roc_auc_score(y, preds)
auc