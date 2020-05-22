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

import analyzer.dataset as ds
import analyzer.loaders.hartford.hartford as hartford

def get_hartford_predictions(model_type, model_lab, data_path, website_path = "/home/hwiberg/research/COVID_risk/website/"):
  with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
    model_file = pickle.load(file)
  # #Extract the inputs of the model
  model = model_file['model']
  columns = model_file['columns']
  imputer= model_file['imputer']
  name_datasets = np.asarray(['discharge', 'comorbidities', 'vitals', 'lab', 'demographics', 'swab'])
  extra_data = False
  demographics_data = True
  
  if model_lab == "with_lab":
    discharge_data = True
    comorbidities_data = True
    vitals_data = True
    lab_tests = True
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])
  elif model_lab == "without_lab":
    discharge_data = True
    comorbidities_data = True
    vitals_data = True
    lab_tests = False
    swabs_data = False
    mask = np.asarray([discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data])
    print(name_datasets[mask])
    
  if model_type == 'mortality':
    prediction = 'Outcome'
  elif model_type == 'infection':
    prediction = 'Swab'
    
  data_hartford = hartford.load_hartford(data_path, 
    discharge_data, comorbidities_data, vitals_data, lab_tests, demographics_data, swabs_data)
    
  X_hartford, y_hartford =  ds.create_dataset(data_hartford,
                                        discharge_data,
                                        comorbidities_data,
                                        vitals_data,
                                        lab_tests,
                                        demographics_data,
                                        swabs_data,
                                        prediction = prediction)
                                        
  ## Identify missing columns
  missing_cols = list(set(columns).difference(X_hartford.columns))
  print("Missing Column Count: "+str(len(missing_cols)))
  X_missing = X_hartford.reindex(columns = columns)
  X = imputer.transform(X_missing)
  df_X = pd.DataFrame(X, columns = columns, dtype=np.float)
  preds = model.predict_proba(df_X)[:,1]
  auc = metrics.roc_auc_score(y_hartford, preds)
  print("AUC: "+str(auc))
  return pd.DataFrame({'y': y_hartford, 'prob_pos': preds})

  #Load model corresponding to model_type and lab

# version = "inpatient"; site = "main"
# data_path = '/nfs/sloanlab003/projects/cov19_calc_proj/hartford/hhc_'+version+'_'+site+'.csv'
# result_path = '../../covid19_hartford/'

# model_type = "mortality"
# model_lab = "with_lab"

# res = hhc.get_hartford_predictions(model_type, model_lab, data_path)
# res.to_csv(result_path+'predictions_'+model_type+'_'+model_lab+'.csv')






