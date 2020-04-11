import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os

#Julia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from interpretableai import iai


# Other packages
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from analyzer.loaders.cremona import load_cremona
from analyzer.dataset import create_dataset
from analyzer.utils import create_dir
from analyzer.learners import train_oct


SEED = 1
folder_name = 'cv10_script_seed1'
output_folder = 'predictors/outcome'

# Load cremona data
data = load_cremona('../data/cremona/')

# Create dataset
X, y = create_dataset(data)
(X_train, y_train), (X_test, y_test) = iai.split_data('classification',
                                                      X, y, train_proportion=0.8, seed=SEED)


# Train trees
output_path = os.path.join(output_folder, 'trees', folder_name)
create_dir(output_path)
oct_scores = train_oct(X_train, y_train, X_test, y_test, output_path, seed=SEED)




#  # ## Try XGBoost and RF
#
#  # In[109]:
#
#
#  #INITIALIZE A LIST TO KEEP TRACK OF ALL BEST MODELS DEVELOPED
#
#  #DEFINE FUNCTION THAT COMPUTES ACCURACY, TPR, FPR, AND AUC for GIVEN MODEL
#  def Scores(model, t_X, t_Y, te_X, te_Y):
#
#      # misclassification accuracies
#      accTrain = np.round(sum(model.predict(t_X) == t_Y)/len(t_Y),2)
#      accTest = np.round(sum(model.predict(te_X) == testY)/len(te_Y),2)
#
#      pred_t_Y = model.predict_proba(t_X)[:, 1]
#      pred_te_Y = model.predict_proba(te_X)[:, 1]
#
#      is_fpr, is_tpr, _ = roc_curve(t_Y, pred_t_Y)
#      isAUC = auc(is_fpr, is_tpr)
#
#      ofs_fpr, ofs_tpr, _ = roc_curve(te_Y, pred_te_Y)
#      ofsAUC = auc(ofs_fpr, ofs_tpr)
#
#      return (accTrain, accTest, ofs_fpr, ofs_tpr, isAUC, ofsAUC)
#
#  #DEFINE FUNCTION THAT RETURNS TOP 10 PREDICTORS GIVEN A MODEL
#  def top10(model, trainX):
#      try:
#          varsImpo = pd.DataFrame({'names':trainX.columns, 'vals':model.feature_importances_})
#          varsImpo = varsImpo.sort_values(by='vals', ascending = False)
#
#      except:
#          print("The model does not support the method: feature_importances_")
#
#      return varsImpo
#
#
#  # In[ ]:
#
#
#  #INITIATE 10-FOLD CV
#  param_grid = {
#          "learning_rate": [0.001, 0.01, 0.1],
#          "min_samples_leaf": [4, 8, 12],
#          "n_estimators": [2500, 2000, 1500]
#  }
#  XGB = GradientBoostingClassifier()
#  gridsearch1 = GridSearchCV(estimator = XGB, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 1)
#  gridsearch1.fit(X_train, y_train)
#
#  #RECORD BEST MODEL
#  bestHyp = gridsearch1.best_params_
#  bestXGB = gridsearch1.best_estimator_
#
#
#  # In[ ]:
#
#
#  ##### INITIATE 10-FOLD CV
#  param_grid = {
#          "bootstrap": [True],
#          "max_features": ['sqrt', 'log2'],
#          "min_samples_leaf": [5, 10],
#          "min_samples_split": [3, 5, 8],
#          "n_estimators": [400, 800, 1000]
#  }
#  RF = RandomForestClassifier()
#  gridsearch2 = GridSearchCV(estimator = RF, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 1)
#  gridsearch2.fit(X_train, y_train)
#
#  #RECORD BEST MODEL
#  bestHypRF = gridsearch2.best_params_
#  bestRF = gridsearch2.best_estimator_
#
#
#  # ## Correlation matrix
#
#  # In[15]:
#
#
#  data_corr = data_ml.copy()
#  data_corr['sex'] = data_corr['sex'].astype(object)
#  data_corr.loc[data_corr['sex'] == 'M','sex'] = 0
#  data_corr.loc[data_corr['sex'] == 'F','sex'] = 1
#  X_corr = data_corr.loc[:, data_corr.columns != 'outcome'].astype(np.float64)
#  X_corr.corr()
#
#
#  # In[95]:
#
#
#  mask = np.triu(np.ones_like(X_corr.corr(), dtype=np.bool))
#
#  # Set up the matplotlib figure
#  f, ax = plt.subplots(figsize=(11, 9))
#
#  # Generate a custom diverging colormap
#  cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
#  # Draw the heatmap with the mask and correct aspect ratio
#  sns.heatmap(X_corr.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
#              square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
#
#  # In[29]:
#
#
#  X_corr.corr().iloc[1,3]
#
#
#  # In[106]:
#
#
#  upper = X_corr.corr().where(np.triu(np.ones(X_corr.corr().shape), k=1).astype(np.bool))
#
#  # Find features with correlation greater than 0.25
#  rows, columns = np.where(abs(upper) > 0.25)
#
#
#  # In[107]:
#
#
#  list(zip(upper.columns[rows], upper.columns[columns]))
#
#
#  # ## Anonymize, translate and store
#
#  # In[ ]:
#
#
#  # data_with_comorbidities.reset_index(inplace=True, drop=True)
#  # data_with_comorbidities.to_csv('data.csv')
#
