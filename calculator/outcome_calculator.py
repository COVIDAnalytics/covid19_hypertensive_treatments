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
from analyzer.learners import xgboost_classifier
from analyzer.learners import rf_classifier


SEED = 1
prediction = 'icu'
folder_name = 'cv10_script_seed' + str(SEED) + '_' + prediction
output_folder = 'predictors/outcome'

# Load cremona data
data = load_cremona('../data/cremona/')

# Create dataset
X, y = create_dataset(data, prediction = prediction)
(X_train, y_train), (X_test, y_test) = iai.split_data('classification',
                                                      X, y, train_proportion=0.8, seed=SEED)


# Train trees
output_path = os.path.join(output_folder, 'trees', folder_name)
create_dir(output_path)
oct_scores = train_oct(X_train, y_train, X_test, y_test, output_path, seed=SEED)


#PARAMETERS GRID
param_grid_XGB = {
        "learning_rate": [0.001, 0.01, 0.1],
        "min_samples_leaf": [4, 8, 12, 20],
        "n_estimators": [2500, 2000, 1500, 500]}



param_grid_RF = {
        "bootstrap": [True],
        "max_features": ['sqrt', 'log2'],
        "min_samples_leaf": [5, 10],
        "min_samples_split": [3, 5, 8],
        "n_estimators": [2500, 2000, 1500, 500]}

xgboost_classifier(X_train, y_train, X_test, y_test, param_grid_XGB)
rf_classifier(X_train, y_train, X_test, y_test, param_grid_RF)





#  ## Correlation matrix

#  In[15]:


#  data_corr = data_ml.copy()
#  data_corr['sex'] = data_corr['sex'].astype(object)
#  data_corr.loc[data_corr['sex'] == 'M','sex'] = 0
#  data_corr.loc[data_corr['sex'] == 'F','sex'] = 1
#  X_corr = data_corr.loc[:, data_corr.columns != 'outcome'].astype(np.float64)
#  X_corr.corr()


#  # In[95]:


#  mask = np.triu(np.ones_like(X_corr.corr(), dtype=np.bool))

#  # Set up the matplotlib figure
#  f, ax = plt.subplots(figsize=(11, 9))

#  # Generate a custom diverging colormap
#  cmap = sns.diverging_palette(220, 10, as_cmap=True)

#  # Draw the heatmap with the mask and correct aspect ratio
#  sns.heatmap(X_corr.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
#              square=True, linewidths=.5, cbar_kws={"shrink": .5})


#  # In[29]:


#  X_corr.corr().iloc[1,3]


#  # In[106]:


#  upper = X_corr.corr().where(np.triu(np.ones(X_corr.corr().shape), k=1).astype(np.bool))

#  # Find features with correlation greater than 0.25
#  rows, columns = np.where(abs(upper) > 0.25)


#  # In[107]:


#  list(zip(upper.columns[rows], upper.columns[columns]))


#  # ## Anonymize, translate and store

#  # In[ ]:


#  # data_with_comorbidities.reset_index(inplace=True, drop=True)
#  # data_with_comorbidities.to_csv('data.csv')

