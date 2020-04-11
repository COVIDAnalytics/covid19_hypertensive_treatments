import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#Julia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from interpretableai import iai

import seaborn as sns

import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



import analizer.loader as loader

# Load cremona data
data = loader.load_cremona('../data/cremona/')






# TODO: Continue from here


# In[3]:


# Select only covid patient
anagraphics = anagraphics[anagraphics['ESITO TAMPONE'].str.contains('POSITIVO')]


# ## Filter datasets

# ### Pick only people who have recovered or died

# In[4]:


anagraphics = anagraphics[anagraphics["DATA DI DIMISSIONE/DECESSO"].notnull()]  # Pick only people with end date
anagraphics = anagraphics[~anagraphics['ESITO'].str.contains('TRASFERITO')] # Remove patients who have been transferred

def fix_outcome(outcome):
    if 'DIMESSO' in outcome:
        return 'DIMESSO'
    elif outcome == 'DECEDUTO':
        return outcome
    else:
        raise ValueError('Not recognized')

anagraphics['ESITO'] = anagraphics['ESITO'].apply(fix_outcome)  # Make outcome binary


# ### Filter common patients and drop duplicates

# In[5]:


# Drop duplicates
n_orig = len(anagraphics)
anagraphics.drop_duplicates(['CODICE FISCALE', "ESITO"], inplace=True)
n_unique = len(anagraphics)
print("Removed %d duplicates with same (ID, outcome)" % (n_orig - n_unique))

# Drop patients with multiple diagnosis
n_orig = len(anagraphics)
anagraphics.drop_duplicates(['CODICE FISCALE'], inplace=True)
n_unique = len(anagraphics)
print("Removed %d duplicates with same (ID)" % (n_orig - n_unique))

# Drop patients with multiple diagnosis
n_orig = len(anagraphics)
anagraphics.drop_duplicates(['NOSOLOGICO'], inplace=True)
n_unique = len(anagraphics)
print("Removed %d duplicates with same (Nosologico)" % (n_orig - n_unique))


# ### Filter patients present in all datasets

# In[6]:


# Filter by Codice Fiscale
patients_codice_fiscale = anagraphics[anagraphics["CODICE FISCALE"].isin(drugs["Codice Fiscale"])]['CODICE FISCALE'].unique()
drugs_covid = drugs[drugs['Codice Fiscale'].isin(patients_codice_fiscale)]
anagraphics = anagraphics[anagraphics['CODICE FISCALE'].isin(patients_codice_fiscale)]

# Filter by Nosologico
patients_nosologico = er_vital_signs[er_vital_signs['SCHEDA_PS'].isin(anagraphics["NOSOLOGICO"])]['SCHEDA_PS'].unique()
anagraphics = anagraphics[anagraphics['NOSOLOGICO'].isin(patients_nosologico)]
er_vital_signs = er_vital_signs[er_vital_signs['SCHEDA_PS'].isin(patients_nosologico)]

# Filter again Codice Fiscale
patients_codice_fiscale = anagraphics['CODICE FISCALE'].unique()
drugs_covid = drugs[drugs['Codice Fiscale'].isin(patients_codice_fiscale)]

print("Len patients by CF :", len(patients_codice_fiscale))
print("Len patients by Nosologico :", len(patients_nosologico))


# In[7]:


# Add Codice Fiscale to ER data
er_vital_signs['CODICE FISCALE'] = ""
for p in patients_codice_fiscale:
    nosologico = anagraphics[anagraphics['CODICE FISCALE'] == p]['NOSOLOGICO'].values
    if len(nosologico) > 1:
        raise ValueError("Duplicates in nosologico")
    er_vital_signs.loc[er_vital_signs['SCHEDA_PS'] == nosologico[0], 'CODICE FISCALE'] = p


# ### Fix swapped dates

# In[8]:


idx_wrong = pd.to_datetime(anagraphics["DATA DI DIMISSIONE/DECESSO"], format='%m/%d/%y') < pd.to_datetime(anagraphics["DATA DI RICOVERO"], format='%m/%d/%y')
wrong_dates = anagraphics[['DATA DI DIMISSIONE/DECESSO','DATA DI RICOVERO']].loc[idx_wrong]
anagraphics.loc[idx_wrong, 'DATA DI RICOVERO'] = wrong_dates['DATA DI DIMISSIONE/DECESSO']
anagraphics.loc[idx_wrong, 'DATA DI DIMISSIONE/DECESSO'] = wrong_dates['DATA DI RICOVERO']


# ### Compute length of stay

# In[9]:


def n_days(t):
    if isinstance(t, str):
        date = datetime.datetime.strptime(t, '%m/%d/%y')
        new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
        day_of_the_year = (date - new_year_day).days + 1
        return day_of_the_year
    else:
        return np.NaN
anagraphics['length_of_stay'] = anagraphics["DATA DI DIMISSIONE/DECESSO"].apply(n_days) - anagraphics["DATA DI RICOVERO"].apply(n_days)
print("Minimum length of stay ", anagraphics['length_of_stay'].min())
print("Maximum length of stay ", anagraphics['length_of_stay'].max())


# # Create final dataset

# In[10]:


# Anagraphics (translated to english)
anagraphics_features = ['SESSO', "ETA'", 'ESITO']
dataset_anagraphics = pd.DataFrame(columns=anagraphics_features, index=patients_codice_fiscale)
dataset_anagraphics.loc[:, anagraphics_features] = anagraphics[['CODICE FISCALE'] + anagraphics_features].set_index('CODICE FISCALE')
dataset_anagraphics = dataset_anagraphics.rename(columns={"ETA'": "age", "SESSO": "sex", "ESITO": "outcome"})
dataset_anagraphics.loc[:, 'sex'] = dataset_anagraphics.loc[:, 'sex'].astype('category')
dataset_anagraphics.loc[:, 'outcome'] = dataset_anagraphics.loc[:, 'outcome'].astype('category')
dataset_anagraphics['outcome'] = dataset_anagraphics['outcome'].cat.rename_categories({'DIMESSO': 'discharged', 'DECEDUTO': 'deceased'})
dataset_anagraphics


# In[11]:


# Comorbidities from drugs
comorbidities = comorbidities_data['therapy_for_filtered'].dropna().unique().tolist()
dataset_comorbidities = pd.DataFrame(0, columns=comorbidities, index=patients_codice_fiscale)
for p in patients_codice_fiscale:
    drugs_p = drugs_covid[drugs_covid['Codice Fiscale'] == p]['Principio Attivo']
    for d in drugs_p:
        if d != 'NESSUN PRINCIPIO ATTIVO':
            comorb_d = comorbidities_data[comorbidities_data['Active_substance'] == d]['therapy_for_filtered']
            if len(comorb_d) != 1:
                import ipdb; ipdb.set_trace()
                raise ValueError('Error in dataset. We need only one entry per active substance')
            comorb_d = comorb_d.iloc[0]
            if not pd.isnull(comorb_d):
                dataset_comorbidities.loc[p, comorb_d] = 1

# Drop columns with all zeroes
for c in comorbidities:
    if dataset_comorbidities[c].sum() == 0:
        dataset_comorbidities.drop(c, axis='columns', inplace=True)
dataset_comorbidities = dataset_comorbidities.astype('category')

# Final
data_with_comorbidities = dataset_anagraphics.join(dataset_comorbidities)


# In[12]:


# Data with ER vitals
vital_signs = ['SaO2', 'P. Max', 'P. Min', 'F. Card.', 'F. Resp.', 'Temp.', 'Dolore', 'GCS', 'STICKGLI']
dataset_vitals = pd.DataFrame(np.nan, columns=vital_signs, index=patients_codice_fiscale)
for p in patients_codice_fiscale:
    vitals_p = er_vital_signs[er_vital_signs['CODICE FISCALE'] == p][['NOME_PARAMETRO_VITALE', 'VALORE_PARAMETRO']]
    for vital_name in vital_signs:
        # Take mean if multiple values
        vital_value = vitals_p[vitals_p['NOME_PARAMETRO_VITALE'] == vital_name]['VALORE_PARAMETRO']
        vital_value = pd.to_numeric(vital_value).mean()
        dataset_vitals.loc[p, vital_name] = vital_value

# Drop columns with less than 30% values
nan_threashold = 35
percent_missing = dataset_vitals.isnull().sum() * 100 / len(dataset_vitals)
missing_value_dataset_vitals = pd.DataFrame({'percent_missing': percent_missing})
vital_signs = missing_value_dataset_vitals[missing_value_dataset_vitals['percent_missing'] < nan_threashold].index.tolist()
dataset_vitals = dataset_vitals[vital_signs]

# Input missing values
imputed_dataset_vitals = iai.impute(dataset_vitals)
imputed_dataset_vitals.index = dataset_vitals.index

# Rename to English
imputed_dataset_vitals = imputed_dataset_vitals.rename(columns={"P_ Max": "systolic_blood_pressure",
                                                                "P_ Min": "diastolic_blood_pressure",
                                                                "F_ Card_": "cardiac_frequency",
                                                                "Temp_": "temperature_celsius",
                                                                "F_ Resp_": "respiratory_frequency"})


# Final
data_with_comorbidities_and_signs = data_with_comorbidities.join(imputed_dataset_vitals)


# In[13]:


data_ml = data_with_comorbidities_and_signs
data_ml


# ## Train trees

# In[110]:


SEED = 1
X = data_ml.loc[:, data_ml.columns != 'outcome']
y = data_ml.loc[:, 'outcome']
(X_train, y_train), (X_test, y_test) = iai.split_data('classification', X, y, train_proportion=0.8, seed = SEED)


# In[16]:


grid = iai.GridSearch(
    iai.OptimalTreeClassifier(
        random_seed = SEED,
    ),
    max_depth=range(1, 10),
    minbucket=[5, 10, 15, 20, 25, 30, 35],
    ls_num_tree_restarts=200,
)
grid.fit_cv(X_train, y_train, n_folds=10, validation_criterion = 'auc')


# In[17]:


print('In Sample AUC', grid.score(X_train, y_train, criterion='auc'))
print('Out of Sample AUC', grid.score(X_test, y_test, criterion='auc'))


# In[18]:


grid.get_best_params()


# In[20]:


grid.get_learner()


# ## Save trees

# In[21]:


best_learner = grid.get_learner()
best_learner.write_json('trees/cv10_seed1234567/learner.json')
best_learner.write_questionnaire('trees/cv10_seed1234567/app.html')


# ## Try XGBoost and RF

# In[109]:


#INITIALIZE A LIST TO KEEP TRACK OF ALL BEST MODELS DEVELOPED

#DEFINE FUNCTION THAT COMPUTES ACCURACY, TPR, FPR, AND AUC for GIVEN MODEL
def Scores(model, t_X, t_Y, te_X, te_Y):

    # misclassification accuracies
    accTrain = np.round(sum(model.predict(t_X) == t_Y)/len(t_Y),2)
    accTest = np.round(sum(model.predict(te_X) == testY)/len(te_Y),2)

    pred_t_Y = model.predict_proba(t_X)[:, 1]
    pred_te_Y = model.predict_proba(te_X)[:, 1]

    is_fpr, is_tpr, _ = roc_curve(t_Y, pred_t_Y)
    isAUC = auc(is_fpr, is_tpr)

    ofs_fpr, ofs_tpr, _ = roc_curve(te_Y, pred_te_Y)
    ofsAUC = auc(ofs_fpr, ofs_tpr)

    return (accTrain, accTest, ofs_fpr, ofs_tpr, isAUC, ofsAUC)

#DEFINE FUNCTION THAT RETURNS TOP 10 PREDICTORS GIVEN A MODEL
def top10(model, trainX):
    try:
        varsImpo = pd.DataFrame({'names':trainX.columns, 'vals':model.feature_importances_})
        varsImpo = varsImpo.sort_values(by='vals', ascending = False)

    except:
        print("The model does not support the method: feature_importances_")

    return varsImpo


# In[ ]:


#INITIATE 10-FOLD CV
param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "min_samples_leaf": [4, 8, 12],
        "n_estimators": [2500, 2000, 1500]
}
XGB = GradientBoostingClassifier()
gridsearch1 = GridSearchCV(estimator = XGB, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 1)
gridsearch1.fit(X_train, y_train)

#RECORD BEST MODEL
bestHyp = gridsearch1.best_params_
bestXGB = gridsearch1.best_estimator_


# In[ ]:


##### INITIATE 10-FOLD CV
param_grid = {
        "bootstrap": [True],
        "max_features": ['sqrt', 'log2'],
        "min_samples_leaf": [5, 10],
        "min_samples_split": [3, 5, 8],
        "n_estimators": [400, 800, 1000]
}
RF = RandomForestClassifier()
gridsearch2 = GridSearchCV(estimator = RF, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 1)
gridsearch2.fit(X_train, y_train)

#RECORD BEST MODEL
bestHypRF = gridsearch2.best_params_
bestRF = gridsearch2.best_estimator_


# ## Correlation matrix

# In[15]:


data_corr = data_ml.copy()
data_corr['sex'] = data_corr['sex'].astype(object)
data_corr.loc[data_corr['sex'] == 'M','sex'] = 0
data_corr.loc[data_corr['sex'] == 'F','sex'] = 1
X_corr = data_corr.loc[:, data_corr.columns != 'outcome'].astype(np.float64)
X_corr.corr()


# In[95]:


mask = np.triu(np.ones_like(X_corr.corr(), dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(X_corr.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[29]:


X_corr.corr().iloc[1,3]


# In[106]:


upper = X_corr.corr().where(np.triu(np.ones(X_corr.corr().shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.25
rows, columns = np.where(abs(upper) > 0.25)


# In[107]:


list(zip(upper.columns[rows], upper.columns[columns]))


# ## Anonymize, translate and store

# In[ ]:


# data_with_comorbidities.reset_index(inplace=True, drop=True)
# data_with_comorbidities.to_csv('data.csv')

