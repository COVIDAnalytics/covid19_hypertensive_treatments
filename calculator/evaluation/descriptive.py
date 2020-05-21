import numpy as np
import pandas as pd
import pickle

import evaluation.descriptive_utils as u
import analyzer.dataset as ds

from scipy import stats

#%% Generate dataset
model_type = "mortality"; model_lab = "with_lab";

website_path = '../../website/'

with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)

model = model_file['model']
features = model_file['json']
columns = model_file['columns']
imputer= model_file['imputer']

X, y = u.get_dataset_preload(model_type, model_lab, columns, imputer, impute = False)

#%% Mortality Split
data = X.drop(['Location'], axis = 1).copy()
data['Outcome'] = y

data_a= data.query('Outcome == 1')
data_b = data.query('Outcome == 0')
summary_table = u.pairwise_compare(data_a, data_b, features, title_mapping = u.title_mapping,
                                 filter_A = 'Non-Survivor', filter_B = 'Survivor')

summary_table.to_csv('../results/summary_tables/descriptive_derivation_bysurvival.csv',
                      index = False)

#%% Country Split
data = X.copy()
data['Outcome'] = y

data_a= data.query('Location == "Cremona"')
data_b = data.query('Location == "Spain"')

data = pd.concat([data_a, data_b])
summary_table = u.pairwise_compare(data_a, data_b, features,
                                title_mapping = u.title_mapping,
                                filter_A = 'Cremona', filter_B = 'Spain')

summary_table.to_csv('../results/summary_tables/descriptive_derivation_bycountry.csv',
                      index = False)


#%% Greece

data_a = X.copy()
data_a['Outcome'] = y

val_df = pd.read_csv("../../covid19_greece/general_greek_registry.csv")

val_df = val_df.loc[val_df['Outcome'].isin([0,1])]
if model_lab == 'without_lab':
    val_df = val_df.rename(columns={'ABG: Oxygen Saturation (SaO2)':'SaO2'})
if val_df['Body Temperature'].mean() < 45:
    val_df['Body Temperature'] = ((val_df['Body Temperature']/5)*9)+32
    
data_b = val_df.reindex(columns = columns+['Outcome'])

summary_table = u.pairwise_compare(data_a, data_b, features,
                                title_mapping = u.title_mapping,
                                filter_A = 'Derivation', filter_B = 'Greece')

summary_table.to_csv('../results/summary_tables/descriptive_derivation_greece.csv',
                      index = False)


#%% Hartford
# data_a = X.copy()
# data_a['Outcome'] = y

# df_hhc = pd.read_csv("/home/hwiberg/research/COVID_risk/covid19_hartford/hhc_20200518.csv")

# if model_lab == "with_lab":
# 	df_hhc.rename(columns={'SaO2':'ABG: Oxygen Saturation (SaO2)'}, inplace = True)

# df_hhc.replace({'Alive':0,'Expired':1}, inplace = True)

# X_missing = df_hhc.reindex(columns = columns)
# y_hhc = df_hhc['Outcome']

# data_b = X_missing.copy()
# data_b['Outcome'] = y_hhc

# data = pd.concat([data_a, data_b])
# summary_table = pairwise_compare(data, data_a, data_b, features,
#                                 title_mapping = title_mapping,
#                                 filter_A = 'Derivation', filter_B = 'Hartford')

# summary_table.to_csv('../results/summary_tables/descriptive_derivation_hartford.csv',
#                       index = False)


#%% Hartford by type

df_hhc = pd.read_csv("/home/hwiberg/research/COVID_risk/covid19_hartford/hhc_20200520.csv")

if model_lab == "with_lab":
 	df_hhc.rename(columns={'SaO2':'ABG: Oxygen Saturation (SaO2)'}, inplace = True)

df_hhc.replace({'Alive':0,'Expired':1}, inplace = True)
df_hhc.rename({'Chronic Kidney Disease':'Chronic kidney disease'})


# df_hhc = df_hhc.reindex(columns = [columns+['Patient_Class','Outcome']])

data_a = df_hhc.query('Patient_Class == "Inpatient"')
data_b = df_hhc.query('Patient_Class != "Inpatient"')


data = pd.concat([data_a, data_b])
summary_table = u.pairwise_compare(data_a, data_b, features,
                                title_mapping = u.title_mapping,
                                filter_A = 'Inpatient', filter_B = 'Other')

summary_table.to_csv('../results/summary_tables/descriptive_derivation_hartford_byclass.csv',
                      index = False)


#%% Plot CRP values
import matplotlib.pyplot as plt

ft = 'C-Reactive Protein (CRP)'

fig, axes = plt.subplots(1, 3, figsize = (50, 20))
X.loc[X['Location']=='Spain'].hist(ft, bins=100, ax=axes[0], range = [0,600])
X.loc[X['Location']=='Cremona'].hist(ft, bins=100, ax=axes[1], range = [0,600])
val_df.hist(ft, bins=100, ax=axes[2], range = [0,600])
axes[0].set_title('Spain')
axes[1].set_title('Cremona')
axes[2].set_title('Greece')

f = plt.gcf()
f.savefig('../results/summary_tables/crp_compare.png', bbox_inches='tight')
plt.clf()
