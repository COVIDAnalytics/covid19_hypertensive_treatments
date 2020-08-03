import evaluation.treatment_utils as  u
import evaluation.descriptive_utils as d
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import matplotlib.pyplot as plt


from julia import Julia           
# jl = Julia(compiled_modules = False)
jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
from interpretableai import iai

#%% Set Problem Parameters
#Paths for data access

outcome = "COMORB_DEATH"

if outcome == "COMORB_DEATH":
    outcome_path = 'COMORB_DEATH/'
else:
     outcome_path = ''

data_path = '../../covid19_treatments_data/'
results_path = '../../covid19_treatments_results/'
version_folder = "matched_all_treatments_der_val_update/"
version_folder_results = "matched_all_treatments_der_val_update_nomedhx/"

save_path = results_path + version_folder_results + outcome_path + 'summary/'
preload = True
matched = True
match_status = 'matched' if matched else 'unmatched'

# treatment_list = ['All', 'Chloroquine_and_Anticoagulants','Chloroquine_and_Antivirals']
#treatment_list = ['Chloroquine_Only', 'All', 'Chloroquine_and_Anticoagulants','Chloroquine_and_Antivirals', 'Non-Chloroquine']
treatment_list = ['Chloroquine_Only', 'Chloroquine_and_Anticoagulants','Chloroquine_and_Antivirals', 'Non-Chloroquine']
algorithm_list = ['lr','rf','cart','xgboost','oct','qda','gb']
# algorithm_list = ['lr','rf','cart','qda','gb']

#%%  Evaluate specific version
data_version = 'test' # in ['train','test','validation','validation_cremona','validation_hope']:
weighted_status = 'weighted'

#Read in the relevant data
X, Z, y = u.load_data(data_path+version_folder,'hope_hm_cremona_matched_all_treatments_train.csv',
                            split=data_version, matched=matched, prediction = outcome)

summary = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'.csv')
Z_presc = summary['Prescribe']
Y_presc = summary['AverageProbability']

#%% Create multi-class classification tre
grid = iai.GridSearch(
    iai.OptimalTreeClassifier(
        random_seed=1,
    ),
    max_depth=range(3, 9),
    criterion = ['gini','entropy']
)
grid.fit(X, Z_presc, validation_criterion = 'entropy')

lnr = grid.get_learner()
lnr.write_html(save_path+data_version+'_'+match_status+'_'+weighted_status+'_prescription_explanations.html')

lnr.score(X, Z_presc, criterion = 'misclassification')

df_list = [pd.Series(['dataset','misclassification','entropy','gini'])]
for data_version in ['train','test','validation','validation_cremona','validation_hope']:
  X_test, Z_test, y_test = u.load_data(data_path+version_folder,'hope_hm_cremona_matched_all_treatments_train.csv',
                                  split=data_version,matched=matched)
  X_test = X_test.reindex(X.columns, axis = 1, fill_value = 0)
  # Load presccriptions
  summary_test = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'.csv')
  Z_test_presc = summary_test['Prescribe']
  # compute misclassification
  s1 = lnr.score(X_test, Z_test_presc, criterion = 'misclassification')
  s2 = lnr.score(X_test, Z_test_presc, criterion = 'entropy')
  s3 = lnr.score(X_test, Z_test_presc, criterion = 'gini')
  df_list.append(pd.Series([data_version, s1, s2, s3]))
  print("Version = ", data_version, "; Score = ", s1)

df_performance = pd.concat(df_list, axis=1)
df_performance.to_csv(save_path+data_version+'_'+match_status+'_prescription_explanations_performance.csv',
  index = False, header = False)


#%% Descriptive Analysis

features = {'categorical':['DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
       'RENALINSUF', 'ANYLUNGDISEASE', 'AF', 'VIH', 'ANYHEARTDISEASE',
       'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE', 'LIVER_DISEASE',
       'CANCER', 'SAT02_BELOW92',
       'BLOOD_PRESSURE_ABNORMAL_B', 'DDDIMER_B', 'PCR_B', 'TRANSAMINASES_B',
       'LDL_B', 'IN_PREVIOUSASPIRIN', 'IN_OTHERANTIPLATELET',
       'IN_ORALANTICOAGL', 'IN_BETABLOCKERS', 'IN_BETAGONISTINHALED',
       'IN_GLUCORTICOIDSINHALED', 'IN_DVITAMINSUPLEMENT', 'IN_BENZODIACEPINES',
       'IN_ANTIDEPRESSANT', 'CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB',
       'ANTIBIOTICS', 'ACEI_ARBS', 'GENDER_MALE', 'RACE_CAUC', 'RACE_LATIN',
       'RACE_ORIENTAL', 'RACE_OTHER'],
                'numeric':['AGE','MAXTEMPERATURE_ADMISSION','CREATININE', 'SODIUM', 'LEUCOCYTES', 'LYMPHOCYTES',
       'HEMOGLOBIN', 'PLATELETS'],
                'multidrop':[]}

df = pd.concat([X,y,Z_presc], axis=1)
desc_list = []

for p in treatment_list:
    df_sub = df.loc[df['Prescribe']==p,:]
    desc = d.descriptive_table_treatments(df_sub, features, short_version = True)
    desc = desc.drop('Percent Missing', axis=1)
    desc.loc['Treatment'] = p
    desc = desc.add_suffix('_'+p)
    desc_list.append(desc)

desc_all = pd.concat(desc_list, axis=1)
columns = desc_all.index.drop(['Patient Count','Treatment'])
for idx, pair in enumerate(list(itertools.combinations(treatment_list, 2))):
    print(pair)
    p_a, p_b  = pair
    data_a = df.loc[df['Prescribe']==p_a,:]
    data_b = df.loc[df['Prescribe']==p_b,:]
    col_sub = columns[(np.var(data_a[columns], axis=0)>1e-20) & (np.var(data_b[columns], axis=0)>1e-20)]
    sig_test = stats.ttest_ind(data_a[col_sub], data_b[col_sub], 
                               axis = 0, equal_var = False, nan_policy = 'omit')
    df_sig = pd.DataFrame(np.transpose(sig_test)[:,1], columns = ['SigTest'+str(idx)])
    df_sig.index = col_sub
    df_sig.loc['Treatment'] = p_a+', '+p_b
    desc_all = desc_all.merge(df_sig, how = 'left',
                                       left_index = True, right_index = True)

desc_all['Min_Significance'] = np.min(desc_all.loc[:,desc_all.columns.str.startswith('SigTest')], axis =  1)
desc_all.to_csv(save_path+data_version+'_'+match_status+'_'+weighted_status+'_prescription_descriptive.csv', index = True)

#%%
#We would like to see how the treatments get distributed
X['Z'] = Z     
X['Z_presc'] = Z_presc
X['Y'] = y
X['Y_presc'] = Y_presc

cross_treatments = X[['Z_presc', 'Z']].groupby(['Z_presc', 'Z']).size().to_frame('size').reset_index()
cross_treatments = cross_treatments.pivot(index='Z_presc', columns='Z', values='size')

cross_treatments_norm = cross_treatments.div(cross_treatments.sum(axis=1), axis=0)
cross_treatments_norm['size'] = cross_treatments.sum(axis=1)
cross_treatments_norm.to_csv(save_path+data_version+'_'+match_status+'_'+weighted_status+'_cross_prescription_summary.csv', index = True)

#%%
#Plot by age the mortality rate
# data to plot
bins= [0,40,55,70,110]
X['AgeGroup'] = pd.cut(X['AGE'], bins=bins,right=False)

age_table = X.groupby('AgeGroup')[['Y','Y_presc']].mean()
ax = age_table.plot.bar(rot=0)
# Add title and axis names
plt.title('Mortality Rate by Age Group')
plt.ylabel('Mortality Rate')
plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_'+'ageplot.png')
#%%
gender_table = X.groupby('GENDER_MALE')[['Y','Y_presc']].mean()
ax = gender_table.plot.bar(rot=0)
# Add title and axis names
plt.title('Mortality Rate by Gender')
plt.ylabel('Mortality Rate')
plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_'+'genderplot.png')

#%%
so2_table = X.groupby('SAT02_BELOW92')[['Y','Y_presc']].mean()
ax = so2_table.plot.bar(rot=0)
# Add title and axis names
plt.title('Mortality Rate by SATO2')
plt.ylabel('Mortality Rate')
plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_'+'sa02plot.png')

#%%
bins= [0,0.8,2]
X['CreatinineGroups'] = pd.cut(X['CREATININE'], bins=bins,right=False)

cr_table = X.groupby('CreatinineGroups')[['Y','Y_presc']].mean()
ax = cr_table.plot.bar(rot=0)
# Add title and axis names
plt.title('Mortality Rate by Creatinine Group')
plt.ylabel('Mortality Rate')
plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_'+'creatplot.png')











