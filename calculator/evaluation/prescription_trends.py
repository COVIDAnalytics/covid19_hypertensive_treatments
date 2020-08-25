import evaluation.treatment_utils as  u
import evaluation.descriptive_utils as d
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import matplotlib.pyplot as plt


# from julia import Julia           
# # jl = Julia(compiled_modules = False)
# jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
# from interpretableai import iai


treatments = ['CORTICOSTEROIDS','ACEI_ARBS','INTERFERONOR']


data_path = '../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'
outcome = 'COMORB_DEATH'

preload = True
matched = True
match_status = 'matched' if matched else 'unmatched'

SEEDS = range(1, 2)
# algorithm_list = ['lr','rf','cart','qda','gb','xgboost']
algorithm_list = ['lr','rf','cart','oct','xgboost','qda','gb']

#%% Generate predictions across all combinations
 #['CORTICOSTEROIDS', 'INTERFERONOR', 'ACEI_ARBS']

treatment = 'CORTICOSTEROIDS'
treatment_list = [treatment, 'NO_'+treatment]

results_path = '../../covid19_treatments_results/'
version_folder = 'matched_single_treatments_der_val_addl_outcomes/'+str(treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'


#%% Frequency of individual drugs
df = pd.read_csv('../../covid19_treatments_data/hope_hm_cremona_data_clean_imputed_addl_outcomes.csv')
df.groupby('SOURCE_COUNTRY')[treatments].mean()
df.groupby([treatment])[outcome].mean()



#%%  Evaluate specific version
data_version = 'train' # in ['train','test','validation','validation_cremona','validation_hope']:
weighted_status = 'no_weights'
threshold = 0.01

#Read in the relevant data
X, Z, y = u.load_data(data_path,training_set_name,
                    split=data_version, matched=matched, prediction = outcome)

summary = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
Z_presc = summary['Prescribe']
Y_presc = summary['AverageProbability']


X_test, Z_test, y_test = u.load_data(data_path,training_set_name,
                    split='test', matched=matched, prediction = outcome)


#%% Create multi-class classification tre
# grid = iai.GridSearch(
#     iai.OptimalTreeClassifier(
#         random_seed=1,
#     ),
#     max_depth=range(3, 9),
#     criterion = ['gini','entropy']
# )
# grid.fit(X, Z_presc, validation_criterion = 'entropy')

# lnr = grid.get_learner()
# lnr.write_html(save_path+data_version+'_'+match_status+'_'+weighted_status+'_prescription_explanations.html')

# lnr.score(X, Z_presc, criterion = 'misclassification')

# df_list = [pd.Series(['dataset','misclassification','entropy','gini'])]
# for data_version in ['train','test','validation','validation_cremona','validation_hope']:
#   X_test, Z_test, y_test = u.load_data(data_path,'hope_hm_cremona_matched_all_treatments_train.csv',
#                                   split=data_version,matched=matched)
#   X_test = X_test.reindex(X.columns, axis = 1, fill_value = 0)
#   # Load presccriptions
#   summary_test = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'.csv')
#   Z_test_presc = summary_test['Prescribe']
#   # compute misclassification
#   s1 = lnr.score(X_test, Z_test_presc, criterion = 'misclassification')
#   s2 = lnr.score(X_test, Z_test_presc, criterion = 'entropy')
#   s3 = lnr.score(X_test, Z_test_presc, criterion = 'gini')
#   df_list.append(pd.Series([data_version, s1, s2, s3]))
#   print("Version = ", data_version, "; Score = ", s1)

# df_performance = pd.concat(df_list, axis=1)
# df_performance.to_csv(save_path+match_status+'_prescription_explanations_performance.csv',
#   index = False, header = False)


#%% Descriptive Analysis

features = {'categorical':['DIABETES', 'HYPERTENSION', 'DISLIPIDEMIA', 'OBESITY',
       'RENALINSUF', 'ANYLUNGDISEASE', 'AF', 'VIH', 'ANYHEARTDISEASE',
       'ANYCEREBROVASCULARDISEASE', 'CONECTIVEDISEASE', 'LIVER_DISEASE',
       'CANCER', 'SAT02_BELOW92',
       'BLOOD_PRESSURE_ABNORMAL_B', 'DDDIMER_B', 'PCR_B', 'TRANSAMINASES_B',
       'LDL_B', 
       # 'IN_PREVIOUSASPIRIN', 'IN_OTHERANTIPLATELET',
       # 'IN_ORALANTICOAGL', 'IN_BETABLOCKERS', 'IN_BETAGONISTINHALED',
       # 'IN_GLUCORTICOIDSINHALED', 'IN_DVITAMINSUPLEMENT', 'IN_BENZODIACEPINES',
       # 'IN_ANTIDEPRESSANT', 
       'CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB',
       'ANTIBIOTICS', 'ACEI_ARBS', 'GENDER_MALE', 'RACE_CAUC', 'RACE_LATIN',
       'RACE_ORIENTAL', 'RACE_OTHER'],
                'numeric':['AGE','MAXTEMPERATURE_ADMISSION','CREATININE', 'SODIUM', 'LEUCOCYTES', 'LYMPHOCYTES',
       'HEMOGLOBIN', 'PLATELETS'],
                'multidrop':[]}

features['categorical'].remove(treatment)

df = pd.concat([X,y,Z_presc], axis=1)
desc_list = []

for p in treatment_list:
    df_sub = df.loc[df['Prescribe']==p,:]
    desc = d.descriptive_table_treatments(df_sub, features, short_version = True, outcome = outcome)
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
desc_all.to_csv(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_prescription_descriptive.csv', index = True)

#%%
#We would like to see how the treatments get distributed
X['Z'] = Z     
X['Z_presc'] = Z_presc
X['Y'] = y
X['Y_presc'] = Y_presc

## run this and replace png with _treatfreq.png to see frequency of prescription
X['Z_bin'] = Z.replace({treatment:1, 'NO_'+treatment: 0})
X['Z_presc_bin'] = Z_presc.replace({treatment:1, 'NO_'+treatment: 0})

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
bins= [0,0.8,2]
X['CreatinineGroups'] = pd.cut(X['CREATININE'], bins=bins,right=False)

def plot_byfeature(ft, file_name):
    age_table = X.groupby(ft)[['Y','Y_presc']].mean()
    ax = age_table.plot.bar(rot=0)
    # Add title and axis names
    plt.title('Mortality Rate by '+ft)
    plt.ylabel('Mortality Rate')
    plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_'+file_name+'.png')
    
    
    age_table = X.groupby(ft)[['Z_bin','Z_presc_bin']].mean()
    ax = age_table.plot.bar(rot=0)
    # Add title and axis names
    plt.title('Treatment Frequency by '+ft)
    plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_'+file_name+'_treatfreq.png')
    
    
plot_byfeature('AgeGroup','plotAgeGroup')
plot_byfeature('GENDER_MALE','plotGender')
plot_byfeature('CreatinineGroups','plotCreatinine')
plot_byfeature('SAT02_BELOW92','plotSatO2')


#%% Assess expected benefit

# summary = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'_t0.0.csv')
# summary['PROB_DIFF'] = np.abs(summary[treatment]  - summary['NO_'+treatment])
# ax = plt.hist(summary['PROB_DIFF'], 20)
# plt.title('Difference in Outcome Probabilities')
# plt.ylabel('Frequency')
# plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_'+'expected_difference.png')


#%% Look at prescription trends by subpopulation
# import matplotlib
# matplotlib.use('Agg') ## suppress display for cluster

column = 'GENDER_MALE'
col_rename = 'Gender (Male)'

# X['Z'] = Z.str.replace('_',' ')
# X['Z_presc'] = Z_presc.str.replace('_',' ')
# X['Y'] = y
# X['Y_presc'] = Y_presc

hist_true =  X.groupby([column, 'Z']).size().reset_index()
hist_true.columns = [col_rename, 'Treatment','Given']
hist_presc =  X.groupby([column, 'Z_presc']).size().reset_index()
hist_presc.columns = [col_rename, 'Treatment','Prescribed']

hist = hist_true.merge(hist_presc, on = [col_rename, 'Treatment'], how = 'left')
hist.Treatment = hist.Treatment.str.replace('Chloroquine','HCQ')

#%% Create plot
from matplotlib import style
style.use('ggplot')

cols = X[column].value_counts().shape[0]
fig, axes = plt.subplots(1, cols, figsize=(8*cols, 8), sharey = True)

for x, i in enumerate(sorted(X[column].unique())):
    data = hist[hist[col_rename] == i]
    n = data['Given'].sum()
    data['Given'] = data['Given']/n
    data['Prescribed'] = data['Prescribed']/n
    print (data)
    data.plot(x="Treatment", y=["Given", "Prescribed"], kind="bar",
              ax = axes[x])    # axes[x].set_xticklabels(data.index.values)
    axes[x].legend(loc='best')
    axes[x].grid(True)
    axes[x].set_title(col_rename+' = '+str(i))

fig.suptitle('Prescription trends by '+col_rename, fontsize=20)
plt.savefig(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_'+col_rename+'_prescriptions.png',
            bbox_inches = "tight")

#%% compare prescriptions by algorithm

result = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods.csv')

result['Prescribe_Yes'] = result['Prescribe']=='CORTICOSTEROIDS'
result.groupby('Algorithm')['Prescribe_Yes'].mean()

result.groupby('Algorithm').agg({'CORTICOSTEROIDS':'mean',
                                 'NO_CORTICOSTEROIDS':'mean'})

#%%  Evaluate consistency between different treatments

#Define a function to get the covariates matrix

def read_data_of_treatment(t, dataset_version,weighted_status='no_weights', threshold = 0.01):
    treatment = t
    treatment_list = [treatment, 'NO_'+treatment]
    
    data_path = '../../covid19_treatments_data/matched_single_treatments_der_val_addl_outcomes/'
    outcome = 'COMORB_DEATH'

    preload = True
    matched = True
    match_status = 'matched' if matched else 'unmatched'


    version_folder = 'matched_single_treatments_der_val_addl_outcomes/'+str(treatment)+'/'+str(outcome)+'/'
    results_path = '../../covid19_treatments_results/'
    save_path = results_path + version_folder + 'summary/'
    training_set_name = treatment+'_hope_hm_cremona_matched_all_treatments_train.csv'
    
    data_version = dataset_version # in ['train','test','validation','validation_cremona','validation_hope']:
    weighted_status = weighted_status
    threshold = threshold
    
    #Read in the relevant data
    X, Z, y = u.load_data(data_path,training_set_name,
                    split=data_version, matched=matched, prediction = outcome)

    summary = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
    Z_presc = summary['Prescribe']
    Y_presc = summary['AverageProbability']

    X['Z'] = Z     
    X['Z_presc'] = Z_presc
    X['Y'] = y
    X['Y_presc'] = Y_presc
    
    ## run this and replace png with _treatfreq.png to see frequency of prescription
    X['Z_bin'] = Z.replace({treatment:1, 'NO_'+treatment: 0})
    X['Z_presc_bin'] = Z_presc.replace({treatment:1, 'NO_'+treatment: 0})

    return X

def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2, 
                              left_on = df1.columns.drop(cols_exclude_corts).tolist(),
                              right_on = df2.columns.drop(cols_exclude_ace).tolist(),
                              indicator=True,
                              how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    return diff_df

corts = read_data_of_treatment('CORTICOSTEROIDS', 'train')
ace = read_data_of_treatment('ACEI_ARBS', 'train')

cols_exclude_corts = ['Z', 'Z_presc', 'Y', 'Y_presc', 'Z_bin', 'Z_presc_bin','ACEI_ARBS']
cols_exclude_ace = ['Z', 'Z_presc', 'Y', 'Y_presc', 'Z_bin', 'Z_presc_bin','CORTICOSTEROIDS']

corts_add = corts[cols_exclude_corts]
ace_add = ace[cols_exclude_ace]

corts_match = corts.drop(cols_exclude_corts, axis=1)
ace_match = ace.drop(cols_exclude_ace, axis=1)

same_rows =  dataframe_difference(corts, ace, which='both')

#For each patient we need to evaluate the following scenarios:
# 1. Evaluate corticosteroids when ACE_ARBS = 0 
# 2. Evaluate corticosteroids and ACE_ARBS = 1
# 3. Evaluate ACE_ARBS and corticosteroids = 0
# 4. Evaluate ACE_ARBS and corticosteroids = 1 

df0_1 = same_rows[(same_rows['ACEI_ARBS']==0)& (same_rows['Z_presc_bin_x']==1)& (same_rows['CORTICOSTEROIDS']==1) & (same_rows['Z_presc_bin_y']==0)]
df0_0 = same_rows[(same_rows['ACEI_ARBS']==0)& (same_rows['Z_presc_bin_x']==0)& (same_rows['CORTICOSTEROIDS']==0) & (same_rows['Z_presc_bin_y']==0)]
df1_0 = same_rows[(same_rows['ACEI_ARBS']==1)& (same_rows['Z_presc_bin_x']==0)& (same_rows['CORTICOSTEROIDS']==0) & (same_rows['Z_presc_bin_y']==1)]
df1_1 = same_rows[(same_rows['ACEI_ARBS']==1)& (same_rows['Z_presc_bin_x']==1)& (same_rows['CORTICOSTEROIDS']==1) & (same_rows['Z_presc_bin_y']==1)]

print('We prescribe CORTS, without ACEs versus prescribe NO ACEs with corticosteroids', round(100*(df0_1['Y_presc_x']-df0_1['Y_presc_y']).abs().mean(),2),'%, n=', len(df0_1))    
print('We prescribe NO CORTS, without ACEs versus prescribe NO ACEs without corticosteroids', round(100*(df0_0['Y_presc_x']-df0_0['Y_presc_y']).abs().mean(),2),'%, n=', len(df0_0))
print('We prescribe ACE, without corticosteroids versus prescribe NO CORTS with ACE', round(100*(df1_0['Y_presc_x']-df1_0['Y_presc_y']).abs().mean(),2),'%, n=', len(df1_0))
print('We prescribe CORTS, with ACEs versus prescribe ACEs with corticosteroids', round(100*(df1_1['Y_presc_x']-df1_1['Y_presc_y']).abs().mean(),2),'%, n=', len(df1_1))     








