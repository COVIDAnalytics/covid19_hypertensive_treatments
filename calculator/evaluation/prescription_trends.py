import evaluation.treatment_utils as  u
import evaluation.descriptive_utils as d
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import matplotlib.pyplot as plt

#%% Version-specific parameters

# version = 'matched_single_treatments_hope_bwh/'
# train_file = '_hope_matched_all_treatments_train.csv'
# data_list = ['train','test','validation_all','validation_partners']

version = 'matched_single_treatments_hypertension/'
train_file = '_hope_hm_cremona_matched_all_treatments_train.csv'
data_list = ['train','test','validation_all','validation_partners',
              'validation_hope','validation_hope_italy']

#%% General parameters

data_path = '../../covid19_treatments_data/'+version
        
preload = True
matched = True
match_status = 'matched' if matched else 'unmatched'

SEEDS = range(1, 2)
algorithm_list = ['rf','cart','oct','xgboost','qda','gb']
# prediction_list = ['COMORB_DEATH','OUTCOME_VENT','DEATH','HF','ARF','SEPSIS']
outcome = 'COMORB_DEATH'

treatment = 'ACEI_ARBS'
treatment_list = [treatment, 'NO_'+treatment]

training_set_name = treatment+train_file

results_path = '../../covid19_treatments_results/'
version_folder = version+str(treatment)+'/'+str(outcome)+'/'
save_path = results_path + version_folder + 'summary/'

training_set_name = treatment+train_file

#%% Compare prescriptions by algorithm

# summary_weight = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_weighted_t'+str(threshold)+'.csv')
# summary_weight['Algorithm'] = 'weighted'
# summary_weight.rename({'Unnamed: 0':'ID','AverageProbability':'Prescribe_Prediction'}, axis=1, inplace = True)

data_version = 'validation_all'
threshold = 0.05

summary_vote = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_no_weights_t'+str(threshold)+'.csv')
summary_vote['Algorithm'] = 'vote'
summary_vote.rename({'Unnamed: 0':'ID','AverageProbability':'Prescribe_Prediction'}, axis=1, inplace = True)

result = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods.csv')

result = pd.concat([result[['ID','Algorithm','Prescribe','Prescribe_Prediction']],
          # summary_weight[['ID','Algorithm','Prescribe','Prescribe_Prediction']],
          summary_vote[['ID','Algorithm','Prescribe','Prescribe_Prediction']]],
          axis=0, ignore_index = True)

comparison =  pd.DataFrame(columns = ['algorithm','prescription_count','prescription_prob','agreement_weighted', 'agreement_no_weights'])

for alg in result.Algorithm.unique():
    alg_presc = result.loc[result['Algorithm']==alg].set_index('ID')
    t_count = sum(alg_presc['Prescribe'] == treatment)
    prob = alg_presc['Prescribe_Prediction'].mean()
    # agreement_weighted = (alg_presc['Prescribe']==summary_weight['Prescribe']).mean()
    agreement_weighted = np.nan
    agreement_vote = (alg_presc['Prescribe']==summary_vote['Prescribe']).mean()
    comparison.loc[len(comparison)] = [alg, t_count, prob,
                                        agreement_weighted, agreement_vote]

comparison.to_csv(save_path+data_version+'_'+match_status+'_t'+str(threshold)+'_'+'agreement_byalgorithm.csv', index = False)




#%%  Evaluate specific version
data_version = 'validation_all' # in ['train','test','validation','validation_cremona','validation_hope']:
weighted_status = 'no_weights'
threshold = 0.05

#Read in the relevant data
X, Z, y = u.load_data(data_path,training_set_name,
                    split=data_version, matched=matched, prediction = outcome,
                    other_tx = False, replace_na = 'NO_'+treatment)

summary = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'_t'+str(threshold)+'.csv')
Z_presc = summary['Prescribe']
Y_presc = summary['AverageProbability']


X_test, Z_test, y_test = u.load_data(data_path,training_set_name,
                    split='test', matched=matched, prediction = outcome,
                    replace_na = 'NO_'+treatment)


X['Z'] = Z     
X['Z_presc'] = Z_presc
X['Y'] = y
X['Y_presc'] = Y_presc

## run this and replace png with _treatfreq.png to see frequency of prescription
X['Z_bin'] = Z.replace({treatment:1, 'NO_'+treatment: 0})
X['Z_presc_bin'] = Z_presc.replace({treatment:1, 'NO_'+treatment: 0})

bins= [0,40,55,70,110]
X['AgeGroup'] = pd.cut(X['AGE'], bins=bins,right=False)
bins= [0,0.8,2]
X['CreatinineGroups'] = pd.cut(X['CREATININE'], bins=bins,right=False)

#%% We would like to see how the treatments get distributed
cross_treatments = X[['Z_presc', 'Z']].groupby(['Z_presc', 'Z']).size().to_frame('size').reset_index()
cross_treatments = cross_treatments.pivot(index='Z_presc', columns='Z', values='size')

cross_treatments_norm = cross_treatments.div(cross_treatments.sum(axis=1), axis=0)
cross_treatments_norm['size'] = cross_treatments.sum(axis=1)
cross_treatments_norm.to_csv(save_path+data_version+'_'+match_status+'_'+weighted_status+'_cross_prescription_summary.csv', index = True)

#%%
#Plot by age the mortality rate
# data to plot

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
plot_byfeature('HYPERTENSION','plotHypertension')
plot_byfeature('BLOOD_PRESSURE_ABNORMAL_B','plotLowBP')
plot_byfeature('PCR_B','plotHighCRP')
plot_byfeature('DDDIMER_B','plotHighDD')
plot_byfeature('ANYHEARTDISEASE','plotHeartDisease')

#%% Table form
s_list = []
for ft in X.columns:
    if ((len(X[ft].unique()) == 2 )& (0 in X[ft].unique())) | (ft == "AgeGroup"):
        s = X.groupby(ft)[['Z_bin','Z_presc_bin']].mean()
        s.loc[:, 'Feature'] = ft
        s.index.name = 'Feature_Value'
        s.reset_index(inplace = True)
        s_list.append(s)

summ_all = pd.concat(s_list, axis=0)
summ_all['Change_Absolute'] = (summ_all['Z_presc_bin'] - summ_all['Z_bin'])
summ_all['Change_Relative'] = (summ_all['Z_presc_bin'] - summ_all['Z_bin'])/summ_all['Z_bin']
summ_all.to_csv(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_'+'prescription_rate_by_feature.csv', index = True)

summ_all = summ_all.query('Feature != "AgeGroup"')
summ_pivot = summ_all.pivot(index = 'Feature', columns = 'Feature_Value', values = 'Change_Relative')
summ_pivot = summ_pivot.add_prefix('Change_Relative_')
summ_pivot['Prescription_Difference'] = summ_pivot['Change_Relative_1.0'] - summ_pivot['Change_Relative_0.0']
summ_pivot.to_csv(save_path+data_version+'_'+match_status+'_'+weighted_status+'_t'+str(threshold)+'_'+'prescription_relative_change_summary.csv', index = True)

#%% Plots into panels

import string

SPINE_COLOR = 'gray'

bins= [0,40,55,70,110]
X['AgeGroup'] = pd.cut(X['AGE'], bins=bins,right=False)
bins= [0,0.8,2]
X['CreatinineGroups'] = pd.cut(X['CREATININE'], bins=bins,right=False)

plot_features = {
 'ANYHEARTDISEASE':'Heart Disease',
 'AF':'Atrial Fibrillation',
 'ANYLUNGDISEASE':'Lung Disease',
 'SAT02_BELOW92':'Low Oxygen Saturation',
 # 'HYPERTENSION':'Hypertension',
 'AgeGroup':'Age Group',
 'GENDER_MALE':'Gender (1=Male)'
 # 'BLOOD_PRESSURE_ABNORMAL_B':'Low Systolic BP',
}

n_display =len(plot_features)   # -1 because we remove age
n_cols = 2

u.latexify()

letters = ["\\textbf{(" + s + ")}" for s in list(string.ascii_lowercase)[0:n_display*n_cols+1]]


fig, axs = plt.subplots(int(np.ceil(n_display/n_cols)), n_cols,
                        figsize=(10, 10), constrained_layout=True
                        #  facecolor='w', edgecolor='k'
                        )

axs = axs.ravel()  # flatten

# Reference ranges

for i, (k, ft) in enumerate(plot_features.items()):
    idx = i
    print(ft)
    
    tbl_z = X.groupby(k)[['Z_bin','Z_presc_bin']].mean()
    tbl_z.rename(index={1:'Yes',0:'No'}, inplace = True)
    tbl_z.index.name = ft
    tbl_z.plot(ax = axs[idx], kind = 'bar', rot=0, legend = False, color =['#003087','#E87722'])
    
    axs[idx].text(0.05, 0.9, letters[idx],
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs[idx].transAxes,
        weight='bold',
    )

    # tbl = X.groupby(k)[['Y','Y_presc']].mean()
    # tbl.index.name = ft
    # tbl.plot(ax = axs[idx+1], kind = 'bar', rot=0, legend = False, color =['#003087','#E87722'])

    # axs[idx+1].text(0.05, 0.9, letters[idx+1],
    #     horizontalalignment='center',
    #     verticalalignment='center',
    #     transform=axs[idx+1].transAxes,
    #     weight='bold',
    # )


# axs[0].set_title('Prescription Rate', weight='bold')
# axs[1].set_title('Mortality Rate', weight='bold')


handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, ['Observed', 'Recommended'],
           # bbox_to_anchor=(1,0), 
           loc = 'lower right', borderaxespad=0.1)
fig.tight_layout()
fig.subplots_adjust(bottom = 0.06)

fig.savefig(save_path+data_version+'_t'+str(threshold)+'_feature_plot.pdf', bbox_inches='tight')
 
plt.close()

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
       #  'CORTICOSTEROIDS', 'INTERFERONOR', 'TOCILIZUMAB',
       # 'ANTIBIOTICS', 'ACEI_ARBS', 
       'GENDER_MALE', 'RACE_BLACK','RACE_CAUC', 'RACE_LATIN',
       'RACE_ORIENTAL'],
                'numeric':['AGE','MAXTEMPERATURE_ADMISSION','CREATININE', 'SODIUM', 'LEUCOCYTES', 'LYMPHOCYTES',
       'HEMOGLOBIN', 'PLATELETS'],
                'multidrop':[]}

# if ~np.any(X.columns.str.contains('RACE_OTHER')):
#     features['categorical'].remove('RACE_OTHER')

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
