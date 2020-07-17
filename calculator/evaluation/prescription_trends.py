import evaluation.treatment_utils as  u
import evaluation.descriptive_utils as d
import pandas as pd
import numpy as np
import itertools
from scipy import stats

# from julia import Julia           
# # jl = Julia(compiled_modules = False)
# jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
# from interpretableai import iai

#%% Set Problem Parameters
#Paths for data access

data_path = '../../covid19_treatments_data/'
results_path = '../../covid19_treatments_results/'
version_folder = "matched_limited_treatments_der_val_update/"
save_path = results_path + version_folder + 'summary/'
preload = True
matched = True
match_status = 'matched' if matched else 'unmatched'

treatment_list = ['All', 'Chloroquine_and_Anticoagulants','Chloroquine_and_Antivirals']
algorithm_list = ['lr','rf','cart','xgboost','oct','qda','gb']
# algorithm_list = ['lr','rf','cart','qda','gb']

#%%  Evaluate specific version
data_version = 'test' # in ['train','test','validation','validation_cremona','validation_hope']:
weighted_status = 'weighted'

#Read in the relevant data
X, Z, y = u.load_data(data_path+version_folder,'hope_hm_cremona_matched_cl_noncl_removed_train.csv',
                                split=data_version,matched=matched)

summary = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_summary_'+weighted_status+'.csv')
Z_presc = summary['Prescribe']

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

    

