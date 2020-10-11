import evaluation.treatment_utils as  u
import evaluation.descriptive_utils as d
import pandas as pd
import numpy as np
import itertools
from scipy import stats
import matplotlib.pyplot as plt

from julia import Julia
jl = Julia(sysimage='/home/hwiberg/software/julia-1.2.0/lib/julia/sys_iai.so')
from interpretableai import iai


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

data_version = 'train'
threshold = 0.05

train_X, Z, y = u.load_data(data_path,training_set_name,
                    split=data_version, matched=matched, prediction = outcome,
                    other_tx = False, replace_na = 'NO_'+treatment)

df_result = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods_benefit.csv')
benefit = df_result.groupby('ID').agg({'Benefit':'mean'})['Benefit']


### Run Model
grid = iai.GridSearch(
    iai.OptimalTreeRegressor(
        random_seed=1,
    ),
    max_depth=range(3, 8),
)

grid.fit(X, benefit)

lnr = grid.get_learner()
lnr.write_html(save_path+data_version+'_benefit_tree.html')

grid.score(X, benefit['Benefit'], criterion='mse')

for data_version in ['train','test','validation_all']:
	X, Z, y = u.load_data(data_path,training_set_name,
	                    split=data_version, matched=matched, prediction = outcome,
	                    other_tx = False, replace_na = 'NO_'+treatment)
	X = X.reindex(train_X.columns, axis = 1, fill_value = 0)
	df_result = pd.read_csv(save_path+data_version+'_'+match_status+'_bypatient_allmethods_benefit.csv')
	benefit = df_result.groupby('ID').agg({'Benefit':'mean'})['Benefit']
	mse = lnr.score(X, benefit, criterion = 'mse')
	print("Data = ", data_version, ": MSE = ", round(mse, 3))


