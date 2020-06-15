import os
#  os.chdir('git/covid19_calculator/calculator/')

#%% Load packages
import itertools

# import evaluation.descriptive as ev
import evaluation.importance as imp

website_path = '../../website/'
data_path = '../../covid19_clean_data/xgboost/'

# subgroups = {'Age < 55': 'Age_below55',
#     'Age >= 55 & Age < 80': 'Age_55to79',
#     'Age >= 80': 'Age_atleast80',
#     'Gender ==  0': 'Male',
#     'Gender == 1': 'Female'}

for model_type, model_lab in itertools.product(['infection'],
                                               ['with_lab', 'without_lab']):
    if model_type == 'mortality':
        data_path = '../../covid19_clean_data/xgboost/'
    else: 
        data_path = '../../covid19_clean_data/pnas_models/xgboost/'
    # for model_type, model_lab in itertools.product(['mortality'],['without_lab']):
    print("Model: %s, %s" % (model_type, model_lab))
    # save_path = '../results/'+model_type+'/model_'+model_lab+'/'
    save_path = website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'_'
    imp.feature_importance_website(model_type, model_lab, website_path, data_path,
                           save_path, feature_limit=10)


    #  imp.feature_importance_website(model_type, model_lab, website_path, data_path, save_path, title_mapping_summary,
    #                                 feature_limit=10)





    # for s in subgroups.keys():
    #     print("Subgroup: %s" % s)
    #     imp.feature_importance(model_type, model_lab, website_path, data_path, save_path, title_mapping_summary,
    #                            latex = True, feature_limit = 10, dependence_plot = False,
    #                            data_filter = s, suffix_filter = subgroups[s])
