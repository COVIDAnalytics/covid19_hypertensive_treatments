os.chdir('git/covid19_calculator/calculator/')

#%% Load packages
import itertools

import evaluation.descriptive as ev
import evaluation.importance as imp

#%% Distribution by Feature

## Set paths
website_path = '/Users/hollywiberg/git/website/'
data_path = '/Users/hollywiberg/Dropbox (MIT)/COVID_risk/covid19_clean_data/'

for model_type, model_lab in itertools.product(['mortality','infection'],['with_lab','without_lab']):
    print("Model: %s, %s" %(model_type, model_lab))
    save_path = '../results/'+model_type+'/model_'+model_lab+'/'
    imp.feature_importance(model_type, model_lab, website_path, data_path, save_path)
    # summary = generate_summary(model_type, model_lab, title_mapping, website_path)
    # summary.to_csv(save_path+'descriptive_statistics.csv', index = False)

    

  
