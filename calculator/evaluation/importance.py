import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

def feature_importance(model_type, model_lab, website_path, data_path, save_path):
    assert model_type in('mortality','infection'), "Invalid outcome"
    assert model_lab in('with_lab','without_lab'), "Invalid lab specification"

    ## Load model corresponding to *model_type* and *model_lab*.

    with open(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.pkl', 'rb') as file:
        model_file = pickle.load(file)
        
    model = model_file['model']

    ## Load data: to be replaced once we store X_train and X_test. Currently no imputation
    
    data = pd.read_csv(data_path+model_type+"_"+model_lab+"/train.csv")
    if model_type == "mortality":
        X = data.drop(["Unnamed: 0", "Outcome"], axis=1, inplace = False)
        y = data["Outcome"]
    else:
        X = data.drop(["NOSOLOGICO","Swab"], axis=1, inplace = False)
        y = data["Swab"]

    ## Calculate SHAP values (for each observation x feature)
    explainer = shap.TreeExplainer(model);
    shap_values = explainer.shap_values(X);

    ## Summarize SHAP values across all features
    # This acts as an alterative to the standard variable importance plots. Higher SHAP values translate to higher probability of mortality.
    shap.summary_plot(shap_values, X, show=False,max_display=100)
    f = plt.gcf()
    f.savefig(save_path+'summary_plot.jpg', bbox_inches='tight')
    # f.savefig(website_path+'assets/risk_calculators/'+model_type+'/model_'+model_lab+'.jpg', bbox_inches='tight')
    plt.clf() 
 
    ## Deep-dive into individual features
    # For a given feature, see how the SHAP varies across its possible values. 
    # The interaction_index lets you choose a secondary index to visualize.
    # If omitted, it will automatically find the variable with the highest interaction.

    for i in X.columns:
        shap.dependence_plot(i, shap_values, X,show=False)
        f = plt.gcf()
        f.savefig(save_path+'dependence_plot_'+i+'.jpg', bbox_inches='tight')
        plt.clf()